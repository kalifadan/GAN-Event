import torch
import pandas as pd
from src.models.EventsModule import EventsModule
from src.models.Discriminator import DiscriminatorModel


class AdversarialEventsModule(EventsModule):
    def __init__(self, hparams, emb_dict, gen_model):
        super().__init__(hparams, emb_dict)
        self.bce_loss = torch.nn.BCELoss()
        self.wd_dis = hparams['weight_decay_dis'] if 'weight_decay_dis' in hparams else 0
        self.lr_dis = hparams['lr_dis'] if 'lr_dis' in hparams else 0.01
        self.g_lmb = hparams['g_lmb'] if 'g_lmb' in hparams else 1
        self.d_lmb = hparams['d_lmb'] if 'd_lmb' in hparams else 1
        self.generator = gen_model
        self.discriminator = DiscriminatorModel(out_dim=self.generator.out_dim, hparams=hparams)
        self.batch_dict, self.lengths, self.x, self.real_days, self.generated_days = None, None, None, None, None
        self.saving_results = hparams['saving_results'] if 'saving_results' in hparams else False
        self.results_prefix = hparams['results_prefix'] if 'results_prefix' in hparams else 'Cosine_HD_'

    @staticmethod
    def adversarial_accuracy(y_pred, y_true):
        values = (y_pred >= 0.5).squeeze(dim=1).int()
        true_values = y_true.squeeze(dim=1)
        assert values.size(0) > 0 and values.size(0) == true_values.size(0)
        return torch.eq(values, true_values).sum().item() / values.size(0)

    def adversarial_loss(self, y_hat, y):
        return self.bce_loss(y_hat, y)

    def forward(self, x, test_mode=False):
        return self.generator(x, test_mode)

    def step(self, batch: dict, optimizer_idx: int, name: str = 'loss', batch_idx: int = None):
        results, loss, dis_loss, gen_loss = {}, None, None, 0

        # train generator
        if optimizer_idx == 0:
            self.batch_dict, self.lengths = batch
            self.x = self.generator.embedding_model(self.batch_dict)
            self.real_days = self.x.detach().clone()
            self.generated_days, mask_indices = self((self.x, self.lengths))  # generate counterfactual days

            for row_idx, row_len in enumerate(self.lengths):
                masked_predict = self.generated_days[row_idx][mask_indices[row_idx]]
                masked_target = self.real_days[row_idx][mask_indices[row_idx]]
                gen_loss += self.gen_loss(masked_predict, masked_target)
            gen_loss /= self.x.size(0)

            # restore the unmasked events only
            self.generated_days[~mask_indices] = self.real_days[~mask_indices]

            # Here the optimizer is not the discriminator's optimizer so it's legal to call the discriminator forward
            y_pred = self.discriminator((self.generated_days, self.lengths))    # make predictions
            y_true = torch.ones_like(y_pred)    # ground truth - all fake
            dis_loss = self.adversarial_loss(y_pred, y_true)    # adversarial loss is binary cross-entropy
            acc = AdversarialEventsModule.adversarial_accuracy(y_pred.detach(), y_true.detach())

            loss = self.g_lmb * gen_loss + self.d_lmb * dis_loss

            results[f'generator/{name}_loss'] = loss
            results[f'generator/{name}_discriminator_accuracy'] = acc
            results[f'generator/{name}_generator_loss'] = gen_loss
            results[f'generator/{name}_discriminator_loss'] = dis_loss

        # train discriminator & measuring discriminator's ability to classify real from generated samples
        if optimizer_idx == 1:
            y_pred = self.discriminator((self.real_days, self.lengths))
            y_true = torch.ones_like(y_pred)
            real_loss = self.adversarial_loss(y_pred, y_true)
            real_acc = AdversarialEventsModule.adversarial_accuracy(y_pred.detach(), y_true.detach())

            y_pred_fake = self.discriminator((self.generated_days.detach(), self.lengths))
            y_fake = torch.zeros_like(y_pred_fake)
            fake_loss = self.adversarial_loss(y_pred_fake, y_fake)
            fake_acc = AdversarialEventsModule.adversarial_accuracy(y_pred_fake.detach(), y_fake.detach())

            loss = (real_loss + fake_loss) / 2     # discriminator loss is the average of these losses (both BCE)

            results[f'discriminator/{name}_discriminator_real_loss'] = real_loss
            results[f'discriminator/{name}_discriminator_fake_loss'] = fake_loss
            results[f'discriminator/{name}_discriminator_loss'] = loss
            results[f'discriminator/{name}_discriminator_fake_acc'] = fake_acc
            results[f'discriminator/{name}_discriminator_real_acc'] = real_acc

        for k, v in results.items():
            self.log(k, v)

        return loss, results

    def training_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None):
        loss, results = self.step(batch, optimizer_idx, name='train', batch_idx=batch_idx)
        return {'loss': loss, 'prog': results}

    def validation_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None):
        results = {}
        for i in range(len(self.optimizers())):
            results.update(self.step(batch, i, name='val', batch_idx=batch_idx)[1])
        return results

    def test_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None):
        results = {}
        for i in range(len(self.optimizers())):
            results.update(self.step(batch, i, name='test', batch_idx=batch_idx)[1])
        return results

    def get_embeddings_generator(self, data_loader, dates):
        embeddings = None
        for batch in data_loader:
            batch_dict, lengths = batch
            x = self.generator.embedding_model(batch_dict).cuda()
            out, weights_matrix_list = self.generator((x, lengths), test_mode=True)
            new_out = torch.zeros((out.shape[0], out.shape[2]), dtype=torch.float, device=self.device)
            for row_idx, row_len in enumerate(lengths):
                masked_event = self.get_masked_embedding()
                for event_idx in range(row_len):
                    cur_x = batch_dict['embeddings'][row_idx][:row_len].float().cuda()
                    if row_len > 1:
                        cur_x[event_idx] = masked_event
                    cur_out, _ = self.generator((cur_x.unsqueeze(0), torch.tensor([row_len])), test_mode=True)
                    cur_out = cur_out[0][event_idx]
                    new_out[row_idx] += cur_out
                new_out[row_idx] /= row_len
            embeddings = new_out if embeddings is None else torch.cat([embeddings, new_out])
        df = pd.DataFrame(list(zip(dates[:len(embeddings)], embeddings.tolist())), columns=['date', 'embeddings'])
        return df

    def on_test_end(self):
        print("Test end function:")
        if self.saving_results:
            print("Start testing GAN")
            embedding_train_dates, train_dates, val_dates, test_dates = self.trainer.datamodule.get_dates()
            results_prefix = self.results_prefix
            df_train = self.get_embeddings_generator(self.trainer.datamodule.embedding_train_dataloader(), embedding_train_dates)
            df_test = self.get_embeddings_generator(self.trainer.datamodule.test_dataloader(), test_dates)
            df_train.to_pickle("../gan_embeddings/" + results_prefix + "causal_mean_train_gan_embeddings.pkl")
            df_test.to_pickle("../gan_embeddings/" + results_prefix + "causal_mean_test_gan_embeddings.pkl")

    def configure_optimizers(self):
        optimizer1 = torch.optim.AdamW(self.generator.parameters(), lr=self.lr_gen, weight_decay=self.wd_gen)
        optimizer2 = torch.optim.AdamW(self.discriminator.parameters(), lr=self.lr_dis, weight_decay=self.wd_dis)
        scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=50, gamma=0.95, last_epoch=-1)
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=50, gamma=0.95, last_epoch=-1)
        return [optimizer1, optimizer2], [scheduler1, scheduler2]
