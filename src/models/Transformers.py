from torch.nn import TransformerEncoderLayer
from torch.nn import TransformerEncoder


class MyTransformerEncoderLayer(TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # Changes from the original code: (1) attn_weights from self.self_attn.
        # see https://pytorch.org/docs/1.6.0/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer.forward
        src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights


class MyTransformerEncoder(TransformerEncoder):
    def forward(self, src, mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        weights_matrix_list = []
        output = src

        for mod in self.layers:
            output, attn_weights = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            weights_matrix_list.append(attn_weights)

        if self.norm is not None:
            output = self.norm(output)

        return output, weights_matrix_list




