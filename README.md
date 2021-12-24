# Leveraging World Events to Predict E-Commerce Consumer Demand under Anomaly

## Dan Kalifa, Uriel Singer, Ido Guy, Guy D. Rosin, and Kira Radinsky. 

_In Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining (WSDM '22), February 21--25, 2022, Tempe, AZ, USA._


> **Abstract:** Consumer demand forecasting is of high importance for many e-commerce applications, including supply chain optimization, advertisement placement, and delivery speed optimization. 
However, reliable time series sales forecasting for e-commerce is difficult, especially during periods with many anomalies, as can often happen during pandemics, abnormal weather, or sports events. Although many time series algorithms have been applied to the task, prediction during anomalies still remains a challenge. 
In this work, we hypothesize that leveraging external knowledge found in world events can help overcome the challenge of prediction under anomalies. We mine a large repository of 40 years of world events and their textual representations. 
Further, we present a novel methodology based on transformers to construct an embedding of a day based on the relations of the day’s events. 
Those embeddings are then used to forecast future consumer behavior. 
We empirically evaluate the methods over a large e-commerce products sales dataset, extracted from eBay, one of the world’s largest online marketplaces. We show over numerous categories that our method outperforms state-of-the-art baselines during anomalies. We contribute the code and data to the community for further research.

![GitHub Logo](https://user-images.githubusercontent.com/57223242/119711281-04b85e80-be68-11eb-8907-1649b3cc847e.png)

This repository provides a reference implementation of GAN-Event and forecasting baselines as described in the paper.

## Setup

This code was tested with `Python 3.8`.

1. Setup a new virtual environment by running:
```shell script
python3 -m venv venv/
source venv/bin/activate
```

2. Run:
```shell script
pip3 install -r requirements.txt
```

3. Add the virtual environment:
```shell script
python3 -m ipykernel install --user --name=venv
```

## Usage
In order to use the GAN-Event LSTM model or one of the baselines,
you should first create the day embeddings using the GAN-Event model, 
and then run the GAN-Event LSTM or a baseline for the forecasting task.
### GAN-Event (Day Embeddings Generation)
Run the following command:
```shell script
python3 src/run.py
```
In addition, you can edit the `hparams` variable to control the different
parameters for the GAN-Event, in the file `src/run.py`.

### Forecasting Models
Run the following command and open `experiments.ipynb`.
Then, chose `venv` as the kernel for the notebook: 
```shell script
cd src
jupyter notebook
```

## License
This project is licensed under the MIT license. 
Check [LICENSE](LICENSE) for more information.


## Data
- **World-event dataset**: The world event dataset is accessible in the "data" folder. 
- **E-commerce dataset**: As we cannot disclose actual sale values due to business sensitivity, we supply a mock time series as an alternative.
The dataset can be found in the notebook "experiments" in the "src" folder.

