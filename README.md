# Interpretable Multivariate Time Series Forecasting with Temporal Attention Convolutional Neural Networks

This repository contains the official implementation for the models described in [Interpretable Multivariate Time Series Forecasting with Temporal Attention Convolutional Neural Networks](https://research.vu.nl/en/publications/interpretable-multivariate-time-series-forecasting-with-temporal-).

If you find this work helpful in your research, consider citing our paper:

```
@INPROCEEDINGS{pantiskas2020tacn,  
author={L. {Pantiskas} and K. {Verstoep} and H. {Bal}},  
booktitle={2020 IEEE Symposium Series on Computational Intelligence (SSCI)},   
title={Interpretable Multivariate Time Series Forecasting with Temporal Attention Convolutional Neural Networks},   
year={2020},  
volume={},  
number={},  
pages={1687-1694},  
doi={10.1109/SSCI47803.2020.9308570}}
```

## Requirements

The code is written in Python 3.7.7 and has the following dependencies for the training and evaluation notebooks:
* tensorflow==2.1.0
* tensorflow-addons==0.8.2
* tqdm==4.46.0
* seaborn==0.10.1
* scipy==1.4.1
* scikit-learn==0.22.1
* pygam==0.8.0
* pydotplus==2.0.2
* pandas==1.0.3
* numpy==1.18.1
* matplotlib==3.1.3
* ipywidgets==7.5.1
* eli5==0.10.1 

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train a model from the paper, either execute the suitable training Jupyter notebook or convert it to Python script with:
```
jupyter nbconvert --to script NotebookName.ipynb
```
and run it with:
```train
python NotebookName.py
```
The scripts require interactive input.

## Evaluation

To evaluate a model from the paper, it is suggested to execute the suitable evaluation Jupyter notebook in order to easily view the resulting graphs apart from the metrics. 
The evaluation notebooks load the pre-trained models in their respective folder by default, so if you want to load your own trained models you will have to edit the location in the suitable cell in the notebooks. 
## Pre-trained Models

The pre-trained models with the parameters described in the paper are in their respective Weights folders.
Each folder contains 10 _.h5_ weight files, which correspond to 10 trained instances of the same model with different seeds.

## Results

Our model is evaluated in the task of multi-step forecasting on the following datasets:
* [Air Quality variables](https://archive.ics.uci.edu/ml/datasets/Air+Quality)
* [Water quality variables](https://www.data.qld.gov.au/dataset/ambient-estuarine-water-quality-monitoring-data-near-real-time-sites-2012-to-present-day)


Below we can see the performance metrics of our method and the baselines:

|**Air Quality Dataset** |                 |                 ||
|----------------|-------|-----------------|-----------------|
|                |       | Baseline        | Proposed Model  |
| CO             | RMSE  | 1.56 +- 0.10    | 1.32 +- 0.04    |
|                | MAE   | 1.1 +- 0.07     | 0.93 +-0.03     |
| Benzene        | RMSE  | 7.59 +- 0.31    | 7.02 +- 0.52    |
|                | MAE   | 5.69 +- 0.27    | 4.85 +- 0.25    |
| NOX            | RMSE  | 244.43 +- 22.63 | 214.49 +- 10.31 |
|                | MAE   | 180.62 +- 15.56 | 152.25 +- 7.46  |
| NO2            | RMSE  | 52.35 +- 13.29  | 48.57 +- 1.53   |
|                | MAE   | 41.83 +- 12.66  | 36.94 +- 1.18   |
| Training time (sec)  |       | 477 +- 7        | 524 +- 11       |
|**Water Quality Dataset**   |                 |                 |
| Temperature    | RMSE  | 0.59 +- 0.07    | 0.50 +- 0.02    |
|                | MAE   | 0.39 +- 0.04    | 0.33 +- 0.02    |
| Training time (sec)  |       | 4554 +- 143     | 5310 +- 262     | 

## Training and evaluation random seeds

| Air Quality Baseline |                                      | Air Quality Proposed Model |                                      | Water Quality Baseline |                                      | Water Quality Proposed Model |                                      |
|----------------------|--------------------------------------|----------------------------|--------------------------------------|------------------------|--------------------------------------|------------------------------|--------------------------------------|
| Seed                 | Experiment Id                        | Seed                       | Experiment Id                        | Seed                   | Experiment Id                        | Seed                         | Experiment Id                        |
| 2650481              | 3af068ef-70de-4a71-86e2-e3a07b2a8b20 | 2650491                    | 36c93c9e-3105-47f5-9e85-5beb23e94e68 | 2639467                | 5aa4c32f-3a25-49b8-997b-1bdb4af37939 | 2639478                      | b4b94f1c-6d65-459c-adb8-389cbd0e97c5 |
| 2650482              | 60a8f6f3-f528-4e00-94ad-b4a963457319 | 2650492                    | 25d6d2d6-2265-4431-88a2-bc49831eb8c6 | 2639468                | 80c4e454-726a-405f-9e83-f524f20939a9 | 2639479                      | 3a4e66b9-5244-4bc0-9076-a3eca47d5fe4 |
| 2650483              | 4ed1d1c2-476d-4741-a980-55e830c8c79b | 2650493                    | 492bc644-16a5-41b9-b226-20c2dbefce13 | 2639469                | d1ef568b-174d-4f9e-97b6-6813c4270a5c | 2639480                      | 78337ffe-d26e-4e7b-9497-0bd0cb9c2974 |
| 2650484              | 01e02876-b021-4215-9111-4122b3519778 | 2650494                    | 6cd2a5a8-35f7-431a-a671-7c54aaf36ac3 | 2639470                | 317ff62b-8bcd-40c1-811d-b5ba67d2ab7c | 2639481                      | c6c30ea5-33ab-4900-bc0e-d2fd98c1bcab |
| 2650485              | 8a62a6c5-8570-474a-aa4a-a758573778bb | 2650495                    | a54c34b0-69ee-43d2-b900-61e0fb0d8228 | 2639471                | 83f3dcf9-491a-4c20-8c39-064c65163d35 | 2639482                      | db7b10b4-4b52-4a26-97ca-56186eb924a1 |
| 2650486              | 6e1973ff-8fcc-47ce-a28e-41c7daa661d2 | 2650496                    | f43a22dd-ac6f-4db9-a5b4-17d4a7ef9c5b | 2639472                | cef6c672-f8eb-4916-9221-82051633b99c | 2639483                      | f08332bc-d654-4219-a7c1-e0e6854fb2b5 |
| 2650487              | 3a6c674e-abcf-4879-bd8c-3f893769c9f0 | 2650497                    | f62368ea-9cc9-4fd6-8f7a-a28b6bf3b4b4 | 2639473                | 7e31a1d4-5fed-4924-939e-b2cde7fdf96b | 2639484                      | 7fba0620-13b4-4306-8a2a-6c82b025a8fe |
| 2650488              | a5697c28-0c05-466b-a17a-5de349cc156a | 2650498                    | 7c4c024c-f347-4027-9c19-1264357ec174 | 2639474                | 350cbc49-a71e-4336-839b-2be9ed889eca | 2639485                      | 67f95b23-e4a5-4b5b-9512-6c65e3918545 |
| 2650489              | 35801255-51d1-42a3-86fe-b3efd7094b58 | 2650499                    | 393c6468-7a41-455f-8fdd-70cd674b3b11 | 2639475                | 6847ae60-80d8-4580-bb3e-10ee1e9ccaf3 | 2639486                      | b82aad09-1631-4540-9058-3c6eff69511e |
| 2650490              | db1722ce-f7ab-499b-b865-6165fe73cc3b | 2650500                    | 510e465d-c041-4fb3-b76c-f514fde218ae | 2639476                | b6108ad8-4a30-49d8-aaad-4d8d1c40ad37 | 2639487                      | bcb62682-39db-483e-aa24-b9afa625d99e |

The random seed that was used for all evaluation experiments is **45112**.
## Visualization examples
**Temperature attention distribution**
![](Temp_degC_abs_att.gif)
**pH attention distribution**
![](pH_abs_att.gif)
**Dissolved oxygen attention distribution**
![](DO_mg_abs_att.gif)