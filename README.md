# Out-of-Distribution Detection Based on Distance Metric Learning

## Introduction
This project is for the paper "Out-of-Distribution Detection Based on Distance Metric Learning". Some codes are from [ODIN](https://github.com/facebookresearch/odin) and [Mahalanobis-based OOD Detection](https://github.com/pokaxpoka/deep_Mahalanobis_detector/).

## Requirements
- Python 3.7
- Pytorch 1.5

## Experiments
We provide two expeiments of ood detection 
(In-distribution:MNIST/Out-Distribution:FashionMNIST and In-distribution:FashionMNIST/Out-Distribution:MNIST)

### Model training 
### (Resnet34, Siamise with ResNet34, Triplet with ResNet34) 
- Run all code in network_trainner.ipynb
- You can see trained models in trained_models

### Out-of-distribution Detection 
### (Baseline Method, ODIN, Mahalanobis-based Method, DML-based Method (ours) )
- Run all code in MAIN_1c.ipynb
- All results will be printed in the each cell of Main_1c notebook  

