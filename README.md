# Out-of-Distribution Detection Based on Distance Metric Learning

## Introduction
This project is for the paper "Out-of-Distribution Detection Based on Distance Metric Learning". Some codes are from [Mahalanobis-based OOD Detection](https://github.com/pokaxpoka/deep_Mahalanobis_detector/).

This repository contains source code of experiments used in paper of Out-of-Distribution Detection Based on Distance Metric Learning (SMA 2020 in Korea).

## Requirements

- Python 3.7
- Pytorch 1.5

## Experiments
### in-distribution : MNIST / Out-Distribution : FashionMNIST
Train model (Resnet34, Siamise with ResNet34, Triplet with ResNet34) 
- Run all code in network_trainner.ipynb
- You can see trained models in trained_models

Out-of-distribution Detection (Baseline Method, ODIN, Mahalanobis-based Method, DML-based Method (ours) )
- Run all code in MAIN_1c.ipynb
- All results will be printed in the each cell of Main_1c notebook  
