# Conventional neural network performance comparison
This repository contains the data and code for the classification performance of conventional neural networks which is compared with Distance Encoding Biomorphic-Informational Neural Network (DEBI-NN) as part of the publication [DEBI-NN: Distance-Encoding Biomorphic-Informational Neural Networks for Minimizing the Number of Trainable Parameters] (in submission). 

## Requirements and System specifications
All experiments were performed on Linux Ubuntu 20.04.4 LTS (64-bit) using Python 3.9.7. The specific requirements are listed in the `requirements.txt`.

## Datasets
The already preprocessed datasets are listed in the `data/` folder and are open-access as listed below.

### 1. Heart failure clinical records Data Set

Source: https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records

Citation Request:

[Davide Chicco, Giuseppe Jurman: "Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone". BMC Medical Informatics and Decision Making 20, 16 (2020).](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5)

### 2. Mammographic Mass Data Set

Source: http://archive.ics.uci.edu/ml/datasets/Mammographic+Mass

Citation Request:

[M. Elter, R. Schulz-Wendtland and T. Wittenberg (2007)
The prediction of breast cancer biopsy outcomes using two CAD approaches that both emphasize an intelligible decision process.
Medical Physics 34(11), pp. 4164-4172](https://aapm.onlinelibrary.wiley.com/doi/full/10.1118/1.2786864)

### 3. and 4. BreastMNIST and PneumoniaMNIST

Source: https://medmnist.com

Citation Request:

[Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni. "MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification". arXiv preprint arXiv:2110.14795, 2021.](https://arxiv.org/pdf/2110.14795.pdf)

Jiancheng Yang, Rui Shi, Bingbing Ni. "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis". IEEE 18th International Symposium on Biomedical Imaging (ISBI), 2021.

## Run comparison
To run the comparison on all datasets as described in the paper, run:
```
python comparison.py
```
The results will be saved in a `results/` folder.
