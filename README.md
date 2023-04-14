# Decoupling MaxLogit for Out-of-Distribution Detection

>In machine learning, it is often observed that standard training outputs anomalously high confidence for both in-distribution (ID) and out-of-distribution (OOD) data. Thus, the ability to detect OOD samples is critical to the model deployment. An essential step for OOD detection is post-hoc scoring. MaxLogit is one of the simplest scoring functions which uses the maximum logits as OOD score. To provide a new viewpoint to study the logit-based scoring function, we reformulate the logit into cosine similarity and logit norm and propose to use MaxCosine and MaxNorm. We empirically find that MaxCosine is a core factor in the effectiveness of MaxLogit. And the performance of MaxLogit is encumbered by MaxNorm. To tackle the problem, we propose the Decoupling MaxLogit (DML) for flexibility to balance MaxCosine and MaxNorm. To further embody the core of our method, we extend DML to DML+ based on the new insights that fewer hard samples and compact feature space are the key components to make logit-based methods effective. We demonstrate the effectiveness of our logit-based OOD detection methods on CIFAR-10, CIFAR-100 and ImageNet and establish state-of-the-art performance.

## Description

Official PyTorch implementation of "Decoupling MaxLogit for Out-of-Distribution Detection"

## Data

```
|--data
|--|--cifar-100-python
|--|--ood_test
|--|--|--svhn
|--|--|--places365
|--|--|--LSUN_R
|--|--|--LSUN_C
|--|--|--iSUN
|--|--|--dtd
```
For the Out-of-distribution dataset, please refer to [KNN](https://github.com/deeplearning-wisc/knn-ood).


## How to inference

```python
python test_DML+.py cifar100 

--MCF cifar100_wrn_Focal_NN_epoch_199.pt 

--MNC cifar100_wrn_Center_NN_epoch_199.pt
```



## Results

```
Files already downloaded and verified
loading models
loading models

Textures Detection
                                Ours
FPR95:                  41.87   +/- 0.64
AUROC:                  91.11   +/- 0.20
AUPR:                   97.87   +/- 0.05


SVHN Detection
                                Ours
FPR95:                  17.58   +/- 0.39
AUROC:                  97.21   +/- 0.08
AUPR:                   99.43   +/- 0.02


LSUN-C Detection
                                Ours
FPR95:                  17.48   +/- 0.26
AUROC:                  97.23   +/- 0.05
AUPR:                   99.42   +/- 0.01


LSUN-R Detection
                                Ours
FPR95:                  27.72   +/- 0.66
AUROC:                  94.51   +/- 0.15
AUPR:                   98.75   +/- 0.04


iSUN Detection
                                Ours
FPR95:                  29.89   +/- 1.06
AUROC:                  93.84   +/- 0.24
AUPR:                   98.58   +/- 0.06


Places365 Detection
                                Ours
FPR95:                  71.70   +/- 0.28
AUROC:                  82.45   +/- 0.48
AUPR:                   95.69   +/- 0.20


Mean Test Results
                                Our DML
FPR95:                  34.37
AUROC:                  92.72
AUPR:                   98.29

```

