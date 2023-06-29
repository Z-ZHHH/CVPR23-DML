# Decoupling MaxLogit for Out-of-Distribution Detection

>In machine learning, it is often observed that standard training outputs anomalously high confidence for both in-distribution (ID) and out-of-distribution (OOD) data. Thus, the ability to detect OOD samples is critical to the model deployment. An essential step for OOD detection is post-hoc scoring. MaxLogit is one of the simplest scoring functions which uses the maximum logits as OOD score. To provide a new viewpoint to study the logit-based scoring function, we reformulate the logit into cosine similarity and logit norm and propose to use MaxCosine and MaxNorm. We empirically find that MaxCosine is a core factor in the effectiveness of MaxLogit. And the performance of MaxLogit is encumbered by MaxNorm. To tackle the problem, we propose the Decoupling MaxLogit (DML) for flexibility to balance MaxCosine and MaxNorm. To further embody the core of our method, we extend DML to DML+ based on the new insights that fewer hard samples and compact feature space are the key components to make logit-based methods effective. We demonstrate the effectiveness of our logit-based OOD detection methods on CIFAR-10, CIFAR-100 and ImageNet and establish state-of-the-art performance.

## Description

CVPR2023  Official PyTorch implementation of "Decoupling MaxLogit for Out-of-Distribution Detection".

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

### DML+
```python
python test_DML+.py cifar100 

--MCF cifar100_wrn_Focal_NN_epoch_199.pt 

--MNC cifar100_wrn_Center_NN_epoch_199.pt
```
### DML
#### Formal
It is easy to implement DML, here is a brief one.
First, collect the score just like the MaxLogit or MSP,
```
for batch_idx, examples in enumerate(loader):
    data, target = examples[0], examples[1]
    logits, features = your_net(data)

    all_score1 = np.max(to_np(logits), axis=1)      # calculate the cosine similarity          
    all_score2 = features.norm(2, dim=1).numpy()    # calculate the norm of features

    _score1.append(all_score1)
    _score2.append(all_score2)
_score1 = concat(_score1)
_score2 = concat(_score2)
```
Then the DML ood score is calculated by ood_score = _score1 + <img src="https://latex.codecogs.com/svg.image?&space;\lambda&space;" title=" \lambda " /> _score2.

The above net classifier and the features are normalized and output cosine similarity. 
#### Simplified
If you don't want to modify the network forward code. The DML could also be calculated as follows.

```
for batch_idx, examples in enumerate(loader):
    data, target = examples[0], examples[1]
    logits, features = your_net(data)

    all_score1 = np.max(to_np(logits), axis=1)      # Logits is just logit without any modification         
    all_score2 = features.norm(2, dim=1).numpy()    # calculate the norm of features

    _score1.append(all_score1)
    _score2.append(all_score2)
_score1 = concat(_score1)
_score2 = concat(_score2)
```
Then DML ood score is calculated by ood_score = _score1 + <img src="https://latex.codecogs.com/svg.image?&space;\lambda&space;" title=" \lambda " /> _score2. 
This implementation only requires to output the features without any model modification. However, it ignores the effect of classifier norms and may have differences from the first one.

## Results 
**DML+**
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


## Citation
If you find this useful in your research, please consider citing:
```
@InProceedings{Zhang_2023_CVPR,
    author    = {Zhang, Zihan and Xiang, Xiang},
    title     = {Decoupling MaxLogit for Out-of-Distribution Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {3388-3397}
}
```

## Acknowledgement
The repo is built on [LogitNorm](https://github.com/hongxin001/logitnorm_ood) and [ViM](https://github.com/haoqiwang/vim). Thanks for their wonderful work.

