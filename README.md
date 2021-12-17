# Requirement
The codes are implemented in tensorflow--1.14 or 1.15 under the interpreter python3.6 or python3.7. Additionally, if the codes are runned on a Server, one should use the miniconda3 for python 3.7 or 3.6. However, if you dowmload the latest version of miniconda3 from https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh, you will get a miniconda3 based on python 3.8. Hence, you should redirect to the https://docs.conda.io/en/latest/miniconda.html, then download the miniconda3 based on python3.7.

# Corresponding Papers

## Subspace decomposition based DNN algorithm for elliptic type multi-scale PDEs 
created by Xi-An Li, Zhi-Qin John, Xu and Lei Zhang

[[Paper]](https://arxiv.org/pdf/2112.06660.pdf)
## Ideas 
Combining the subspace decompositon in multiscale numerical method and MscaleDNN architecture, the Subsapce decomposed DNN algorithm is developed in this work. In terms of MscaleDNN, one can rdirect to https://github.com/Blue-Giant/MscaleDNN_tf1 .

## Abstract
While deep learning algorithms demonstrate a great potential in scientific computing, its application to multi-scale problems remains to be a big challenge. This is manifested by the ``frequency principle"  that neural networks tend to learn low frequency components first. Novel architectures such as multi-scale deep neural network (MscaleDNN) were proposed to alleviate this problem to some extent. In this paper, we construct a subspace decomposition based DNN (dubbed SD$^2$NN) architecture for a class of multi-scale problems by combining MscaleDNN algorithms with traditional numerical analysis ideas. The proposed architecture includes one low frequency normal DNN submodule, and one (or a few) high frequency MscaleDNN submodule(s), which are designed to capture the smooth part and the oscillatory part of the multi-scale solutions simultaneously. We demonstrate that the SD$^2$NN outperforms existing models such as MscaleDNN, through several benchmark multi-scale problems in regular or perforated domains.

# Noting
The matlab codes in 2D辅助matlab代码/p=2 are useful for E1,E2,E3 and E4.

The matlab codes in 2D辅助matlab代码/p=3Forier_scale are useful for E5.

The matlab codes in 2D辅助matlab代码/p=3Subspace are useful for E6.

