# Requirement
The codes are implemented in tensorflow--1.14 or 1.15 under the interpreter python3.6 or python3.7

# Ideas

# Abstract
Multi-scale Deep neural network (MscaleDNN) algorithms are proposed for multiscale/high-frequency PDEs and have gained remarkable success to multi-scale problems by shrinking high-frequency component into low-frequency space. In this paper, we construct a subspace-decomposed DNN (named as SDNN) architecture to a class of multi-scale problems by combining numerical homogenization and MscaleDNN algorithms. Especially, a novel activation function based on Fourier decomposition and expansion is designed for this SDNN model.  This new architecture includes two parts: a low-frequency or normal DNN submodule and a MscaleDNN submodule, they are used to capture the low-frequency component and the high-frequency component, respectively, for multi-scale problems. We demonstrate the performance of  the new activation function and the SDNN architecture through some benchmark multi-scale problems in regular or irregular domian, and numerical results show that our new activation function is favorable and the SDNN model is clearly superior to MscaleDNN.

# Noting
The matlab codes in 2D辅助matlab代码/p=2 are useful for E1,E2,E3 and E4.

The matlab codes in 2D辅助matlab代码/p=3Forier_scale are useful for E5.

The matlab codes in 2D辅助matlab代码/p=3Subspace are useful for E6.

