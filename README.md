# Requirement
The codes can be implemented in tensorflow--1.14 or 1.15 under the interpreter python3.6 or python3.7

# Ideas

# Abstract
With the development of deep learning, some multi-scale deep neural network (MscaleDNN) algorithms are proposed and have gained remarkable success to multi-scale problems by shrinking high-frequency component into low-frequency space. In this paper, we construct a subspace-decomposed DNN (named after SDNN) architecture for solving a class of nonlinear multi-scale problems by combining homogenization theory and MscaleDNN algorithm.  This new architecture includes two parts: a normal DNN submodule  and a MscaleDNN submodule, they are used to capture the low-frequency component and the high-frequency component, respectively, for multi-scale problems. In addition, some common activation functions are carefully selected for our new method. Especially, a novel activation function based on Fourier decomposition and expansion is designed for the MscaleDNN module, which improves greatly the performance of MscaleDNN. Finally, by introducing some numerical examples of $p$-Laplacian problems with different scale information for independent variables in various dimensional Euclidean spaces, we demonstrate that the SDNN architecture is feasible and can attain favorable accuracy to $p$-Laplacian problems for both low-frequency and high-frequency oscillation cases.

# Noting
The matlab codes in 2D辅助matlab代码/p=2 are useful for E1,E2,E3 and E4.

The matlab codes in 2D辅助matlab代码/p=3Forier_scale are useful for E5.

The matlab codes in 2D辅助matlab代码/p=3Subspace are useful for E6.

