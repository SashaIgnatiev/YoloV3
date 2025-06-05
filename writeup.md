# YoloV1 Writeup

## 1. Overview

### Notation
- S—grid size --> 7
- B—number of bounding boxes per grid --> 2
- C—number of classes --> 20

Thus, each prediction will be encoded as an $S \times S \times (B \times 5 + C)$ tensor. 
Here for each of the $S^2$ grid section, there will be B potential bounding boxes 
(each containing 5 parameters), and a C probabilites of each class.


## 2. The Architecture
Input dimensions: 448 x 448 x 3 (r,g,b)

Output (prediction) dimensions: 7 x 7 x 30

## 3. Loss Function

This is the official loss function from the YoloV1 paper:
$$
\begin{array}{lcl}
\lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} \left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 \right] +\\
\lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} \left[ \left( \sqrt{w_i} - \sqrt{\hat{w}_i} \right)^2 + \left( \sqrt{h_i} - \sqrt{\hat{h}_i} \right)^2 \right] +\\
\sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} (C_i - \hat{C}_i)^2 +
\\ \lambda_{\text{noobj}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{noobj}} (C_i - \hat{C}_i)^2 +\\
\sum_{i=0}^{S^2} \mathbb{1}_{i}^{\text{obj}} \sum_{c \in \text{classes}} \left( p_i(c) - \hat{p}_i(c) \right)^2 \\
\end{array}
$$

