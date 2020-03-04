---
layout: post
title: Implement and Apply a k-Nearest Neighbor (kNN) Classifier -- Exercise
---

In this exercise we are asked to train a k-NN classifier on the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

![_config.yml]({{ site.baseurl }}/images/cifar-10.png)

### Using Two Loops Calculate Distances

This is rather simple, we need to calculate the Euclidean distance between each point in 
out testing and training dataset.

$$
\begin{align*}
  & \phi(x,y) = \phi \left(\sum_{i=1}^n x_ie_i, \sum_{j=1}^n y_je_j \right)
  = \sum_{i=1}^n \sum_{j=1}^n x_i y_j \phi(e_i, e_j) = \\
  & (x_1, \ldots, x_n) \left( \begin{array}{ccc}
      \phi(e_1, e_1) & \cdots & \phi(e_1, e_n) \\
      \vdots & \ddots & \vdots \\
      \phi(e_n, e_1) & \cdots & \phi(e_n, e_n)
    \end{array} \right)
  \left( \begin{array}{c}
      y_1 \\
      \vdots \\
      y_n
    \end{array} \right)
\end{align*}
$$

```python
num_test = X.shape[0]
num_train = self.X_train.shape[0]
dists = np.zeros((num_test, num_train))
for i in range(num_test):
    for j in range(num_train):
        v_sub = X[i] - self.X_train[j]
        dists[i,j] = np.sqrt(np.matmul(v_sub, v_sub.T))
```