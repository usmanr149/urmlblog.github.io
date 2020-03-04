---
layout: post
title: Implement and Apply a k-Nearest Neighbor (kNN) Classifier -- Exercise
---

In this exercise we are asked to train a k-NN classifier on the CIFAR-10 dataset. 
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, 
with 6000 images per class. There are 50000 training images and 10000 test images.

![_config.yml]({{ site.baseurl }}/images/cifar-10.png)

### Using Two Loops Calculate Distances

This is rather simple, we need to calculate the Euclidean distance between each point in 
out testing and training dataset. We have already reshaped the CIFAR-10 data into single 
rows. So the distance between test data $i$ and train data $j$ is given as

$$
\begin{align*}
  dist[i,j] = \sqrt{(\sum_{dim=1}^{dim=3072} (X\_train[i][dim] - X\_ test[j][dim])^2}
\end{align*}
$$

Another way to write this as a dot product:

$$
\begin{align*}
  dist[i,j] = \sqrt{(X\_train[i] - X\_ test[j]) \cdot (X\_train[i] - X\_ test[j])}
\end{align*}
$$

Here is the code implemented in python:

```python
num_test = X.shape[0]
num_train = self.X_train.shape[0]
dists = np.zeros((num_test, num_train))
for i in range(num_test):
    for j in range(num_train):
        v_sub = X[i] - self.X_train[j]
        dists[i, j] = np.sqrt(v_sub.dot(v_sub))
```