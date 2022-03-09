---
layout: article
title: "Well-Classified Examples are Underestimated..."
category: "ML"
tag: "ML"
header-includes:
  - \usepackage[fleqn]{amsmath}
mathjax: true
---


Summary of the paper "Well-Classified Examples are Underestimated in Classification with Deep Neural Networks" of AAAI 2022

## TL;DL;
- I didn't understand Energy related parts.

### Paper Link
https://arxiv.org/abs/2110.06537

## different losses/derivations w.r.t $p$ or $\theta$
where $p = \sigma(f(x))$ and $\sigma$ is sigmoid, and $f(x) \in \mathbb{R}^n$ is the output of the Neural Network.

### MSE with Sigmoid Activation
$L = (p - y)^2$

$$
  \frac{\partial}{\partial \theta} L = 2(p - y)p(1-p) \nabla_\theta f(x)
$$
Since y is either 0 or 1, gradients vanish quadratically as $p$  converges.

### BCE with Sigmoid Activation

$L = -\log p(y|x)$

$$
  \frac{\partial}{\partial \theta} L = (1-p) \nabla_\theta f(x)
$$
in BCE, gradients converges linearly as $p(y \vert x) \rightarrow  1$

Thus, BCE gives more steep graidents than MSE.


### MAE with Sigmoid Activation

*Note 1: not appear in the paper*

$$
  \frac{\partial}{\partial \theta} L = \text{sign}(p - y)p(1-p) \nabla_\theta f(x)
$$
See that $p(1 - p) \simeq p$ if $p \rightarrow 1$ and  $p(1-p) \simeq (1 - p)$ if $p \rightarrow 0$, Therefore, MAE shows similar convergence behavior to BCE e.g., gradient convernges linearly.



It has been noted recently, smaller gradients for high-confidence samples are harmful in Rrepresentation Learning and Authors claims linear decay of gradients is still not good enough.

## Proposed Solution

### 1. A *bonus loss* is proposed, symmetric to log-likelihood function

$$
  L_O = -\log p(y \vert x) + \log (1 - p(y \vert x))
$$

### 2. The bonus function is truncated to linear function;

I think this is made for technical reasons. First, $\log (1 - p(y \vert x))$ diverges to minus inf if p gets to one, it's the main objective of the learning. So, $\log (1 - p(y \vert x))$ is replaced by a linear function and to be continuous with $\log (1 - p(y \vert x))$.

$$
  L_{LE} = -\log p(y \vert x) + C - p(y \vert x)
$$
when $p$ is close to 1. $C$ is determined such that $L_{O}$ and $L_{LE}$ are continuous.

*Note 2: not appear in the paper*
Also, when $p$ is high enough so linear function is used, then it is exactly same when the loss is MAE with sigmoid activation. In this case, $p$ is close to $1$ so the gradients converge linearly.


## implemenation

I implemented these losses based on https://github.com/kuangliu/pytorch-cifar and tuned some parameters.

https://gist.github.com/ita9naiwa/49ab8279d3277ab5d8b0795e1eb0ea1d


|--------------|
|Loss Function|Accuracy on Test set|
|---|-----------|
|CE | 92.67|
|mae + mse | 92.33|
|CE + bonus CE | 91.13|

I couldn't reproduce experiments on the paper, namely, "On same hyperparameter set, Bonus CE gives better in terms of accuracy...". but I didn't try to find good hyperparameter sets for bonus CE.

## Thoughts
- Even I failed to reproduce results, it gives a thoughtful view to look at various loss functions and their gradients.