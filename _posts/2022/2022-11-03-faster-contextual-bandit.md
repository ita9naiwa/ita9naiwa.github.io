---
layout: article
title: "Faster Linear Contextual Bandit by Removing Inverse Operation"
category: "recsys"
tag: "recsys"
comment: true
key: 20221103
mathjax: true
---

Linear contextual bandit is a necessary tool in modern machine learning, especially in Recommender Systems. One of the difficulties of deploying it comes from that it involves an inverse operation. In this post, I will explain how to make it faster without removing inverse operation.

## Cholskey Decomposition

It decomposes a positive-definite matrix into the product of a lower triangular matrix and its transpose.
And, Covariance Matrix of Multivariate Gaussian Distribution is also positive-definite.
Thus, covariance matrix $\Sigma$ in the Gaussian Distribution can be decomposed by $\Sigma = LL^T$ where $L$ is a lower trinalguar matrix.
and, It's quite faster than inverse operation although they have same time complexity (as far as I know, this decomposition is used in some methods calculating matrix inverse)


Let's suppose we defined following variables for training and inference in the Linear UCB and Linear Thompson Sampling.

```python
def get_ingredients(dim=2):
    user_feat = np.random.normal(size=dim)
    mean = np.zeros(dim)
    cov = np.random.random(size=(dim, dim))
    cov = cov.dot(cov.T)
    inv = np.linalg.inv(cov)
    b = np.random.normal(size=dim)
    x = user_feat = np.random.normal(size=dim)
    return dim, mean, cov, inv, b, x
dim, mean, cov, inv, b, x = get_ingredients(dim=2)
```

## Removing Matrix Inverse in Linear UCB

Refer to [original paper](https://arxiv.org/pdf/1003.0146.pdf) for notations. Linear UCB generates score using
$$
    \hat{s} = \theta^T x + \alpha * \text{UCB}
$$
where

- $\theta$ : $\Sigma b$
- $\text{UCB} = \sqrt{x\Sigma^{-1}x}$.

Let's decompose $\Sigma$ using Cholsky Decomposition, $\Sigma = LL^T$.

### deriving mu without inverse calculations

We will drive $\hat{s}$ without using inverse. First step is $\theta^T x$.
$$
    \theta^T x = b^T \Sigma = b^T(LL^T)^{-1}x = (b^T L^{-T}) (L^{-1}x) = (L^{-1}b)^T (L^{-1} x)
$$

### deriving UCB without inverse calculation

Followingly, we will derive $\text{UCB}$ without using inverse too.
$$
    \text{UCB}^2 = x \Sigma^{-1} x = x(LL^T)^{-1}x = (L^{-1}x)^T(L^{-1}x)
$$

### Python Implementation

```python
def usual_ucb_calc():
    # usual UCB inference.
    mu = inv.dot(b).dot(x)
    ucb = np.sqrt(inv.dot(x).dot(x))
    return mu + ucb

def without_inv_ucb_calc():
    # UCB inference without inverse operation.
    v = scipy.linalg.solve_triangular(L, x, lower=True)
    mu = np.dot(scipy.linalg.solve_triangular(L, b, lower=True), v)
    ucb = np.sqrt(np.dot(v, v))
    return mu + ucb
```

### Benchmark

#### in 64 dim

Inference is not much faster than usual cases rahter become slower in lower dimension.

```python
%%timeit -n 1000 -t 5
usual_ucb_calc()
```

> 21.6 µs ± 32.1 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

```python
%%timeit -n 1000 -t 5
without_inv_ucb_calc()
```

> 61.2 µs ± 8.42 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

Training (model update)

```python
%%timeit -n 1000 -t 5
L = np.linalg.cholesky(cov)
```

> 501 µs ± 3.63 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

```python
%%timeit -n 1000 -t 5
inv = np.linalg.inv(cov)
```

> 4.78 ms ± 211 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
> The performance gain at the 64d is about 10x gain in training one sample in 64 dim.

#### in 256 dim

```python
%%timeit -n 1000 -t 5
usual_ucb_calc()
```

> 298 µs ± 4.03 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

```python
%%timeit -n 1000 -t 5
without_inv_ucb_calc()
```

> 183 µs ± 59.7 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)


```python
%%timeit -n 1000 -t 5
L = np.linalg.cholesky(cov)
```

> 11.3 ms ± 474 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

```python
%%timeit -n 1000 -t 5
inv = np.linalg.inv(cov)
```

> 75.3 ms ± 8.17 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)

However, in a very high dimension of 256, it shows both faster inference and training.
In high dimension, it shows about 1.5-2 times faster inference, and learning one sample becomes 5-10x times faster.
We can try if we need a faster model update and a bit slower inference is tolerable (however it is not a very common case I guess).
In higher dimensions, however, inference and training become faster, so we can consider using this method.

## Fast Enough Linear Thompson Sampling

Refer to the [original paper](http://proceedings.mlr.press/v28/agrawal13?ref=https://githubhelp.com) for notations.

Linear Thompson Sampling with Gaussian Prior samples data from
$$
    \text{Normal}(\mu, \alpha^2 B^{-1})
$$
where $\mu$ is some $d$ vector, and $\alpha$ is some hyperparameter, and $B$ is an inverse of some positive-semidefinite matrix.

Sampling from Multivariate Gaussian Distribution is one and largest bottleneck of Linear Thompson Sampling. Let's remove inverse operation and see how much it becomes faster! It has two bottlenecks. First Sampling from Gaussian itself is really slow, and it involves inverse operation.

### Reparametization Trick

Drawing a sample X from Gaussian, $ X \sim \text{Normal}(\mu, a^2  B^{-1})$ We can do the same thing without removing "dependencies" among elements in $X$ and do it faster. It is equivalent to $X \sim \mu + U^{-1}Z$ where $Z$ is $\text{Normal}(0, 1)$. A further gain is from there's no inverse operation of the Covariance Matrix to retrieve $B$.

Thus, in python language, they are equivalent to

```python
    U = scipy.linalg.cholesky(cov) #Scipy choose automatically upper triangle
    rv2 = np.random.multivariate_normal(mean, (alpha ** 2) * inv)
    rv1 = mean + alpha * scipy.linalg.solve_triangular(U, np.random.normal(size=dim), lower=False)
```

### Implementation

```python
%%timeit -n 100 -t 5
rv = mean + alpha * scipy.linalg.solve_triangular(U, np.random.normal(size=dim))
```

> 154 µs ± 281 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

```python
%%timeit -n 100 -t 3
rv = np.random.multivariate_normal(mean, alpha * inv)
```

> 52.1 ms ± 28.8 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)
> (154us vs 52.1ms!) It's about 300 times faster! It's definitely good to be used in most of cases.

### Conclusion

Linear UCB shows moderable performance gain. However, Linear Thompson sampling with Gaussian Prior shows at most 300x times faster inference.

### Reference:

[Sampling from Multivariate Normal (precision and covariance parameterizations)](