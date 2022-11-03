---
layout: article
title: "11월 3일의 일기"
category: 일기
tag: 일기
comment: false
key: 20221103
mathjax: true
---

Linear contextual bandit is a necessary tool in modern machine learning especially in Recommender Systems. One of the difficulties of deploying it comes from that it involves an inverse operation. In this post, I will explain how to make it faster without removing inverse operation.

## Cholskey Decomposition
it decomposes positive-definite matrix into the product of a lower triangular matrix and its transpose. And, Covariance Matrix in MVN is also positive-definite Thus, $\Sigma = LL^T$ where $L$ is a lower trinalguar matrix.

Let's suppose we defined following variables
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

Let's decompose $\Sigma$ using Cholsky Decomposition, $\Sigma = LL^T$
### deriving mu without inverse calculation
$$
    \theta x = b^T(LL^T)^{-1}x = (b^T L^{-T}) (L^{-1}x) = (L^{-1}b)^T (L^{-1} x)
$$

### deriving UCB without inverse calculation
$$
    UCB^2 = x \Sigma^{-1} x = x(LL^T)^{-1}x = (L^{-1}x)^T(L^{-1}x)
$$

### Python Implementation
```python
def usual_ucb_calc():
    mu = inv.dot(b).dot(x)
    ucb = np.sqrt(inv.dot(x).dot(x))
    return mu + ucb

def without_inv_ucb_calc():
    v = scipy.linalg.solve_triangular(L, x, lower=True)
    mu = np.dot(scipy.linalg.solve_triangular(L, b, lower=True), v)
    ucb = np.sqrt(np.dot(v, v))
    return mu + ucb
```

### Benchmark
#### in 64 dim
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

In high dimension, it shows about 1.5-2 times faster inference, and learning one sample becomes 5-10x times faster

## Fast Enough Linear Thompson Sampling
Refer to [original paper](http://proceedings.mlr.press/v28/agrawal13?ref=https://githubhelp.com) for notations.

Linear Thompson Sampling with Gaussian Prior samples data from
$$
    \text{Normal}(\mu, a^2  B^{-1})
$$
where $\mu$ is some $d$ vector, and $\alpha$ is some hyperparameter, and $B$ is an inverse of $d \times d$ matrix

Sampling from Multivariate Gaussian Distribution is one and largest bottleneck of Linear Thompson Sampling. Let's remove inverse operation and see how much it become faster!

### Reparametization Trick
Drawing a sample X from MVN, $ X \sim \text{Normal}(\mu, a^2  B^{-1})$ is equivalent to $X \sim \mu + U^{-1}Z$ where $Z$ is $\text{Normal}(0, 1)$.
Thus, in python language, they are equivalent to

```python
    rv1 = mean + sqrtalpha * scipy.linalg.solve_triangular(U, np.random.normal(size=dim), lower=False)
    rv2 = np.random.multivariate_normal(mean, alpha * inv)
```
### Implementation
```python
%%timeit -n 100 -t 5
rv = mean + sqrtalpha * scipy.linalg.solve_triangular(U, np.random.normal(size=dim))
```
> 154 µs ± 281 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

```python
%%timeit -n 100 -t 3
rv = np.random.multivariate_normal(mean, alpha * inv)
```
> 52.1 ms ± 28.8 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)

it's about 300 times faster!

### Conclusion
Linear UCB shows moderable performance gain. However, Linear Thompson sampling with Gaussian Prior shows 300x times faster inference.
