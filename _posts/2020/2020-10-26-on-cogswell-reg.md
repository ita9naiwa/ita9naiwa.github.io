---
layout: article
title: "An approximation of Cogswell Regularization for SGD update in Recsys."
category: "recsys"
tag: "recsys"
mathjax: true
---



### Cogswell Regularization
Cogswell Regularization has been proposed by [1] in 2015. This is based on the following observations

- Embeddings are spread on the Euclidean Space.
- If two feature vectors are different, then their representations are different.
It would be good to fill the embedding space with a similar density of embeddings if they are in the range of the input network.

So [1] Proposed the following objective.

$$
    L_\text{DeCov} = \frac{1}{2}(\vert \vert C \vert \vert_F - \vert \vert diag(C) \vert \vert_2^2 )
$$

$C$ is empirical covariance among feature vectors, or embeddings, defined as

$$
    \{C_{i, j}\} = \frac{1}{\vert V \vert} \sum_{v \in V}(v_i - \mu_i)(v_j - \mu_j)
$$

where $\mu = \frac{1}{\vert V\vert }\sum_{v \in V} v$

### Collaborative Metric Learning
In 2017, Hsieh proposed L2 regularization based factor based top-k recommendation model, CML. It exploits **batched gradient descent** and **Cogswell Regularization**. For this model, the regularization scheme is so critical; thus, it cannot be removed.

As we observe at the equation of the Cogswell regularizer, we need a set of embeddings to calculate covariance and mean. So, CML is quite hard not applicable for stochastic gradient descent update.

However, for implicit feedback recommender systems, there are still a lot of constraints blocking uses of batched update or deep-learning based framework such as Tensorflow or PyTorch.
For example, in case we only have lots of low-performance CPU and low-memory machines.

Upon those requirements, there exist recommendation frameworks such as [Buffalo](https://github.com/kakao/buffalo), and [Implicit](https://github.com/benfred/implicit) have plenty of models with stochastic gradient descents only.

Anyway, **CML can't be updated with SGD because estimating covariance requires batched input sampling.**

### Solution
I wondered a few days (not spent entire my day to efforts for this but..)

I came up with a simple idea. We can't approximate the covariance matrix, but we can achieve the same goal without the covariance matrix.

1. We want to spread embeddings in the range of the embedding mappings.
2. The range of embedding space is bounded in CML modeling such that $\vert\vert v \vert\vert_2 \leq 1$

Cogswell Loss attempts to distribute embeddings in equal density in the range by making their correlation zero.

First, consider the ideal case,

If embeddings are identically distributed in the range,
then the average distance between two embeddings will be

$$
    D = \mathbb{E}_{p_1}\mathbb{E}_{p_2}[\text{distance}(p_1, p_2)]
$$

where, $p_1$ and $p_2$ is drawn from in the unit hypersphere i.i.d fashion.

$D$ is $36/35$ [I found the answer in Quora](https://www.quora.com/What-is-the-average-distance-between-two-random-points-in-a-sphere#:~:text=The%20surface%20distance%20between%202,pi%20radians%20equals%20360%20degrees)

For logics, the converse of "if A then B" does not usually hold if "if A then B" holds. However, in the probability world, we can say that P(A \vert B) is high, then P(B \vert A) will be high too.


So, at least we can regularize the distances of embeddings towards $ D $. Then, we'll have more chances to get those embeddings less correlated and identically distributed. This can be done by minimizing below.
$$
    \vert \vert D - \text{distance}(v_1, v_2) \vert \vert_2^2
$$ However, distance involves square root calcuation that makes calculating gradient descent clumsy, thus instead I used
$$
L =  \text{abs}(D^2 - \text{distance}(v_1, v_2)^2)
$$

This regularizer can be easily implemented, and the gradient is:
$$
\frac{\partial L}{\partial v_1} = \text{sign}(D^2- \text{distance}(v_1, v_2))[2 \text{distance}(v_1, v_2)\frac{\partial \text{distance}(v_1, v_2)}{\partial v_1}]
$$

I implemented this with Implicit library and tested on ML-1M dataset and found that about 6\% of performance gain was achieved.
{% gist b328c43508193611a83c07ae0553a9f3 %}



### Note:
I'm currently thinking about how it works and finding any relationship between two regularizations.
Anywho, there's a quite big performance gain with this regularization even though I don't know why it works.


### References;
- [1] Cogswell et al., Reducing Overfitting in Deep Networks by Decorrelating Representations, https://arxiv.org/abs/1511.06068
- [2] Hsieh et al., Collaborative Metric Learning, https://dl.acm.org/doi/10.1145/3038912.3052639
