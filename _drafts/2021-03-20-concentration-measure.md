---
layout: article
title: "추천 시스템의 문제 해결하기"
category: "ML"
tag: "ML"
mathjax: true

---

## Recap on Concentration Measure (1)

> For MAB Study Group
> Hyunsung Lee
> Kakao and Sungkyunkwan University.



As a Machine Learning practitioner, **Why do we need to Concentration Measure?**

There is a simple answer.

### Goal 1

Our algorithm $\mathcal{A}$ given training data $X$ produces model $h$. Our model chooses one $h$ from set of possible models (Represetnation Power) $\mathcal{H}$. Let $h^*$ to be the best model upon test dataset on $X_T$, or $h^* = \argmin_{h \in  \mathcal{H}} R(h, X_T)$, where $R(h, X_T)$ is the loss on data $X_T$ of the model $h$.

What we want to know is about the event $E$:
$$
    E  = \{|R(h^*, X_T) - R(h, X_T)|\}
$$

We want to know $\delta = \text{Pr}(|R(h^*, X_T) - R(h, X_T)| > \epsilon)$. We want to know $\epsilon$ if $\delta = 0.01$. We want to know $\delta$ while achieving $\epsilon$ to smaller than $1.0$.

**Concentration measure gives a tool to gear $\epsilon, \delta$, giving theoritical bounds about what model can do and can't do.**

### Markov Ineq. and Chebychev Ineq;
Let $Z$ to be a r.v such that $p(z < 0) = 0$.

#### Ineq. (1): Markov inequality
$$
    E[Z] = \int_0^\infin z p(z) dz \geq \int_{\textcolor{red}{\epsilon}}^\infin zp(z)dz \\
    \geq \int_\epsilon^\infin \textcolor{red}{\epsilon} p(z)dz \geq \textcolor{red}{\epsilon} \int_\epsilon^\infin p(z)dz \geq \epsilon \textcolor{red}{P[Z > \epsilon]}
$$

yielding
$$
    P[Z > \epsilon] \lt \frac{E[Z]}{\epsilon}
$$


#### Ineq. 2: Chebychev inequality
Let put $Z = \vert Z' - \mu \vert $ in Inequality (1) gives

$$
    P[\vert Z'-\mu \vert > \epsilon] = P[(Z'-\mu)^2 > \epsilon^2] \lt \frac{1}{\epsilon^2} E[(Z' - \mu)^2] = \frac{\sigma^2}{\epsilon^2}
$$where $\sigma^2$ is variance of $Z$ and arbitrary $\mu$ but usually mean of $Z$
Let suppose that $Z_1, \dots, Z_n$ are iid and $\bar Z = \frac{1}{n} \sum_i Z_i$, then $\bar Z$ is has mean of $\mu$ and variance of $\frac{1}{n} \sigma^2$, thus by the Ineq. (2),

$$
    P[(\bar Z - \mu)^2 > \epsilon^2] < \frac{\sigma^2}{n\epsilon^2}
$$


