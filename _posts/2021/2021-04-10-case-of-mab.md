---
layout: article
title: "MAB의 경우"
category: "ML"
tag: "ML"
header-includes:
  - \usepackage[fleqn]{amsmath}
mathjax: true
---

# [WIP]
### 앞서서
{{ site.baseurl }}{% link _posts/2021/2021-04-10-about-pattern-recognition.md %}
을 보고 오면 좋을 듯...해요...

> Concept가 $A \rightarrow p(r)$인 경우를 다룬다.
> $p(r)$의 Support는 $[0, 1]$이다.
> 글 군데군데 support가 $\{0, 1\}$인 경우를 가정하고 적힌 부분이 있는데, 문맥에 따라 이해하기 어렵지 않을 것 같아 자세히 명시하지 않았음.


### MAB의 독특한 점.
- MAB에서 특정 가설 $h$에 대한 손실 함수를 Regret, 혹은 Reward라 부른다.
- MAB에서는 특정 observation $(a, r)$이 현재 가설 $h$에 의존적이다.
  - e.g., 가위바위보를 할 때, 가위만 내는 사람($h$)은 $(a=주먹, r)$인 data sample $x$를 관측할 수 없다.
- 좋은 action을 빨리 알아내는 목적이 있다.
  -  약간 편리하게 다시 얘기하자면,MAB는 손실 함수에 $m$: data sample의 숫자도 포함되어 있으며, 이 값이 T보다 크면 안된다거나, 이 값이 클 수록 Error가 크다고 생각할 수 있다.

편의상, concept $C$를 $r(\cdot)$으로 표현하는 것이 편리하다. 사실 얘는 함수는 아니다. 예를 들어서, 내가 복권을 사는 행동 $a_1$이 있다고 하자. 1등 당첨/1등 당첨이 아님 여부는 확률적이지만, 이를 함수로 표현하는 것이 표기상 보기 좋다. e.g., $r(a_1) = 0.9$ 더 표기법을 편리하게 만들려면, $r(a_i) = E_r[c(a_i)]$으로 적는 것이 제일 깔끔하다. 이런 경우 concept가 분포임은 암시적으로 표현된다.

### UCB/e-greedy를 사용하는 MAB의 경우
#### (1) 어떤 가설 집합을 사용할 것인지.
$$
  \mathcal{H} = A \rightarrow \text{reward } r \in [0, 1]
$$

#### (2) 어떻게 오차를 표현할 것인지.
$$
  mE_r[r(a')] - \sum_{i=1}^{m} E_r\left[r\left( f(h_i) \right)\right]
$$
$h_i$는 $1, 2, \dots, i-1$까지의 $(a_i, r_i)$를 관측하고 업데이트 된 hypothesis를 말한다.


#### (3) 오차를 어떻게 줄일 것인지.
UCB에선 어떤 데이터가 들어올 지 사전에 모르는 가정이다. 위에 적혀 있듯이, $r$의 Support만 알고 있다고 가정한다. (min(r), max(r)).

확률변수의 분포에 대해 거의 아는 것이 없는 상황에서도, UCB는 "보수적으로 생각했을 때 가장 좋은 선택"을 하는 방법을 제안했다.

이는 [Hoeffding/Macdirmid Inequality](https://people.eecs.berkeley.edu/~bartlett/courses/281b-sp08/13.pdf), [Azuma Inequality](https://en.wikipedia.org/wiki/Azuma%27s_inequality)의 힘이다.

WIP...


### Thompson Sampling의 경우

#### 어떤 가설 집합을 사용할 것인지
1. Conjugate Prior가 Beta-Bernoulli일 때
$$
  \mathcal{H} = A \rightarrow \text{reward } r \sim \text{Bern}(r \vert \theta) \\
  \theta \sim \text{Beta}(\alpha, \beta)
$$

2. Conjugate Prior가 Normal-Gamma일 때

$$
  \mathcal{H} = A \rightarrow \text{reward } r \sim \text{Normal}(r \vert \theta, \lambda^{-1}) \\
  \theta \sim \text{Normal}(\theta \vert  \mu, \lambda) \\
  \lambda \sim \text{Gamma}(\lambda \vert \alpha, \beta)
$$ where $\lambda^{-1} = (\sigma^2)$.

#### (2) 어떻게 오차를 표현할 것인지
UCB의 경우와 동일하다.
$$
  mE_r[r(a')] - \sum_{i=1}^{m} E_r\left[r\left( f(h_i) \right)\right]
$$

$h_i$는 $1, 2, \dots, i-1$까지의 $(a_i, r_i)$를 관측하고 업데이트 된 hypothesis를 말한다.

#### (3) Predictive Distribution
1. Conjugate Prior가 Beta-Bernoulli일 때
$$
  \theta \sim \text{Beta}(\alpha + n r, \beta + n(1 - r))
$$


2. Conjugate Prior가 Normal-Gamma일 때
$$
  \theta \sim \text{Normal}(\theta \vert  \mu, \lambda) \\
  \lambda \sim \text{Gamma}(\lambda \vert \alpha, \beta)
$$ where $\lambda^{-1} = (\sigma^2)$.

보통, 이에 대해 예측 분포를 만들려면, Marginal Predictive Distribution을 계산한다. 다시 말하자면, $p(r, \theta) \propto p(\theta \vert r)$이므로, Posterior distribution(혹은 Posterior distribution의 커널)에서 그냥 $\theta$를 하나 뽑는 방법인데, 밑의 적분을 계산해야 한다.

$$
  p(\theta) = \int p(\theta)dr
$$


Thompson Sampling에서는 이를 간단하게 처리한다. 이게 가능한 이유는..
$$
  \int p(r, \theta)dr \simeq \frac{1}{m} \sum_{t=1}^{t=m} p(r, \theta)
$$
이기 때문에, 어차피 저런 방식으로 Sampling Draw를 아주 많이 하면, 긴 기간에서 보면 적분을 하는 것과 그리 다르지 않다고 생각하는 것이다. 그래서 코드가 무척 간단해진다.
1. Conjugate Prior가 Beta-Bernoulli일 때
```python
  def get_sample_reward(alpha, beta):
    theta = sample from Beta(alpha + r_sum, beta + (n - r_sum))
    reward = Bernoulli(theta)
    return reward

  def choose_arm():
    Monte_Carlo_Rewards = {arm: get_sample_reward(arm['alpha'], arm['beta']) for arm in arms}
    chosen_arm = choose an arm with highest sample_score in Monte_Carlo_Rewards
    return chosen arm
```
