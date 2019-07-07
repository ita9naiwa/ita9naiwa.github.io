---
layout: article
title: "Logistic Matrix Factorization and Negative Sampling"
category: "recommender systems"
tag: "recommender systems"
mathjax: true
comment: true
key: lmf
---

#### 잡담

카카오에서 영입 제의를 받았다(예이!!). 근데 내가 미필이라 어떻게 될 지 모르겠다(ㅠㅠ)


(내 생각에) 요즘 추천 시스템은 Implicit feedback을 어떻게 해석하는가에 관한 문제인 것 같다. 솔직히 어떻게 해석해야 좋은지 잘 모르겠다. 이를 확률인 $p(l_{ui} \vert \theta)$로 보는게 가장 좋을 것 같긴 한데, WMF는 확률적 해석을 하지 않는, Regression이다.(WMF에 관한 내용은 다른 글을 참고, 혹은 내 [블로그 포스트](https://ita9naiwa.github.io/recommender%20systems/2018/06/10/wmf.html)을 보면 감사...) 과연 유저가 어떤 아이템을 좋아할지는 확률인데, 확률인데... 이를 Regression으로 해결하거나 하는게 좋은 일인가? 이는 아니라고 생각한다. ~~하지만 난 좋은 방법을 제안할 능력은 안되고~~ 더 나은 방법이 있는지 찾아보다가, logistic matrix factorization이라는 [논문](https://web.stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf)을 알게 되었다.


### Introduction
**이 글을 읽는 사람은 기본적인 CF, Matrix Factorization에 대해 잘 알고 있다고 가정합니다(미적분학도 할 줄 알면 좋겠지용).**

Logistic Matrix Factorization은 User $u$와 Item $i$가 interaction할 확률을 모델링하는 Matrix Factorization 모델이다.

즉, 간단히 말해
유저가 어떤 아이템을 좋아할 확률을 다음과 같이 표현하겠다는 얘기이다.

$$
    p(l_{ui}|x_u^Ty_i + b_u + b_i)
$$

where $l_ui$ is the event the user $u$ interacts with the item $i$, $x_u$, $y_i$ are latent representation for user $u$, and item $i$. $b_u$, $b_i$ are usre, item bias, respectively.



### Logistic Matrix Factorization
유저와 아이템의 interaction을 일반적인 matrix factorization과는 다르게  sigmoid 함수를 이용해 표현하는 점 이 모델의 독특한 점이다.
$$
    p(l_{ui}|x_u^Ty_i + b_u + b_i) = \text{sigmoid}(x_u^Ty_i + b_u + b_i)
$$

#### bias term의 해석
어떤 아이템은 많은 유저와의 interaction이 존재한다. 즉, popular한 아이템일 경우 bias의 값이 높다고 해석할 수 있다. 어떤 유저는 많은 아이템과 interaction했다. 즉, 취향이 다양하거나, 다양한 음악을 들은 유저의 경우 bias 값이 높다고 생각할 수 있다. bias term이 이를 잘 표현하는지는 사실 모르겠지만, 이러한 Popularity bias를 해결할 것이라 기대하며 이런 Bias term을 추가했다고 한다.

#### Likelihood function

일반적으로, Logistic Regression과 같은 문제에서는 예측하려는 사건 $y_i$가 Bernoulli 분포를 따른다고 가정한다. 즉 $y_i$는 0 또는 1이다.
또한 likelihood는 다음과 같은 형태를 띄게 된다.

$$
    \prod_i p(y_i)^{y_i}(1-p(y_i))^{1-y_i}
$$

근데, 이 논문에서는 왜인지 likelihood를 다음과 같이 정의했다.

$$
    \prod_i p(y_i)^{c_i}(1-p(y_i))
$$

$c_i$는 (WMF에도 존재하는)Confidence Term이다.

이는 $y_i$의 정의도 바꾸게 되는데, 더 이상 $y_i$는 Bernoulli 분포를 따르지 않고, $\text{Beta}(c_i+1, 2)$ 분포를 따르게 된다.

베타 분포는 값이 0과 1 사이에 존재하는 분포가 된다.
![](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Beta_distribution_pdf.svg/540px-Beta_distribution_pdf.svg.png)

왜 이렇게 정의한지 잘 이해가 가지 않는다. 아무튼 이런 식으로 유저가 아이템을 좋아할 만한 사건을 정의하면, $y_i$는 "유저가 아이템을 좋아한다"는 사건이 아니라, "유저가 아이템을 좋아하는 정도"라는 식으로 해석을 할 수 있다.

$y_i$를 유저 $u$가 아이템 $i$를 좋아하는 정도 $l_{ui}$로 바꿔 likelihood를 다시 한 번 표현해 보면 다음과 같다.

$$
    \textbf{L}(D\vert X, Y, B_u, B_i)\prod_{u, i} p(l_{ui})^{c_{ui}}(1-p(l{ui}))
$$
where $X = [x_1, x_2, \cdots], Y = [y_1, y_2, \cdots], B_u = [b_{u1}, \cdots], B_i = [b_{i1}, \cdots]$

likelihood에 log를 취한 후에, X, Y에  Gaussian prior를 더해주면(혹은, Equivalent하게, L2 regularization을 취하고) 모델의 objective를 유도할 수 있다.

$$
    \log \textbf{L}(D|X,Y,B_u, B_i) = \\
    \sum_{u,i}c_{ui}(r_{ui}) - (1 + c_{ui})\log(1 + \exp(r_{ui})) - \frac{\sigma_u}{2}2x_u^Tx_u -\frac{\sigma_i}{2}2y_i^Ty_i
$$
where $r_{ui} =  x_u^Ty_i + b_u+b_i$

$\sigma_u$와 $\sigma_i$를 $\lambda$로 놓고, $x_u$에 대해 gradient를 구하면 다음과 같다.

$$
\frac{\partial}{\partial x_u} = \sum_i [c_{ui}y_i - \frac{(1+c_{ui})\exp(r_{ui})}{1 + \exp(r_{ui})}y_i -\lambda x_u]
$$
$b_u$에 대한 gradient도 비슷하다.

$$
\frac{\partial}{\partial x_b} = \sum_i [c_{ui} - \frac{(1+c_{ui})\exp(r_{ui})}{1 + \exp(r_{ui})}]
$$
$y_i$, $b_i$에 대한 gradient는 대칭적이므로 생략한다 'ㅅ'...

#### Approximating full loss using negative sampling

total itemset $I$가 커질수록, $\frac{\partial}{\partial x_u}$을 계산하기가 힘들어진다. 이는, $\sum_i$ Term이 모든 아이템에 대해 iteration을 요구하기 때문이다.

전체 아이템에 대해 $\frac{\partial}{\partial x_u}$을 계산하지 않고, 일부 negative item만 선택해서 training을 해도 비교적 좋은 결과를 얻을 수 있다고 저자는 주장하고 있다.

위에서 정의된 Loss 함수는 일부 negative item을 선택하기 위해 딱 맞춰져 있는 것 같다. 이를 위해 Beta distribution을 사건의 확률로 가정한게 아닌가 싶다. Negative item만을 sampling하기 위한 구조는 다음과 같다.

$$
\frac{\partial}{\partial x_u} = \sum_{i\in I^+}{c_{ui}y_i} - \sum_{i\in I^+}  \frac{c_{ui}\exp(r_{ui})}{1 + \exp(r_{ui})}y_i - \sum_{i\in I}  \frac{\exp(r_{ui})}{1 + \exp(r_{ui})}y_i
$$
편의상 regularization은 생략했고, $I^+$은 유저의 interaction이 존재하는 아이템들, $I$는 전체 아이템셋을 말한다.
저기서 오른쪽 항의 마지막 텀인, $\sum_{i\in I}  \frac{\exp(r_{ui})}{1 + \exp(r_{ui})}y_i$을 보면, $i \in I$에서 하는 부분을 $\vert I' \vert << \vert I \vert$가 성립하는 $I'$에 대해서 sampling해도 괜찮을 것 같지 않은가?

마지막 텀을 전체 아이템을 iterate하지 않고, 아이템의 서브셋에서 iterate하는 것으로 model의 time complexity를 다음과 같이 줄일 수 있다.
$$
O(\vert I \vert \times \vert U \vert) \rightarrow O(k (\vert I \vert + \vert U \vert))
$$
$k$는 유저와 아이템의 평균적인 interation 횟수이다.
또한, $I'$을 만들 때 positive item을 포함하는 이유는, 각 유저 별로 Negative item을 detect해서, 이 안에서 샘플링을 하는 것이 엄청 inefficient하기 때문이다.

#### Loss approximation을 했을 때의 성능 그래프

![chart]({{ "/assets/images/lmf/chart.png" | absolute_url }})
loss를 approximate하기 위해 사용하는 sample의 비율(positive interaction의 K배 많게 하는 K)를 조정해가면서, Movielens 100k 데이터셋에 대해 트레이닝한 결과이다. K가 적당히 클 때, 전체 데이터를 사용하는 것 만큼이나 효율적임을 알 수 있다.implementation이 그리 efficient하게 되지는 않아서... 속도에 대한 비교는 할 수 없는게 아쉽다.

이를 파이썬으로 구현하면 다음과 같다.

```python
    def get_deriv(self, idx, user=True, portion_neg_samples=10):
        if user:
            mat = self.Cui
            user_vectors = self.user_vectors
            item_vectors = self.item_vectors
            num_items = self.num_items
        else:
            mat = self.Ciu
            user_vectors = self.item_vectors
            item_vectors = self.user_vectors
            num_items = self.num_users

        pos_item_indices = mat.indices[mat.indptr[idx]:mat.indptr[idx + 1]]
        pos_confidences = mat.data[mat.indptr[idx]:mat.indptr[idx + 1]]
        num_pos_interactions = len(pos_confidences)

        #일부 아이템만을 샘플링해서 이용하고 싶은 경우
        neg_item_indices = np.random.choice(num_items, min(num_pos_interactions * portion_neg_samples, num_items))

        # 전체 아이템을 이용하고 싶은 경우
        #neg_item_indices = np.arange(num_items)
        pos_item_vectors = item_vectors[pos_item_indices]
        neg_item_vectors = item_vectors[neg_item_indices]

        A = np.dot(pos_confidences, pos_item_vectors)

        ret = self.u_score(user_vectors, item_vectors, idx, pos_item_indices)
        ret = (pos_confidences * ret) / (1 + ret)
        B = np.dot(pos_item_vectors.T, ret)

        ret = self.u_score(user_vectors, item_vectors, idx, neg_item_indices)
        ret = ret / (1 + ret)
        C = np.dot(neg_item_vectors.T, ret)
        _uv = user_vectors[idx].copy()
        _uv[-2:] = 0.
        deriv = A - (B+C) - self.reg_param * user_vectors[idx]

```

#### 성능 비교

![chart]({{ "/assets/images/lmf/eval.png" | absolute_url }})

모델이 그리 성능이 좋지 않은 것 같다...
MovieLens dataset에서만 그런 걸 수도 있겠는데, 이런 정도라면 추천하는데 사용하긴 힘들 것 같다.

다만, 유저 $u$가 특정 아이템 $i$를 좋아할 확률을 모델링 할 수 있다는 점에서,

![chart]({{ "/assets/images/lmf/koko.png" | absolute_url }})
이런 걸 구현하는 데에 응용하면 좋지 않을까 생각한다.


### References



#### References:

[1] Logistic Matrix Factorization for Implicit Feedback Data, https://web.stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf
