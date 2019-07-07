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



### Solving ALS



-----

### Implementations
