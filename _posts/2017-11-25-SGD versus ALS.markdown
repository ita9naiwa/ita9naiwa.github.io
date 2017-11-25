---
layout: post
title: "SGD vs. ALS on solving BiasedMF(SVD)"
category: "recommender systems"
tag: "recommender systems"
mathjax: true
---


추천 시스템에서 가장 흔히 사용되는 Matrix Factorization은 Rating matrix $R$을 두 매트릭스 \(U ,V \)로 factorize한다.

그러니까, 흠... 유저의 예상 평점을 맞추는 문제는 Rating matrix를 잘 Reconstruct하는

![image](http://shubham.chaudhary.xyz/blog/img/recommenders/matrix-decomposition.png)

$$
R  \simeq UV^T \text{, where }R_{ui} \neq 0
$$

가 되는 좋은 \(U\)와 \(V\)를 찾는 문제로 바꿀 수 있다. 이는

$$
L(U,V) = \sum_{u,i}{(R_{u,i} - u_u^Tv_i)}
$$

이 Loss function $L(U,V) $을 최소화하는 걸로 바꿀 수 있다. (사실 이건, 주어진 데이터에 대한 Gaussian Prior를 갖는 model의 MLE를 구하는 과정으로 볼 수 있다.) 이에 대해서는 기회가 될 때 더 자세히 적는 게 좋은 것 같고...
Loss function을 minimize하는 방법은 가장 대표적으로 [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent)방법이 있다. 이는 현재 Machine Learning, Optimization 분야에서 가장 기본적이고, 실제로도 가장 많이 사용되는 방법이다. 대체로 아주 많은 상황에서 $f(x)$일 때 $f(x)$를 충분히 작게 하는 $x$을 잘 찾는다고 알려져 있다.

Gradient Descent 방법의 가장 큰 장점은, 임의의 `미분 가능한` 함수를 최소화하는 값을 찾는(앞으로 `최소값을 찾는다`고 표현하겠다) 일을 할 수 있다는 점이다. 복잡한 함수에 대해서도, 미분이 가능하다면 꽤 그럴싸한 답을 찾아낸다. 단점으로는

1) Optimal solution을 찾지 않는 Greedy한 방법이라는 점.
2) Training 시간이 비교적 길다는 점 (이는 대안이 그다지 없다)

Matrix Factorization 방법에서 최소값을 찾고자 하는 값은 비교적 간단하다.
주어진 로스 함수는 user vector  \(avasdv\) u  와 item vector \(v\) 의 Quadratic form으로 나타낼 수 있다. 
고등학교 수학 시간에 배웠던 것 같은데, 2차함수는 항상 최소값을 갖는다. 2차함수 \(f(x) \) 의 최소값은 \(x'\) 가 \( f(x') \text{where }f(x')'= 0 \) 을 만족시키는 값에서이다.

$$
\begin{eqnarray} 
L(U,V) = \sum_{u,i}{R_{u,i} - u_u^Tv_i} \\
= 
\end{eqnarray}
$$

