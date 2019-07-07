---
layout: article
title: "Pixie 리뷰"
category: "Recommender Systems"
tag: "Recommender Systems"
mathjax: true

---



### Introduction
![과녁]({{ "/assets/images/bv/bv1.png" | absolute_url }})
![그래프]({{ "/assets/images/bv/bv2.png" | absolute_url }})

기계학습을 공부하고 있는 사람은 위의 두 그림을 마주친 적이 분명 있을 것이다
~~일단 난 적어도 100번은 봤어..~~

나만 그런지는 모르겠는데, Bias-Variance Tradeoff라는 단어를 사람들이 조금씩 미묘하게 다른 의미로 사용하는 것 같다는 생각이 들었다. (내가 수리통계학를 헛배웠나 싶었다).

사실 실제로 (실제로는 같은 현상이지만) Bias-Variance Tradeoff라는 단어는 두 가지 현상을 가리킨다. 나만 궁금했던게 아니라면, 다른 사람에게도 도움이 될 것 같아서 정리했다.


### Estimation에서의 Bias-Variance Tradeoff

##### 통계학 교과서에 많이 나오는 흔한 추론 과정:
배터리 10개의 수명을 알고있을 때, 실제 배터리의 평균 수명을 추론하는 방법은 여러 가지가 있을 수 있다.

1. 10개 배터리 수명의 평균치(Mean)
2. 10개 배터리 수명의 중앙값(Median)
(사실 더 많은 추정 방법이 있지만 여기선 추정 방법이 중요한게 아니다.)

우리의 추정값 (1), (2) 중 어떤게 더 좋은지 평가하고 싶다고 해 보자.
추정값이 더 좋으려면, 아마 $($추정값$ - \theta)^2$가 작을 것이다.

편의상 앞으로 추정값은 $p$라 하자.

우리가 추정하려는 배터리의 수명을 $\theta$라 두고, 평가를 하기 위해서 열심히 생각한 결과 배터리의 평균 수명은 어떠한 분포 $p(\theta)$를 따른다고 가정했다.
우리는 실제 $\theta$는 잘 모르지만, $\theta$**가 어떤 분포를 따른다고 가정했을 때, 추정값 (1), (2)중 어떤 것이 더 좋은지 판단하고 싶은 것이다.**


가장 흔히 쓰는 방법은 $\theta$에 대해 marginalize하는 것이다.
$E_{\theta}[($추정값$- \theta)^2]$에 대해 계산해본 후, 이 값을 비교하는 방법이 일반적으로 많이 사용되고, 이 값을 추정치의 Mean Squared Error라 한다. 이는 배터리의 평균 수명 뿐만이 아니라, **데이터를 기반으로 추정하려는 모든 값에 대해 적용할 수 있다.**

그리고, MSE는 다음과 같이 분해 가능하다.

$$
    MSE = E[(p - \theta)^2] = E[(p - E[p] + E[p] -  \theta)^2] = \\
    Var(p) + E[(E[p] - \theta)^2]
$$




### Model Complexity에서의 Bias-Variance Tradeoff


### 느낀점
다들 잘 알고 있고, 나만 헷갈리는 걸까? 아니면 다들 모르는데 그냥 넘어가는 걸까? 모르는 것도 모르는...건 아닐 것 같구...
그리고 나는 어디 가서 "기계학습 공부하고 있습니다"라고 말 하면 안될 것 같다.



### References:



[1] 수리통계학, 8장, 김우철.

[2] Pattern Recognition and Machine Learning, Chapter 2, 3 Bishop.