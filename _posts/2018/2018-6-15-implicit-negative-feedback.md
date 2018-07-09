---
layout: article
title: "Implicit Negative Feedback In Bayesian Personalized Ranking"
category: "recommender systems"
tag: "recommender systems"
mathjax: true
comment: true
key: nfimf
---
#### 잡담
이거 진짜 내 일이랑은 하나도 상관 없는 것 아닐까?
이런 글 쓰는 동안에 딥러닝 공부를 조금 더 하는게 도움이 더 되는 것 아닐까? 생각했지만...
역시 관심도 없는 거 하는 것보다는, 별로 도움이 안 되더라도 관심가는 걸 공부해야겠지 하는 생각이 들었다.
Negative Feedback의 활용에 대해서는, 예전부터 관심이 많았으니까.

그리고, interaction matrix의 Sparsity를 고려한다면, 점점 더 다양한 종류의 데이터를 활용하는 것이 중요해질 것 같다고 생각한다.

그렇게 생각하는 이유는 다음과 같다.

1. 항상 CNN/RNN 모델을 돌릴 수는 없고, 여전히 CF 모델의 성능이 content based method보다 성능이 좋다.
2. user와 item의 interaction의 종류가 **같은 서비스 내에서** 증가하고 있다.
  - Youtube를 예로 들자면, 유저가 한 동영상에 대해 취할 수 있는 action은 "클릭", "시청하기(몇분 동안?)" "좋아요", "싫어요", "플레이리스트에 추가하기", 그리고 "공유"로 상당히 많다.
  다양한 Interaction을 다양한 방법으로 해석할 수 있고, 다양한 해석을 통해 더 좋은 CF 모델을 만드는 것이 가능하다고 생각한다.





## Introduction
State-of-the-art Collaborative Filtering 기술은 대부분 implicit feedback을 이용하는 것이 최근의 추세이다.
(Implcit CF, Personalized Ranking, One-class CF와 같은 다양한 이름을 갖고 있다)
즉, 유저가 자신이 좋아한 혹은 interaction한 정보만을 이용해 유저에게 아이템을 추천한다.
유저가 추천받지 않을 아이템을 선택할 수 있긴 하지만, 이는 모델 자체가 갖는 정보가 아닌 모델이 추천을 생성한 결과를 post-processing하는 것이다.
(e.g. Evaluation 시 유저가 이미 본 아이템을 필터링하는 것도 이와 같은 post-processing 과정 중 일부라 볼 수 있다.)

위 제안된 방법의 한계 중 하나는 유저가 남긴 positive feedback만을 모델의 input으로 사용하고 있다는 점이다.

실제 웹 사이트나 커머스 사이트에서는 제공하는 컨텐츠에 대해 '좋아요'와 같은 positve feedback 뿐만이 아니라, '싫어요'와 같은 negative feedback을 남기는 기능 또한 제공하고 있다.[네거티브 피드백에 관해 이전에 남긴 포스트](https://ita9naiwa.github.io/recommender%20systems/2018/05/29/Why.html)
하지만 WMF나, BPR 같은 모델은 이러한 negative feedback을 활용할 수 없다. 개인적으로는 이런 데이터들이 모델에 직접 이용되지 못하는 점이 너무 아쉽다. 추천 시스템에서 사용하는 input 데이터의 본질적인 특성인 sparsity 때문에, 추천 엔진을 만드는 사람은 항상 **조금이라도 더 많은 데이터를 이용하고 싶기 때문이다**

다행히도 negative feedback이 버려지는 것을 아쉬워하는 사람이 나 혼자는 아니었다. 추천 시스템 알고리즘 중 가장 많이 사용되는 WMF 모델에 Negative feedback을 적용하려는 시도가 존재한다. link [ #1](https://gite.lirmm.fr/yagoubi/spark/commit/9e63f80e75bb6d9bbe6df268908c3219de6852d9),
[#2](https://github.com/benfred/implicit/issues/114)

> 근데 이게 왜 논문이 아니라, open-source project에 issue로 먼저 제기되었는지 모르겠다.(관련 논문이 있을지도 모르겠지만 찾지 못했음) 뭔가 실용적인 이슈가 있어서 그런 걸까?..

이 글에서는 다음과 같은 두 부분으로 이루어져 있다.
1. Weighted Matrix Factorizaition 모델에 negative feedback을 반영하기
2. Bayesian Personalized Ranking 모델이 negative feedback을 이용할 수 있게 확장하기

*이 글에서는 당신이 추천 시스템에 관해, 또한 자주 사용되는 모델에 대한 전반적인 지식이 있다고 가정한다.*

결론을 미리 얘기하자면, Explicit Negative feedback은 계륵같은 점이 있는 것 같다.
CF 모델에서, implicit feedback만을 이용하는 것이 아니라, 다른 데이터를 가져와서 이용하고자 하는 노력이 지금까지 있었고(그야 interaction은 본질적으로 sparse하니까.), 앞으로도 계속 이어질 것이라는 점이다.
나도 계속 이 부분을 공부하고 싶다.

> **Note:**
> 글에서 적는 내용 중 일부는 exhaustive하게 테스트되지 않은 내용이 일부 포함되어 있다.
> "이 방법 괜찮을까?"라는 개인적인 질문의 개인적인 대답이다.
> 하지만 이러한 질문을 갖는 것과, 질문을 해결하기 위해 시도해 본 방법이 이 글을 읽는 분에게 insight을 줄 수 있다고 생각해 이 내용을 공유한다.


## Negative Feedback in Factor Based Models




### 가설
- 다양한 상황에서 유저 u는 item i와 negative interaction을 가질 수 있다.
  - negative interaction은 positive interaction과 symmetric하다.
- 우리는 이 negative interaction을 model에 포함하고 싶다.

| interaction category \ type |  Positive Interaction        | Negative Interaction         |
|-----------------------------|------------------------------|------------------------------|
| Preference                  | Liked                        | Disliked                     |
| Rating                      | Higher rating(either 4 or 5) | Lower rating (lower than 3)  |
| Shoppoing                   | Purchased                    | Refunded                     |
| Video                       | Watched                      | Watched during only few secs |

즉, 어떤 interaction은 positive한 signal이라 해석할 수 있지만, **어떤 interaction은 negative signal이라 해석할 수 있다.**



## Negative Feedback in WMF
### 모델을 어떻게 확장할까?
그럼 우선, WMF 모델에서 positive feedback을 어떻게 다루는지 refresh를 해 보자.
WMF 모델에서는 유저 $$u$$와 item $$i$$ 간의 interaction이 존재하면 input value $$y_{u,i} = 1$$이며,
interaction의 intensity에 따라 confidence $$c_{u,i} > 1$$ 값의 크기를 조정한다.
user $$u$$와 item $$i$$ 간의 interaction이 존재하지 않으면 input value $$y_{u,i} = 0$$이며, 유저가 그 아이템의 존재를 모르는지, 아니면 실제로 그 아이템을 싫어하는지 확신할 수 없으므로 positive interaction일 때의 confidence보다 낮은 confidence $$c_{u,i} = 1$$을 갖게 만든다.

> confidence 값이 이렇게 정의된 이유는 단지 이렇게 하는 것이 더 단순하기 때문이다.
> positive feedback의 수가 positive feedback의 수보다 압도적으로 적으므로 positive feedback의 confidence를 높이는 것이 더 편리하다.
> 사실 positive interaction의 confidence가 negative interaction의 confidence보다 크면 된다.

이를 식으로 표현하자면 다음과 같다.

### WMF가 input을 처리하는 방법

$$
\begin{cases}
y_{u,i} = 1 & \text{user } u \text{ positively interacted with item } i        \\
y_{u,i} = 0 &\text{otherwise}
\end{cases}
$$

$$
\begin{cases}
c_{u,i} > 1 & \text{user } u \text{ clicked item } i        \\
c_{u,i} = 1 &\text{otherwise}
\end{cases}
$$

여기서 중요한 점은 user u와 item i의 interaction이 없는 pair의 confidence가 낮은 이유이다.

**유저가 그 아이템을 실제로 싫어하는지 확신할 수 없다!**

이 부분에서 negative interaction이 활약할 차례이다.
유저가 아이템이 싫다고 직접 표현했다면, 유저가 그 아이템을 좋아하지 않는다고 조금 더 확신을 갖고 말할 수 있게 된다.
(말 그대로 with higher confidence ㅋㅋㅋ...)

즉, 유저 u가 싫다고 표현한 item i의 confidence를 높여주는 것만으로, 모델을 확장할 수 있게 된다.

### WMF with negative interactions

#### Case 1
$$
\begin{cases}
y_{u,i} = 1 & \text{user } u \text{ positively interacted with item } i        \\
y_{u,i} = 0 & \text{user } u \text{ negatively interacted with item } i        \\
y_{u,i} = 0 &\text{otherwise}
\end{cases}
$$

$$
\begin{cases}
c_{u,i} > 1 & \text{user } u \text{ interacted item } i        \\
c_{u,i} = 1 & \text{otherwise}
\end{cases}
$$

변한 부분은 user의 potisive interaction과 negative interaction을 구분해 다른 $$y_{u,i}$$ 값을 갖게 한 것 뿐이다.
이것만으로, user가 좋아하지 않는다고 표현한 item을 직접적으로 모델에 포함시킬 수 있다...!

#### Case 2
***검증되지 않은 내용입니다.***

위 모델은 유저의 preference를 어떤 아이템을 좋아하거나, 싫어하는 binary preference만 존재한다고 가정한다. preference가 binary하지 않다고 가정함으로서, WMF 모델을 다른 방법으로도 모델을 확장할 수 있다.

$$
\begin{cases}
y_{u,i} = 1 & \text{user } u \text{ positively interacted with item }   i       \\
y_{u,i} = -1 & \text{user } u \text{ negatively interacted with item  } i       \\
y_{u,i} = 0 &\text{otherwise}
\end{cases}
$$

$$
\begin{cases}
c_{u,i} > 1 & \text{user } u \text{ interacted item } i        \\
c_{u,i} = 1 & \text{otherwise}
\end{cases}
$$

user가 어떤 아이템에 대해 (1) "좋아한다"와 (2) "싫어한다" 뿐만이 아니라, (3) "무관심하다"는 새로운 카테고리를 만들어 기존 모델에서 "싫어한다"고 정의하던 item을 "무관심하다"고 가정한다.
유저가 실제로 "싫어한다"(negatively interacted)는 행동을 보인 item을 user가 싫어하는 아이템이라 정의하는 방법으로 모델을 확장할 수 있다.

## Negative Feedback in BPR
WMF 모델이 비슷한 식으로 확장될 수 있다면, BPR 모델도 같은 방법으로 확장할 수 있지 않을까? 라는 생각이 들었다.
이는 간단하게 구현될 수 있고, 비슷한 방법이 아무래도 있지 않나? 하는 생각이 들었는데, **신기하게도 이 방법을 시도해 본 사람이 아직은 아무도 없었다.**
간단한 실험을 추가해서 첨부한다.

### 원래 방법
BPR(*for implicit matrix factorization*)의 가정과, 가정에 따라 모델이 갖는 objective는 다음과 같다.

User u와 interaction이 있는 item들의 집합 $$I_u = \{i_1, i_2, ..., i_n\}$$가 있고,
User u와의 interaction이 없는 item들의 집합 $$J_{u} = \{j_1, j_2, ..., j_m\}$$이 있을 때, 유저는 item i $$i \in I_u$$를 $$j \in J_u$$보다 선호한다.

이를 편의상,
$$
i >_ {u} j  \text{ where } i \in I_u \text{ and } j \in J_u
$$
라 표현하자.

### BPR with negative interactions
WMF와 마찬가지로, BPR 모델은 interaction을 "좋아한다"와 "싫어한다" 두 가지로 나뉜다.
\#Case 2에서 했던 것처럼 interaction의 종류를 3가지로 나누어 보자.
전체 아이템을 다음과 같은 세 집합 중 하나에 포함시킬 수 있다.

1. $$I_u = \{i_1, i_2, ..., i_n\}$$ user u가 positive하게 interact한 item의 집합
2. $$J_u = \{j_1, j_2, ..., j_m\}$$ user u가 interact하지 않은 item의 집합
3. $$K_u = \{k_1, k_2, ..., k_l\}$$ user u가 negative하게 interact한 item의 집합

편의상... 하나의 notation을 더 추가하자면, $$i, j, k$$는 $$I_u, J_u, k_u$$ 내의 임의의 item이라 하자.

1. $$i >_ {u} j, k $$ -> 이는 원래 BPR의 가정
2. $$i, j >_ {u} k$$ -> 확장 2

(2) 부분을 추가함으로서, BPR 모델이 Negative feedback을 고려할 수 있게 만들었다..!


#### Implementaiton
WMF imlementation은 [Implicit](https://github.com/benfred/implicit) 라이브러리를 그대로 사용했다.

BPR implemenation은 [https://github.com/ita9naiwa/implicit](https://github.com/ita9naiwa/implicit) 내의
implicit/bpr.pyx에 구현했다. 사실 위에서 말한 내용을 그대로 구현하면 되는데, 속도를 조금 빠르게 하는 부분이 조금 불편하다.
혹시 실험을 재현해보고 싶다면 ita9naiwa@gmail.com으로 메일 부탁드립니다.


#### 간단한 실험

**ml-1m dataset에서의 실험 결과**

||||
|------------------------	|--------	|--------------------------	|
| metric \ method        	| BPR    	| BPR with negative sample 	|
| precision@3            	| 0.2744 	| 0.3045 (10% increased)   	|
| precision@5            	| 0.2485 	| 0.2760 (10% increased)   	|
| false discovery rate@3 	| 0.0104 	| 0.0045 (131% decresed)   	|
| false discovery rate@5 	| 0.0106 	| 0.0048 (120% decreased)  	|



#### 엄청 짧은 디스커션
