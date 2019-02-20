---
layout: article
title: "Learning Scheduling Algorithms for Data Processing Clusters 리뷰"
category: "ML Application"
tag: "ML Application"
mathjax: true

---

### 잡담
최근에 연구실에 들어가면서, 공부하는 주제가 바뀌게 되었다. 사실 내가 연구라고 말할 걸 하고 있지는 않지만(하고는 싶지만) 'ㅅ'....공부하는 것들이 낯설다. 낯선 주제이므로, 논문 읽는데 시간이 많이 걸리기도 하고, 뭘 읽어야 하는지도 모르겠고...라는 핑계를 대며 이리저리 놀고만 있었는데, 그러면 안될 것 같다.

### 개요
RL은 정말 많은 일을 잘 할 수 있다고 많이 밝혀지고 있다. 현재는 주로 로보틱스나 게임이지만. 하지만 RL을 Traditional한 환경에 적용할 수 있다고 믿는 사람이 많다. 내 지도교수님도 그렇게 생각한다. 나도 조금은, 그렇게 생각하게 되었다.

이 논문은, 강화학습을 Cluster Scheduling에 대해 적용한 논문이다. Scheduling은 보통, job을 machine에 할당하는 일이다. (나는 잘 모르겠지만, [Job Shop Scheduling](https://en.wikipedia.org/wiki/Job_shop_scheduling)이라고 해서 경영과학 분야에서 많이 연구되고 있는 분야라 한다.)  사실, job이나 machine이 어떻게 정의되는 지에 따라 Scheduling하는 방법이 다르므로, 일반적으로 항상 좋은 스케쥴링 알고리즘은 존재하지 않는다. 논문에서도, 특정한 종류의 Query가 들어오는 Cluster에 대해서만 정의한다. 아무튼, job과, machine을 정의하면 풀려는 문제가 명확해진다.

### 문제 정의
이 논문에서의 Job과 Machine은 Spark의 Query과 Executor와 정확히 일치한다. 논문에서 실험도 실제 스파크 환경에서 실험했다고 한다. 나는 스파크 internals에 대해 잘 모르니, 간단하게 정리해보았다.

#### Job
이 논문에서의 Job의 정의는, Spark나 hadoop같은 데에서 사용하는 query와 같다. 여러 개의 Query $q$가 $m$개의 machines 위에서 돌아간다. 각각의 query는 여러 개의 stage(작은 job)로 이루어져 있고, 각각의 Stage 사이에는 Dependency가 존재한다. 또, 하나의 stage는 작게 나눠서 처리할 수 있다. 어느 정도는 concurrent하게 동작할 수 있다고 한다. 즉, executor를 많이 할당할 수록 stage를 빨리 처리할 수 있다. 하지만, 너무 높은 Concurrency는 큰 도움이 되지 않는다.

#### Machine
Stage 내의 small task를 수행할 수 있다. machine도 수가 많지만, 모든 수행 가능한 task를 동시에 수행하기엔 수가 부족하다.

풀고자 하는 문제는 다음과 같다.

- Batch: n개가 미리 주어져 있는 경우, 어떻게 이 n개의 job이 빨리 처리되도록  executor를 잘 배치하고 싶다.
- Streaming: 시간에 따라 job이 계속 들어온다. 들어오는 job들이 빨리 처리되도록 exeutor를 잘 배치하고 싶다.
	- 두 경우 모두, 문제에서 정의한 Metric은 개별 job의 Completion time이다.

### Motivation

1. Handling DAG-structured Jobs
여러 Data processing systems가 만드는 쿼리는 꽤나 복잡하다. (스파크의 용어를 빌리자면) 하나의 쿼리는 여러 작은 연산(Stages)으로 이루어져 있고, 각각의 연산은 이전 연산에 dependent하다. 이런 걸 Directed Acyclic Graph라고 한다.
![DAG Execution](https://cdn.datafloq.com/cms/2017/08/09/dag-execution.png)

이러한 Stage 간의 의존성을 잘 고려해서 스케쥴링을 하면 당연히! 성능이 더 좋겠지만, 동시에 많은 job을 처리해야 하는 상황에서 그러한 요소를 반영하는 스케쥴러를 만들기 어려웠다. 근데 최근의 기계학습/강화학습 방법론을 사용하면 잘 할 수 있지 않을까?

2. Appropriate Parallelism
![Figure 2: TPC-H queries scale differently with parallelism: Q9 on a 100 GB input sees speedups up to 40 parallel tasks, while Q2 stops gaining at 20 tasks; Q9 on a 2 GB input needs only 5 tasks. Picking â€œsweet spotsâ€ on these curves for a mixed workload is non-trivial.](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/3dec6f21dfd04fb4207ec6811706a7c9babaa426/3-Figure2-1.png)
아까 한 Stage에 여러 executor를 할당할 수 있다고 했는데, 얘는 Diminishing Return이다. 근데, input이 왕창 큰 job같은 경우 parallelism을 활용하기 좋겠고, 반대인 경우 별로일 수도 있을 것이다. 이런 정보를 잘 반영하는 휴리스틱을 찾기는 어려운데, 최근의 기계학습/강화학습이라면 뭔가 해내지 않을까?

~~사실 이쯤되면 기계학습이 *"어이어이  **기계학습쿤**, 믿고 있었다구?"* 정도의 만능 도구라 생각하는게 아닐까...~~
또 잘 되기도 하고... 근데 난 이렇게 잘 되는 걸 못 찾고.....

## 제안된 방법

### Network 구성
우선 먼저 요약하자면, 
1. Job들을 graph embedding 방법을 써서 embedding한다. (DAG 전체(Query)/ 노드(Stage) 별 각각)
2. 어떤 Stage를 스케쥴링할 지 정하고, 얼마나 많은 Executor를 할당할 지 정한다.
3. 이를 RL로 트레이닝한다.
이다. 한번에 쭉 읽자니 어려웠는데, 단계별로 나눠서 읽어보니 생각보다 쉬웠으므로, 단계별로 정리하자 ^^;;

#### 1. Job들을 embedding하기.



![Figure 5: Graph embedding transforms raw information each node of job DAGs into a vector representation. This example shows two steps of local message passing and two levels of summarizations.](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/3dec6f21dfd04fb4207ec6811706a7c9babaa426/5-Figure5-1.png)



job $i$는 graph $G$로 표현할 수 있고, ${v_1, v_2, ...v_n}$개의 stage(그래프적으로 얘기하자면 노드)를 갖고 있다.


stage $v$의 embedding $e_v$는 다음과 같이 정의된다.
$$e_v = g(\sum_{w\in \xi(v)}f(e_w)) + x_v$$

$x_v$ 는 stage $v$의 input feature vector로, (1) # of tasks, (2) average task duration, (3) # of executors currently working on the stage, (4) # of available executors, (5) whether available executors are local to job으로 정의되어 있다.
$f, g$는 각각 하나의 neural network이다.



Job $i$의 embedding $y_i$는 다음과 같이 정의된다.
$$y_i = g'(\sum_{v\in G_i}f'(e_v)) $$
$f', g'$는 각각 하나의 neural network이다.

사실 이 식이 정확한지는 잘 모르겠다. 위와 같은 방법으로 비슷하게 정의할 수 있다고 적어놔서... 아마 크게 다르지는 않을 것이다.
모든 job의 Global summary $z$는 다음과 같이 정의된다.
$$z = g''(\sum_{i}f''(y_i)) $$ 

위 내용을 다음과 같이 표현할 수 있다.


![Figure 6: For each node v in job i, the node selection network uses the message passing summary eiv , DAG summary yi and global summary z to compute a priority score qiv used to sample a node.](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/3dec6f21dfd04fb4207ec6811706a7c9babaa426/6-Figure6-1.png)


#### 2.어떤 stage를 스케쥴링하고, 얼마나 많은 Executor를 할당할까?

이번에 어떤 stage i을 scheduling할지는 다음과 같은 확률 분포로 정의한다.
$$\Pr[a_t= v] = \frac{\exp(q(e_v, y_{p(v)}, z))]}{\sum_{w}\exp(q(e_w, y_{p(w)}, z))]}$$
$p(v)$는 $v$를 포함하는 job $i$를 가리킨다.


Exectuor를 할당할 때, Stage 레벨에서  executor를 할당할 수도 있지만, 여기서는 Job 레벨에서 Executor를 할당한다. 사실, job 레벨에서 Executor 수를 정해 놓으면, 그 밑의 단계(stage 단계)에서의 parallelism은 spark scheduler가 알아서 정해주기도 하고(확실하지 않음)...이렇게 처리하는 것이, inference time을 엄청나게 줄여준다고 한다.


![Figure 15: Without the domain-specific conditional probability insight (Â§5.2), Decimaâ€™s inference time grows with cluster size.](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/3dec6f21dfd04fb4207ec6811706a7c9babaa426/12-Figure15-1.png)

세로축이 inference time인데, 위 방법을 적용한 경우 executor의 수가 늘어도 예측 시간에 그리 큰 변화가 생기지 않는다는 사실을 알 수 있다.

job level에서의 얼마나 많은 Executor를 할당할지는 다음과 같이 정의한다.
$$\Pr[b_t ==N] = \text{MLP}(y_i)$$
Output이 $[1, 2, ..., N]$인 MLP를 정의해 이를 그대로 사용한다. 모든 Node에 대해 이 MLP를 계산해야 한다면, 확실히 양이 많긴 할 것 같다.

이를 그림으로 나타내면 다음과 같다.

![Figure 7: Decimaâ€™s policy for jointly sampling a node and parallelism limit is implemented as the product of a node distribution, computed from graph embeddings (Â§5.1), and a limit distribution, computed from the DAG summaries.](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/3dec6f21dfd04fb4207ec6811706a7c9babaa426/7-Figure7-1.png)


### 강화 학습 환경 구성

우선 먼저 요약하자면 다음과 같다.
- State observation:
	- DAG로 구성된 job list와, 간략한 클러스터의 구성 정보(자세히 나와있지는 않은 것 같다)
- Action:
	1) 어떤 Stage에 job을 할당할지
	2) 몇개의 Exectuor를 할당할지. 총 2차원이다.
- Reward:
	-  $-\tau \times J$, $\tau$는 last action 이후로 지난 시간(이 논문의 구현에서는 'second' 단위),  $J$는 종료되지 않은 job의 수(실행되고 있거나, pending되어 있거나 하는 Job들의 수)

State가 주어지고, agent는 어떤 액션을 취한다. environment는 agent가 가능한 액션이 있을 때마다 agent를 실행하는 방식이다. 가능한 액션이 생기는 이벤트는 (1)특정 job이 종료되었다(따라서 다른 job에 Executor를 할당할 수 있다), (2) 새로운 job이 큐에 추가되었다와 같은 경우를 말한다. 여기서 신기한 점은, **시간 단위**가 아니라, possible action 단위로 timestep을 구성하고 있다는 점이었다. 이렇게 하면 Return 계산이 번거로워지지 않을까 생각했지만, 크게 문제가 되지는 않는 것 같다.

#### Training 방법
[REINFORCE with baseline](https://jay.tech.blog/2017/01/03/policy-gradient-methods-part-1/) 알고리즘을 사용했다.
> Note
> 강화학습을 system 분야에 응용한 경우, 보통 REINFORCE 알고리즘을 사용하는 경우가 많다. 어디서 REINFORCE 알고리즘이 Robust하기 때문이라고 들은 적이 있는데, 잘은 모르겠다. 사실, 직접 구현을 하거나 실험을 한 경우, REINFORCE 알고리즘이 일반적으로 가장 Robust하긴 했다. 다만, 이 알고리즘의 capacity나, data efficiency가 좋지 않은 것이 문제지만.




input stream을 이용한 simulation 환경에서의 RL의 성능을 높일 수 있(다고 제안하)는 두 가지 방법이 있다.

#### 1. Sequence dependent baselines (저자가 이런 이름을 썼는지는 잘 모르겠다)
좋은 모델(제너럴라이즈를 잘 하는)을 만드려면, 다양한 인풋 시퀀스에 대해 모델을 트레이닝해야 한다. 하지만 job arrival pattern(어떤 시간에 어떤 job이 오는 지)가 reward에 주는 영향이 크다. 근데, 일반적인 RL 모델에서의 Critic, 혹은 Baseline은 이를 입력으로 삼지 않는다. 즉 이를 신경쓰지 않는 보통 critic을 만들면 variance가 왕창 커질 수 있다. 이 논문에서는 **각각의 input sequence당 baseline을 다시 만드는 방법으로 이를 해결했다.** 사실 이 문제 때문에 A3C라던가 PPO라던가 하는 Actor-Critic 방법을 사용하지 않은 것 같기도 하다.

#### 2. Differential Rewards
(discount==1이라 가정하면) 일반적인 RL 환경이라 생각할 때, 이 환경의 Objective는 $\Epsilon[-\sum_jT(j)]$, ($T(j)$는 j을 수행하는 데에 걸리는 시간)이다. 하지만, job scheduler와 같은 경우, 긴 긴 시간동안 서버 위에 떠 있고, 긴 긴 시간동안 들어오는 job들을 적당히 잘 처리하는 게 목표이다. 
즉, objective가 
$$\lim_{n->\inf} \Epsilon[-\frac{\sum_jT(j)}{n}]$$
이라 생각하는 것이 더 합리적일 것이다.
다행히도, 이렇게 formulation을 바꾸는 일이 그리 어렵지는 않다. Sutton책 10.3 챕터에 이를 간단히 바꾸는 방법을 소개해주고 있다. $\dot r := r_t - \hat r$,  $\hat r :=$ average value of $r$으로 바꿔주기만 하면 된다고 한다. 자세히는 설명하지 않겠지만(사실 나도 잘 모르겠다) 이렇게 reward formulation을 바꿔주는 것이 성능에 많은 영향을 미친다고 한다.

#### 실험 결과 

job completion time에서 더 우수하다고 하다 'ㅅ'...
![Figure 9: Decimaâ€™s learned scheduling policy achieves 21%â€“3.1Ã— lower average job completion time than baseline algorithms for 100 batches of 20 concurrent TPC-H jobs in a real Spark cluster.](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/3dec6f21dfd04fb4207ec6811706a7c9babaa426/9-Figure9-1.png)

논문에서 제안한 그래프 임베딩/ executor 수 정하는 두 모듈 둘 다 성능에 미치는 영향이 많다는 점을 보여주고 있다 'ㅅ'...
![Figure 14: Breakdown of the contributions due to key ideas in Decima. Omitting any of these idea results in Decima underperforming the tuned weighted fair policy.](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/3dec6f21dfd04fb4207ec6811706a7c9babaa426/12-Figure14-1.png)

### 느낀점

아이디어 자체도 좋지만, 이 두가지가 특별히 대단한 것 같다.
1. 아이디어를 실제로 구현해서 working result를 만들어냈다.
2. 정말 다양한 실험을 해서, 다양한 방법으로 이 결과가 우연이 아님을, 확실히 좋은 결과임을 보여주고 있다.
이 두가지 점을 배울 수 있다는 점에서, 도움이 많이 된 것 같다.
