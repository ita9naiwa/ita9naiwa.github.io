---
layout: article
title: "NeuralSort:"
category: "recommender systems"
tag: "recommender systems"
mathjax: true

---


### 잡담
ICLR 2019에 실린 [Stochastic Optimization of Sorting Networks via Continuous Relaxations
](https://arxiv.org/abs/1903.08850)의 리뷰.
글을 안 쓴 지 너무 오래되서 글을 쓰는 방법을 잊어버린 것 같다.;;


### Motivation
정렬이나 top-k element를 선택하는 일은 여러 머신 러닝 task에서 기본적인 연산 중 하나이다.

그런 예시 중 하나로, vector $q$가 주어졌을 때, 주어진 query $q$와 비슷한 여러 vector $v_1, \cdots, v_n$을 KNN을 생각할 수 있다.

이는 각각의 vector $v_i$에 대해 score  $s(q, v_i) = q^Tv_i \text{ or } (q-v_i)^2$를 계산해서 score를 정렬해서 가장 큰/혹은 작은 k개의 아이템을 돌려주는 방식이 가장 일반적이다.

이런 Sorting은 보통 미분가능하지 않아서 Gradient Based Optimization을 할 수가 없다.

그래서 IR/RecSys에서는 보통 이런 방식으로 업데이트를 한다.

- MSE loss: Minimize $[s(q, v) - y(q, v)]^2$  $y(q,v)$ if $v$ is relevant to $q$
- cross-entropy loss: Minimize $\textbf{I}[y(q, v) = 1]\log s(q,v) + \textbf{I}[y(q,v) = 0]\log(1 - s(q,v))$

위에 적힌 두 Objective가 실제로 사용되는 metric이 아닌 surrogate function이고 내가 정말 필요로 하는 목적과는 다르다.

(sorting, top-k search)에 end-to-end tailored되어 있지 않다는 문제가 있다.
![accuracy-loss](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2019/01/Line-Plot-Showing-Learning-Curves-of-Loss-and-Accuracy-of-the-MLP-on-the-Two-Circles-Problem-During-Training.png)
정확히 매칭되는 그림은 아닌데, 아무튼 이런 현상이 존재한다. Epoch 100 이후에 Loss는 계속 줄고 있지만, Accuracy는 더 이상 거의 증가하지 않는다.

이 논문에서는 미분 가능한 soft-정렬 연산자를 제시함으로 이런 문제를 해결하고, End-to-end style로 Sort/Permutation/TopK를 optimize하기 위한 미분 가능한 Operator를 제시했다.

### Objective involving permutation

#### Permutation
Permutation $\pi$란 [1, 3, 2]와 같이 $1$부터 $n$이 하나씩 유일하게 들어있는 vector를 말한다.

$\pi=[3,2,1]$일 때, 다른 array $s [a, b, c]$에 $\pi$를 적용하면, $[c,b,a]$가 된다. 보통 python에서 자주 쓰는 notaiton으로 적는다면, $[c, b, a] = s[\pi]$가 되고, $s$에 대한 내림차순 정렬이 된다. 앞으로 우리가 관심있는 Permutation은 대체로 정렬이므로, 편의상  $z$를 $s$에 대한 정렬 permutation, $z=sort(s)$이라 하자.

#### Represent Permuation as Matrix

이런 permutation은 또한 matrix로 나타낼 수 있다.

$$
P^{(z)}s = \begin{bmatrix}
0 & 0 & 1  \\
0 & 1 & 0 \\
1 & 0 & 0
\end{bmatrix}[a, b, c]^T = [c, b, a]^T
$$

$P^{(z)}{[k, :]}$는 $j$'th element가 1이고, 나머지는 다 0이다. $s$에서 $k$번째로 큰 element는 $s_j$라는 의미이다.

이런 정렬을 하는 matrix를 $P_{z}$라 하자.

P는 이렇게 계산할 수 있다.
$$
	P_{[i, j]} = \begin{cases}1 & \text{if} ~j = \argmax[(n+1 - 2i)s - A\mathbf{1} \\0  & \text{otherwise}\end{cases}
$$
where $A_{[i, j]} = \mid s_i - s_j \mid$.

불연속적인 $P^{(z)}$의 row를 relax하는 식으로 Permutation 혹은 Sort를 미분 가능하게 바꾸는 간단한 방법을 제안했다. 그냥 argmax를 softmax with temperature $\tau$로 바꿔준 꼴이고, 엄청 간단하다.

$$
	\hat P_{[i, :]} = \begin{cases}1 & \text{if} ~j = \text{softmax}\left[\tau^{-1}[(n+1 - 2i)s - A\mathbf{1}] \right]\\0  & \text{otherwise}\end{cases}
$$ where $A_{[i, j]} = \mid s_i - s_j \mid$

이러한 미분 불가능한 연산(정렬)을 미분 가능한 연산으로 relax하는 일반적인 방법은 이를 continuous한 공간에 매핑하는 것이다. 비슷한 예시로는, 흔히 사용되는 Classification objective가 있다.

- 0/1 loss for binary classification -> logistic or hinge loss
	- thresholding gives a mapping from real value to discrete decision.

이 matrix의 다음과 같은 성질을 보존하는 Real-value matrix를 제안햇다.
$\hat P$는 다음과 같은 성질을 만족하는 행렬이다. ($P$도 이를 만족한다)

1. Non-negativity: $U_{[i, j]} \geq 0 \forall i,j\in \{1, 2,..., n\}$
2. Row Affinity: $\sum_{j=1}^n U_{[i, j]}= 1$
3. Argmax Permuatiotn: $u = [u_1, u_2, ..., u_n]$ where $u_i = \argmax_j(U_{[i,:]})$  then $u$ is a valid permutation of $\{n\}$

(1), (2)는 discrete permutation에 대한 stochastic relaxation을 의미한다.
$P_{[k, :]}$는 $j$'th elementh가 1이고, 나머지는 다 0이다. 즉, $s$에서 $k$번째로 큰 element는 $s_j$라는 의미였는데, $\hat P_{[k, :]}$는 $k$번째로 큰 element가 무엇인지에 대한 확률분포(에 가까운 것)이라 생각할 수 있다. $\hat P$는 argmax로 정의되어 있던 $P$의 i번째 row를 Softmax로 대체한 것이다.


- $u = [u_1, ..., u_n]$, where $u_i = \argmax(P_{[i, :]})$ , then $u$ = $\text{sort}(s)$
- $\tau \rightarrow +0$인 경우 $\hat P \rightarrow P$이다.

$\hat P$은 *미분 가능한 정렬 연산자*이다.


### Gumbel Trick을 이용한 Stochastic Permuation and model update
- $P_x$: vector $x$에 대한 permutation matrix.
- $\hat P_x :$ vector $x$에 대한 relaxed permuation matrix.

#### Gumbel Distribution
$$
	g = gumbel(0, 1) =  -\log(-\log(\text{uniform}(0, 1))
$$
이다. 길이가 $n$인 vector $v$가 있고, $n$개의 i.i.d 한 $g_1, g_2, ..., g_n$이 있다고 하자.

$I_1 = \argmax([v_1 + g_1, v_2 + g_2, ..., v_n + g_n]$)$이면,

$$\Pr(I_1 = i) = \frac{\exp(v_i)}{\sum_j\exp(v_j)}$$

이 성립한다. 이는 조금 더 일반화될 수 있다.

k 이 n보다 작거나 같을 때, $I_1, ..., I_k = \text{argtop-k}([v_1 + g_1, v_2 + g_2, ..., v_n + g_n])$이라면,
$$
\Pr(I_1 = i_1, I_2 = i_2, ..., I_k = i_k) = \prod_{i=1}^k \frac{\exp(v_i)}{\sum_{j \notin (I_1,..., I_{k-1})} \exp(v_j)}
$$
가 성립한다.

$v = \log(s)$로 생각하고, $v + g$에 대한 Relaxed Permutation matrix를 여러 개 만들어서 Objective를 업데이트해도 괜찮은 것이다.

$$
	L = E_{q(z|s)}[f(P_z)] \\
	= E_g[f(P_{sort(\log s + z)})] \\
	\simeq E_g[f(\hat P_{sort(\log s + z)})] \\
$$

$$
	\nabla_s L = E_g[ \nabla_s f(\hat P_{sort(\log s + z)})]
$$이 성립한다. (By REINFORCE)



### 읽고 든 생각

1. [Neuralsort Github](https://github.com/ermongroup/neuralsort)에 구현체가 있는데, 코드가 그렇게 어렵지 않아서, 읽어보면 재밌고, 내가 하는 연구에 어떻게던 써먹을 여지가 있지 않을까 하는 생각이 들었다.

2. 내가 아는 한에서 미분 불가능한 Objective(주로 combinatorial problem)을 Neural Network으로 푸는 방법은 REINFORCE를 이용하는 거였다.

$\nabla_\theta L = E_{z \sim q}[f(z) \nabla_{\theta} \log (q(z \| s; \theta)]$

이 식을

$\nabla_{\beta} L = E_{z \sim q}[f(z;\beta) \nabla_{\beta} \log (q(z \| s; \theta)] + E_{z \sim q}[\nabla_{\beta} f(z;\beta)]$로 생각해서, $\beta$에 대해 optimize를 하는 방식이라 되게 신기했다. 비슷한 류의 work이 몇개 더 있는데, Gumbel-Trick을 이용해 without replacement의 Categorical Distribution을 sampling하는 방법도 있고...
