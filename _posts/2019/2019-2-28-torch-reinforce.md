---
layout: article
title: "멀티프로세스 REINFORCE 알고리즘 구현
category: "RL"
tag: "RL"
mathjax: true

---

### 잡담

나도 블로그에 사람들이 많이 들어왔으면 좋겠다... 강화학습 얘기 하면 사람들 많이 들어올까... 근데 난 강화학습을 잘 몰라...
어떡하지....



### 목적

1. Rollout을 여러 프로세스에서 동시에 진행시키고 싶은데, [REINFORCE](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)의 가정은 깨고싶지 않다. 다른 distributed 환경을 위해 만들어진 알고리즘(A3C, PPO)와 같은 경우, A3C는 아마 rollout하는 policy들의 parameter가 다름을 감안하고 만든 알고리즘(인 것 ) 같고, PPO는 잘 모르겠지만, 암튼, Pytorch로 간단하게 Reinforce 알고리즘을 구현하고 싶었다. 예를 들어서, 이를 테면, 일반적으로 이런 방법을 많이 쓰는 것 같다.

### 간단한 해결 방법
Policy Gradient를 계산하기 위해서는,
$p(a|s)$, Return $R$, Advantage $\delta$가 필요하다.


개별 rollout worker는 다음과 같이 동작하면 될 것 같다고 생각해서, 일단 이렇게 만들어봤다.
```python
def rollout(model, env, queue):
	# do some meaningful things
	...
	queue.put((a, s, R, adv))
```

이렇게 하면 되나 싶었는데,  몇가지 문제점이 생긴다. 평상시엔 문제가 안 생길 수도 있지만, 왠지 문제가 생기는 이유는 다음과 같다.
**지금 환경은, 스테이트가 왕창왕창 짱짱크다.**

### s가 크고, 한번 $p(a|s)$을 계산하는 데에 시간이 왕창 걸린다.

1. **state가 큰 경우, state를 전달하다 에러가 난다.**
2.  accumulating threads에서 그래디언트 계산을 다시 해 줘야 한다. 근데 별로 그러고 싶지 않다.

굳이 한번 계산한 $p(a|s)$를 다시 계산해줘야 하나... 싶어서 고민을 했는데,

gradient를 직접 전달하면 괜찮은 것이었다!

rollout part
```python
def rollout(model, env, queue):
	local_model.zero_grad()
	loss.backward()
	grads = [p.grad for p in local_model.parameters()]
	queue.put(grads)
```

accumulation part
```python
for grads in list_of_grads:
for p, g in zip(global_model.parameters(), grads):
	p.grad += g
for p in global.model.parameters():
	p.grad /= len(list_of_grads)
```
이랬더니 에러가 나는고야 'ㅅ'...
`
line 135, in <module>
    main()
line 114, in main
    p.grad += g
TypeError: add(): argument 'other' (position 1) must be Tensor, not NoneType

`
그래서 찾아보니깐,
gradient 계산을 직접 하지 않으면(backward같은 계산 등으로), parameter 클래스의 grad 값은 None이라고 한다.
이걸 직접 초기화하는 건 조금 그래서, 다른 방식으로 어떻게 zero initialization을 하는데 잘 안되서,

결국

```python
for grads in list_of_grads:
for p, g in zip(global_model.parameters(), grads):
	if p.grad is  None:
		p.grad = g
	else:
		p.grad += g
for p in global.model.parameters():
p.grad /= len(list_of_grads)
```
이런 식으로 바꿔줬다.
일단 잘 동작하니 만족.

근데 이걸 더 잘 하는 방법이 있을 것 같은데..

암튼 결과는 [여기에 정리](https://gist.github.com/ita9naiwa/c4ad65931c8a49499671355351b79bce)

Tested on Torch 1.0.0
