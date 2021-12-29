---
layout: article
title: "개발자스럽게 공부하는 방법"
category: "ML"
tag: "ML"
header-includes:
  - \usepackage[fleqn]{amsmath}
mathjax: true
---

### 잡담

> 약간 웃기라고 가볍게 쓴 것도 있는데 실제로 이렇게 공부하면 도움이 됨

1. 공부 할 모티베이션이 잘 주어짐
2. 진짜로 PR 날렸을 때 받는 답변들은 배울 것들이 많음

------------------------

1. 공부할 거리를 찾는다(혹은 해야지 생각만 하고 있던 것들을 잡는다)
나같은 경우는 [JAX reference documentation — JAX documentation](https://jax.readthedocs.io/en/latest/index.html)인데,
좋아보이는데 쓸 기회가 없었음. 시간도 남아서 마침 공부하기로 했음.



2. 공부하고 싶은 걸 가져다 쓰는 프로젝트 중에서 너무 크지 않고 너무 작지도 않은 걸 찾는다.
[https://github.com/google/jaxopt](https://github.com/google/jaxopt)

- 너무 큰 프로젝트는 추상화도 잘 되어 있고 리팩토링 수준도 높아서 나같은 뉴비가 따라해볼 만한 건 남아있지 않음
- 너무 작은 프로젝트는 내가 PR 날렸을때 내 코드를 읽고 좀 까 줄 사람이 없음


3. PR을 읽어보고 따라해볼만한 걸 찾는다
![1]({{ "/assets/images/devway/1.png" | absolute_url }})

왠지 할만해보여서 비슷한 거 해볼 수 있을 거 같았음.

해보기 전까지는...



4. 대가리 깨지며 비슷한 걸 만든다
![1]({{ "/assets/images/devway/2.png" | absolute_url }})

요걸 구현하기로 했는데 일단 머리가 아프고 이해가 안 가고 막 짜증이 나는데

이 와중에 JAX의 대략적인 흐름이 이해가 가기 시작하고...
예시도 실제로 쳐보고...
노트에 이것저것 써보고...

예를 들어서, JAX에서는 pytree 개념이 잘 이해가 안 갔는데
[Pytrees — JAX documentation](https://jax.readthedocs.io/en/latest/pytrees.html)
문서로는 뭔소린지 모르겠었음.

내 머리는 추상적인 상황에서 예시를 만들 정도로 똑똑하진 않은데 (4)를 하다 보면 구체적인 예시가 많이 나옴.
실제로 예시도 좀 만들어보고 생각좀 해 보니깐 약간 pytree에 대해 감이 온 것 같았음.

5. PR을 날린다
![1]({{ "/assets/images/devway/3.png" | absolute_url }})

이때 기분 젤 좋음

6. 좀있으면 나보다 똑똑한 아저씨들이 JAX 가르쳐주러 와서 이거 잘못됬다 이거 고쳐라 해 줄 예정
복습에 숙제까지 일석이조 ㅆㅅㅌㅊ 천재적방법


-------

> 해보면 실제로 도움이 엄청나게 된다는 것을 알 수 있습니다...