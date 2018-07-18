---
layout: article
title: "추천 시스템에서 다양성이 중요한 이유"
category: "productivity"
tag: "productivity"
comment: true
key: prod2
mathjax: true
---

현재 인터넷 어디에서나 추천 시스템을 찾아볼 수 있다.
인터넷에서 흔히 찾아볼 수 있는 추천 시스템의 예시는 다음과 같다.

1. 페이스북의 피드
2. 유투브의 동영상/음악 추천
3. 네이버/다음같은 포털 사이트의 뉴스/블로그 글 추천
4. 구글의 검색 엔진

추천 시스템을 다양하게 정의할 수 있지만, 제가 가장 좋아하는 추천 엔진의 정의는 다음과 같습니다.

> 추천 시스템은 검색 엔진의 검색 결과를 의도적으로 제한하는 방법이다.

이 정의는 추천 시스템이 존재하는 가장 큰 이유를 아름답게 요약해준다고 생각합니다.
예를 들어, 시간을 때울 만한 영화를 보고 싶은 경우를 생각해 봅시다.
당신은 영화 대여점에 들어갔습니다.
영화 대여점 책장에 놓인 수많은 영화의 타이틀을 보면서, 무슨 책을 사야할지 고민해야 본 경험은 다들 있지 않으신가요?
혹은 인터넷에서 무언가를 사고자 할 때, 수많은 물건을 확인하느라, 심지어 내가 어떤 걸 맘에 들어했는지도 까먹어버린 경험도 다들 있을 것 같습니다. 이는 Paradox of Choice라고 하는 현상입니다.
추천 시스템은 이 문제를 해결해주는 훌륭한 방법 중 하나입니다.
[The Paradox of Choice 대한 훌륭한 포스트](https://blog.todoist.com/ko/2015/07/13/science-analysis-paralysis-overthinking-kills-productivity-can/)

추천 시스템이 **유저가 좋아할 만한 아이템을 잘 선택할 수 있다면**, 위의 문제를 해결하는 가장 좋은 방법이 될 수 있을 것이라 생각합니다. 다만, 이제 중요한 얘기는, 추천 시스템이 어떻게 좋은 아이템을 선택하는가에 관한 문제입니다.

유저가 좋아하는 것을 맞추는 방법엔 여러 가지가 있지만, 역시, 가장 쉬운 방법은  유명한 아이템을 추천해주는 것이 아닐까요.


유명한 아이템을 유명하게 만드는 이유는, 그 아이템이 정말로 훌륭해서, 많은 사람들이 그 아이템을 좋아하기 때문이니까요.
![Long-tail]({{ "/assets/images/diversity_in_recsys/long-tail.png" | absolute_url }})
이 그래프가 마케팅, 추천 시스템 업계에서 많이 사용되지만, 원래 의미는 다음과 같다고 생각합니다. long-tail에도 많은 기회가 있지만, "결국 가장 많은 사람의 관심을 받으며, 가장 많이 팔리는 것은 유명한 아이템들이다."

유명한 아이템을 추천하는 경우 추천 시스템의 존재 가치가 조금 퇴색되는 경향이 있는 것 같아요. 유저가 이미 알고 있는 아이템을 추천하는게 의미가 있을까요? 유저가 이미 알고있는 아이템을 추천하는 것은 사실 더 큰 문제점도 갖고 있습니다.

추천 시스템은 유저의 행동 자체에 영향을 미칩니다.

> (Recommender Systems) mediate more and more of what we do. They Guide an increasing proportion of our choices - where to eat, where to sleep, who to sleep with and what to read. From Google to Yelp to Facebook, they help shape what we know.
>
> 인터넷은 점점 우리의 선택을 점점 더 많이 결정한다. 어디서 무엇을 먹는가, 어디에서 자는가, 누구랑 잠을 같이 자는가, 혹은 무엇을 읽는가까지. 구글, 옐프(식당 추천 서비스), 페이스북까지, 인터넷은 당신의 지식을 결정한다.
>
> *Eli Pariser* The Filter Bubble의 저자, TED 강연자.

현재 추천 시스템의 평가 방법은, "유저가 본 아이템을 얼마나 잘 맞추는가?"(Precision, Recall, MAE, Hold-out, k-fold cross validations...와 같은 대부분의 추천 시스템 평가 방식이 이러합니다.)에 맞추어져 있고, 유명한 아이템을 얼마나 추천하는가에 관한 문제는 전혀 고려하지 않고 있습니다.

추천 모델은, 특정 아이템(아마 유명한)이 얼마나 자주 추천되는가에 대해서는 고려하지 않고, 심지어 추천 모델을 만드는 사람도 이에 대해 무지한 경우도 많습니다. 정확도에만 집중하는 추천 시스템이 가져오는 문제는 다음과 같습니다.

Accuracy에만 집중한 추천 시스템은
1. 유저가 좋아할 만한 아이템을 골라 몇 개를 추천한다.
  - 유명한 아이템이 선택될 가능성이 높다.
    - 유명한 것 자체는 아이템의 Quality의 indicator이므로, 좋은 아이템일 가능성은 높지만.
2. User의 view history에 유명한 아이템이 더 추가된다.
3. 추천 시스템은 기본적으로 **유저의 view history** 를 기반으로 아이템을 추천한다.
4. 결국 추천 시스템은 다시 유명한 아이템을 추천하게 된다.
이 효과는 Rich-Get-Richer Effect라는 이름으로 알려져 있으며, 기업의 수익과, 유저의 만족도 양쪽 측면에서 악영향을 미친다.

![Rich-Get-Richer Effect]({{ "/assets/images/diversity_in_recsys/Rich-Get-Richer.png" | absolute_url }})

위 표는 어떤 추천 시스템을 사용하는 인터넷 쇼핑 몰의 판매량 데이터이다.
추천 시스템을 적용한 후, 유명한 아이템이 더 추천이 많이 되고, 유명한 아이템의 판매량이 증가한다. 하지만, 유명하지 않은 아이템의 경우 추천 시스템이 적용된 이후 판매량이 감소한다.

이 결과를 받아들인다면, 추천 시스템이 해야할 일은 다음과 같아진다.

1. 추천 시스템은 유저가 좋아할 만한(Relevant to request)한 아이템을 추천해야 한다.(추천 시스템의 전제)
2. 동시에 추천 시스템은 다양한 아이템을 추천해주어야만 한다.


-Fleder, Daniel, and Kartik Hosanagar. "Blockbuster culture's next rise or fall: The impact of recommender systems on sales diversity."

#### 추천 시스템에서 사용할 수 있는 Diversity Measures.
위 포스트는 음...
