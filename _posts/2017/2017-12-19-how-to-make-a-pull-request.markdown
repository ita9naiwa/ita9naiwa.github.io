---
layout: "post"
title: "How to make a pull request"
date: "2017-12-19 04:29"
---
### How to Make a Pull Request

[TOC]

이 글은 [링크](https://www.codenewbie.org/blogs/how-to-make-a-pull-request)를 번역한 글이다. [grpc](https://grpc.io/)를 사용하던 도중, 타이포를 발견하고 살면서 처음...?으로 pull request를 해봤는데, 인터넷에 찾아 보니까 비슷한 경우가 꽤 많은 것 같다. 다들 이렇게 시작하는 거라구...



처음 `pull request`를 만드는 건 재밌으면서도 스트레스가 쌓이는 일이다. "나같은 초보가 이런 프로젝트에 기여할 수 있는 걸까?", "어디서 시작해야 좋을까?', 혹은 "pull request가 뭔데?"라는 생각을 할 수도 있겠다. 이 페이지에서는 [The Odin Project](https://www.theodinproject.com/)를 이용한 실제 시나리오를 가져와, 차근차근  오픈소스 프로젝트에 간단한 contribution(pull request)를 하는 방법을 소개한다. 오타를 고치는, 어느 개발자라도 할 수 있는 간단한 일이다!



#### 요구사항

이 튜토리얼은 읽는 사람이 깃을 설치하고, 간단한 깃 사용법을 안다고 가정한다.

- Git - http://git-scm.com/book/en/v2/Getting-Started-Installing-Git

- Github - https://help.github.com/articles/set-up-git/


#### Step 1 : Finding the Typo

오픈소스 프로젝트에 기여(Contribute)하는 정말 많은 방법이 있다. 예를 들어 한 프레임워크에 새로운 기능을 추가한다던가, JAVA로 짠 소스 코드를 Scala로 포팅한다던가... 그리고, README file이나 documentation에 있는 오타를 고치는 일 또한 프로젝트에 기여하는 방법 중 하나이고, `모든 레벨의 프로그래머`가 할 수 있다!



페이지 내의 문서를 읽다가 이런 오타를 발견했다. 으음... 이건... 흠...
**... HTML for presentation, CSS for markup, and ...**
실제로는 HTML이 markup language이고, CSS가 presentation format이라는 사실을 아는 당신은 이를 고치고 싶다!


![](https://lh3.googleusercontent.com/nAFKsX3-LQY0m7cQxXBSBYySFqY43PRetVlQRt7tx6xajMoThmUT016tqX61S2uNMUI2Uksl3hA9w39r-YodORbElJ9FepL4oRPJUM1LcjnzWr6B0X_6Wx3In7VGpOZE9Q)


#### Step 2 : Locate the Repository on Github

코드를 변경하기 위해서는 우선 Github reopsitory(repo)에 저장되어 있는 프로젝트 code를 가져와야 한다. repo는 프로젝트 코드가 담겨 있는 장소를 말한다.

![](https://lh6.googleusercontent.com/qnMY3cWPW-wfXVIH3WggmoAFBd0cxS0LZvoGaUdD1TQS6xA7mXWkSFbHTDKlnxgcEDZp_M9H7oZp6nBsjz6sgUFqpzeGAmlF7gcNZeVKzt3BWfuCMpXbgOoWtes8_1KOpA)

여기서, 코드가 담겨져 있는 repo를 확인할 수 있다...


#### Step 3: It's time to fork
Fork(포크)란, 한 repo의 복사본을 만드는 일을 말한다. github 계정이 있다면, 자신의 repositories에 이런 식으로, 오픈소스 프로젝트를 가져올 수 있다.


![ ](https://lh6.googleusercontent.com/j2W8peTpzluqo04YvWdjNDEN1nFrJwmBzoWX2J5TrRcEpBsZbBE4t5oc2AdiZO5chV-DEtlN8IwzvEPF6VnXCpjZShkMMDM2lKn7A6UsJiSFhk9H7sWqSGkb94zyX_raKg)
fork 버튼을 누르면, 내 repository에 project가 저장된다.

![](https://lh6.googleusercontent.com/4IFJTZurVMRn438tf-fQgIaHc3I-viIqH4r5GHhImcgTgiaXUCYmTBwVHUV2joKT4IrpJY1R-kyb-eOPCIEhuozG3mFzjyhqjiRQ65uiLj_gZg16z-7aLDjY3dPC8WQ2yA)

#### Step 4 : Attack of the Clones!
이제 우리는 odin project의 사본을 개인 저장소에 들고 있다. 사실 아직 한 단계가 더 남아 있다. 이 코드를 고치기 위해서는, 이 프로젝트를 내 컴퓨터(Local)로 가져와야 한다.


```
git clone "https:github.com/<your github account>/curriculum.git
```

####Step 5: Now we get to fix that typo? `Bout time!

코드는 적당히 알아서 고치는 것으로 하고...

```
git add <changed files>
#git add --all  //for all files
#git add readme.md // readme file만 고친 경우
git push -m "fix typo"
git commit
```

이렇게 고친 코드를 개인 repository에 업로드할 수 있다. 이제 pull request를 원래 project repository에 남기면 된다!

![](https://lh3.googleusercontent.com/0TMtTC6huKFYQ_NX2qN-OucnaVcrB-W3lLY1cnR6KiUG738l73ik4XnXA4IJrXycPYi_fx1MJt42x21_WcalfUXtBxnluy9mprplKIv5IkIYtRS_IYLrjFYD3E_1QSv6eQ)



New pull Request를 누르고, `compare across forks` 버튼을 또 누른 뒤 내 local repo를 선택한다. (곧바로 만든 repo이므로, branch는 아마 master밖에 없을 것이다)


![](https://lh4.googleusercontent.com/-3vDtJfw-ab8ngqKxBxnFEDkFqVUw_i7uElwRFh1sBbg4BpENZy7PoIq1m8rfNZYyfMZ_WjgZfHfsr8pTvK6gjWFSGzmXfe_wW1MgN-FJvOput99KaXpk2DMoVwAWp7jMw)


request를 할때 간단한 message?.. 음... 어떠한 것을 고쳤는지 이유와 바뀐 코드에 대해 설명을 적는 것이 좋다. 깃허브 내의 대부분 프로젝트에서, pull request에 대한 답장이 오기까지는 보통 1 - 3일의 시간이 걸린다. 적당히 잘 기다리고 나면,




![](https://lh3.googleusercontent.com/_bGiqz_3leW3Iau381UljsDrizplDvAkiUAcStIrkgL79aqUn5-N_FRiyoZqnise1yluf8SQ0dctlG4PtvrC66Qq-u-SsTqI5_78hgPt5IGp4d5IAuGGe7gjUZuBhdW2WA)

과 같은 알림을 받을 것이다. `Merged`의 의미는 네 pull request가 project와 합쳐졌다(반영이 되었다)는 뜻이다.

ㅊㅋㅊㅋㅊㅋㅊㅋ!! 첫번째 pull request를 성공적으로 마쳤다 >_<

####  결론

오픈소스에 기여하는 것은 개발자에게 큰 의미를 지니지만, 꼭 그게 큰 개선이 아니어도 좋고, 겁을 먹지 않아도 좋다~~(고 한다.. 여전히 무서운데...)~~. 오타를 고치는 일이 새로운 기능의 추가는 아니지만, 혼란을 줄이고, 다른 사람에게 도움을 준다는 점에서 기여를 하는 부분이 있다고 생각한다. 우리가 매일 사용하는 많은 프레임워크, 라이브러리, 그리고 많은 도구들은 오픈소스이다. 받은 만큼, 돌려줘보는 건 어떨까...
