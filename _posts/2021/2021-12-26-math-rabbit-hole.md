---
layout: article
title: "친숙함에 관해, 혹은 수학 우물을 피하는 방법에 대해"
titles:
  ko: "친숙함에 관해, 혹은 수학 우물을 피하는 방법에 대해"
  ja: "親しみやすさについて、あるいは数学の井戸を避ける方法について"
  zh: "关于熟悉感，或者如何避免掉进数学之井"
  en: "On Familiarity, or How to Avoid the Math Well"
languages: [ko, ja, zh, en]
default_lang: ko
category: "일기"
tag: "일기"
header-includes:
  - \usepackage[fleqn]{amsmath}
mathjax: true
---

<section class="post-translation" data-lang="ko" data-default-language markdown="1">

### 잡담
[soft question - On "familiarity" (or How to avoid "going down the Math Rabbit Hole"?) - Mathematics Stack Exchange](https://math.stackexchange.com/questions/617625/on-familiarity-or-how-to-avoid-going-down-the-math-rabbit-hole?noredirect=1&lq=1)

혼자 독학하다가 첨에 엄청 쩔쩔매다 공부 방법을 바꿨었는데, 뭔가 엄청 공감가는 글이라 보고 기억해두기 위해 번역한 뒤 올림.
저 스레드를 다 번역할 수는 없어서, 그냥 내가 생각하기에 중요한 부분만 가져왔음.


누구든 수학을 혼자 공부해본 사람은, 수학 우물에 빠져본 적이 있다.

예를 들어, 새로운 단어인 `벡터 공간`이란 단어를 마주쳤고, 이를 배우고 있다고 생각해 보자. 다양한 정의를 찾아보고선, 이 모든 정의가 `체`라는 단어를 사용하고 있다. 이번엔 `체`의 정의를 찾아 나선다. 같은 상황이 반복된다. `군`는 또, `군`이라는 단어를 사용해 정의된다. 이번엔 또 `group`이라는 단어의 정의를 찾아야 한다. 이런 걸 나는 `수학 우물을 파고 들어간다.`라고 부른다.

이런 상황을 처음 마주한 사람은 "뭐 `vector space`를 배우려면, 나 같으면 그렇게 할 거야"라고 생각할 수도 있다. 그냥, 이건 예시 중 하나일 뿐이다. 이런 행동은, 사실 개인이 깐깐하고 어려운 길을 택하는 것이 아니라, 전적으로 잘못된 길을 걷고 있다는 예시라고 나는 생각한다.

-----

가끔은, (특히 처음 그 주제를 공부할 때면) 정말 정확한 정의를 몰라도 괜찮다고 생각한다. 어느 정도는 (잘못되지 않았다면) 모호한 정의로 출발해도 좋다고 생각한다. 이는 수학의 발전해온 방향과도 일치한다. 미분이 발명된 이후에, 수열과 극한과 대한 엄밀한 정의가 이뤄졌지만, 현재의 미분의 정의는 엄밀한 수열과 극한의 정의에 의존한다.

물론, 그 주제를 더 깊게 공부할수록 더 깊은 이해는 필수적이다. "왜 이 조건이 필요한지?", "이게 실제로 의미하는 바는 무엇인지?" 처음에 이 모든 것을 알기를 기대하진 말자. 특히, 특히, `벡터 공간`이 아니라, 더 추상적이고 어려워서, 수학 우물에 빠지기 더 쉬울 때...


---

`벡터 공간`는 이런 정의를 암기하는 것으로 배우는 것은 아니다.

> 벡터 공간은 집합 V와 체 S로 이뤄져 있으며, 다음과 같은 성질을 만족한다...
> ... (성질들의 목록)

적어도 나는 이렇게 배우지 않았다. 들리는 바에 의하면, 다른 사람들도 이렇게 배우지 않았다고 한다. 벡터 공간에 대한 엄밀한 정의는 벡터 공간이 무엇인지 아는 사람을 위해 존재한다. 다시 말해서, 엄밀한 정의는 아마도 이미 아는 것을 상기시켜주기 위해 존재할지도 모른다.

그 대신에, 벡터 공간이 무엇인지 배우고 싶다면, 선형대수학에 대한 간단한 책을 사서 읽기 시작하는 것이 좋다. _Linear Algebra and Its Applications (Strang, 1988)_ 을 방금 집어서 침대에 가져왔는데, 책의 초반부에는 벡터 공간은 정의되지도 않는다. 2장에서, 벡터 공간에 대한 아이디어가 엄밀하지 (1장에서 이미 소개된 $\mathbb{R}^n$에 집중하며) 않은 방식으로 소개되고, 그리고 벡터 공간의 중요한 성질을 강조한다. "두 벡터를 더한 것도 벡터이며, 벡터를 어떤 스칼라로 곱해준 것도 벡터이다."라는 말과, 여러 예시를 보여준다.

선형대수를 공부할 때 사실 체가 뭔지 알 필요는 없다. 그냥, 선형대수를 배우는 것이다.
을
나중에 임의의 체를 공부할 때가 올 때, 이렇게 생각하는 편이 더 좋다고 생각한다. "벡터 공간이랑 비슷한데, 숫자가 아니라 그냥 임의의 원소가 들어갈 수 있는 경우."

만약 당신이 끝나지 않는 수학 우물을 파고 있다면, 해보는 것도 가치가 있을지도 모른다. 라마누잔은 그렇게 수학을 배웠다고 한다. 하지만, 우리가 라마누잔 같은 천재가 아니라면, 라마누잔이 아닌 다른 사람들이 하는 방식으로 수학을 배우는 것이 좋다고 생각한다.

### 걍 내생각
수학뿐만 아니라 컴퓨터 공학에서도 이 규칙이 적용되는 것 같다.
그리고, 서두를 수록 수학 우물에 빠지기 쉬운 것 같다. 빠른 길은 없고, 그냥 오래 하고 많이 해야 하는 것 같다;;

</section>

<section class="post-translation" data-lang="ja" markdown="1">

### 雑談
[soft question - On "familiarity" (or How to avoid "going down the Math Rabbit Hole"?) - Mathematics Stack Exchange](https://math.stackexchange.com/questions/617625/on-familiarity-or-how-to-avoid-going-down-the-math-rabbit-hole?noredirect=1&lq=1)

独学していて、最初はものすごく苦戦した末に勉強法を変えたことがあったんだけど、ものすごく共感できる文章だったので、読んだことを覚えておくために翻訳して投稿。
あのスレッドを全部翻訳することはできないので、単に自分が重要だと思った部分だけ持ってきた。


数学を一人で勉強したことがある人なら誰でも、数学の井戸に落ちたことがある。

たとえば、`ベクトル空間`という新しい言葉に出会い、それを学んでいるとしよう。いろいろな定義を探してみると、そのすべての定義で`体`という言葉が使われている。今度は`体`の定義を探しにいく。同じ状況が繰り返される。`群`もまた、`群`という言葉を使って定義される。今度はまた`group`という言葉の定義を探さなければならない。こういうことを私は`数学の井戸を掘り進む。`と呼ぶ。

こういう状況に初めて出会った人は、「まあ、`vector space`を学ぶなら、自分でもそうするだろう」と思うかもしれない。ただ、これは例の一つにすぎない。こういう行動は、実は個人が細かいことにこだわって難しい道を選んでいるのではなく、完全に間違った道を歩いている例だと私は思う。

-----

ときには、（特にそのテーマを初めて勉強するときなら）本当に正確な定義を知らなくても大丈夫だと思う。ある程度は、（間違ってさえいなければ）曖昧な定義から出発してもいいと思う。これは数学が発展してきた方向とも一致する。微分が発明された後になって、数列と極限についての厳密な定義がなされたが、現在の微分の定義は厳密な数列と極限の定義に依存している。

もちろん、そのテーマを深く勉強すればするほど、より深い理解は不可欠だ。「なぜこの条件が必要なのか？」「これは実際には何を意味するのか？」最初からこのすべてを知ることを期待するのはやめよう。特に、特に、`ベクトル空間`ではなく、もっと抽象的で難しく、数学の井戸に落ちやすいときには……


---

`ベクトル空間`は、こういう定義を暗記することで学ぶものではない。

> ベクトル空間は集合 V と体 S からなり、次のような性質を満たす……
> ……（性質の一覧）

少なくとも私はこうやって学んだわけではない。聞くところによると、ほかの人たちもこうやって学んだわけではないという。ベクトル空間についての厳密な定義は、ベクトル空間が何かを知っている人のために存在する。言い換えれば、厳密な定義はおそらく、すでに知っていることを思い出させるために存在するのかもしれない。

その代わり、ベクトル空間が何かを学びたいなら、線形代数学についての簡単な本を買って読み始めるのがいい。_Linear Algebra and Its Applications (Strang, 1988)_ をたった今手に取ってベッドに持ってきたが、本の序盤ではベクトル空間は定義すらされない。第2章で、ベクトル空間についてのアイデアが厳密ではない（第1章ですでに紹介された $\mathbb{R}^n$ に焦点を当てる）形で紹介され、そしてベクトル空間の重要な性質が強調される。「二つのベクトルを足したものもベクトルであり、ベクトルに何らかのスカラーを掛けたものもベクトルである」という言葉と、いくつもの例が示される。

線形代数を勉強するとき、実のところ体が何なのかを知る必要はない。ただ、線形代数を学ぶのだ。
を
後になって任意の体を勉強するときが来たら、こう考えるほうがいいと思う。「ベクトル空間に似ているけど、数ではなく単に任意の要素を入れられる場合。」

もしあなたが終わりのない数学の井戸を掘っているなら、やってみる価値もあるかもしれない。ラマヌジャンはそうやって数学を学んだという。だが、私たちがラマヌジャンのような天才でないなら、ラマヌジャンではないほかの人たちがするやり方で数学を学ぶのがいいと思う。

### ただの自分の考え
数学だけでなく、コンピュータ工学でもこのルールが当てはまる気がする。
そして、急げば急ぐほど数学の井戸に落ちやすい気がする。近道はなく、ただ長く続けて、たくさんやるしかない気がする;;

</section>

<section class="post-translation" data-lang="zh" markdown="1">

### 闲聊
[soft question - On "familiarity" (or How to avoid "going down the Math Rabbit Hole"?) - Mathematics Stack Exchange](https://math.stackexchange.com/questions/617625/on-familiarity-or-how-to-avoid-going-down-the-math-rabbit-hole?noredirect=1&lq=1)

一个人自学时，一开始折腾得够呛，后来改了学习方法。看到这篇文章觉得特别有共鸣，所以翻译后发上来，留着提醒自己读过它。
没法把那个帖子全部翻译出来，所以就只摘了些我认为重要的部分。


凡是独自学过数学的人，都曾掉进过数学之井。

比如说，假设你遇到了一个新词`向量空间`，正在学习它。查找各种定义后，发现所有定义都使用了`域`这个词。这次又去找`域`的定义。同样的情况不断重复。`群`又是用`群`这个词来定义的。这次又得去找`group`这个词的定义。我把这种事称为`往数学之井里越挖越深。`

第一次遇到这种情况的人可能会想：“嗯，要学`vector space`的话，换成我也会这么做。”不过，这只是其中一个例子。我认为，这种行为其实并不是个人较真、选择了一条艰难的路，而是一个走在完全错误道路上的例子。

-----

有时候，（尤其是第一次学习那个主题时）即使不知道真正精确的定义也没关系。我觉得在一定程度上，可以从一个模糊的定义出发（只要它不是错的）。这也符合数学发展的方向。微分被发明之后，数列与极限才得到了严密的定义，但如今微分的定义却依赖于数列与极限的严密定义。

当然，对那个主题钻研得越深，越深入的理解就越不可或缺。“为什么需要这个条件？”“这实际上意味着什么？”不要期待自己一开始就知道这一切。尤其是，尤其是，当它不是`向量空间`，而是更加抽象、更加困难，也更容易掉进数学之井的时候……


---

学习`向量空间`不是靠背诵这样的定义。

> 向量空间由集合 V 和域 S 构成，并满足如下性质……
> ……（性质列表）

至少我不是这样学会的。听说其他人也不是这样学会的。关于向量空间的严密定义，是为已经知道向量空间是什么的人而存在的。换句话说，严密定义也许是为了提醒你已经知道的东西而存在的。

相反，如果想学习向量空间是什么，最好买一本简单的线性代数书，然后开始读。我刚刚拿起 _Linear Algebra and Its Applications (Strang, 1988)_ 带到了床上，书的开头甚至没有定义向量空间。在第 2 章中，向量空间的概念以并不严密的方式（集中于第 1 章已经介绍过的 $\mathbb{R}^n$）被引入，并强调了向量空间的重要性质。“两个向量相加得到的仍是向量，将向量乘以某个标量得到的仍是向量。”书中这样说，并展示了多个例子。

学习线性代数时，其实没有必要知道域是什么。就只是学习线性代数。
把
以后到了学习任意域的时候，我觉得最好这样想：“它和向量空间差不多，只不过放进去的不是数字，而是任意元素。”

如果你正在挖一口没有尽头的数学之井，也许试试这么做也值得。据说拉马努金就是这样学习数学的。但是，如果我们不是拉马努金那样的天才，我觉得最好还是用其他那些不是拉马努金的人所用的方式来学习数学。

### 就是我的想法
感觉这条规则不仅适用于数学，也适用于计算机工程。
而且，感觉越着急就越容易掉进数学之井。没有捷径，就只能长期坚持，多做多学;;

</section>

<section class="post-translation" data-lang="en" markdown="1">

### Chitchat
[soft question - On "familiarity" (or How to avoid "going down the Math Rabbit Hole"?) - Mathematics Stack Exchange](https://math.stackexchange.com/questions/617625/on-familiarity-or-how-to-avoid-going-down-the-math-rabbit-hole?noredirect=1&lq=1)

When I was studying on my own, I struggled like crazy at first and then changed how I studied, and this was something I really related to, so I translated it and posted it here to remember reading it.
I couldn't translate that entire thread, so I just brought over the parts I thought were important.


Anyone who has studied math on their own has fallen into the math well before.

For example, suppose you have encountered a new term, `vector space`, and are trying to learn it. You look up various definitions and find that all of them use the word `field`. This time, you go looking for the definition of `field`. The same situation repeats itself. A `group`, in turn, is defined using the word `group`. Now you have to look up the definition of the word `group` again. I call this `digging down into the math well.`

Someone encountering this situation for the first time might think, “Well, if I were learning `vector space`, I would do that too.” This is just one example, though. I think this kind of behavior is actually an example not of someone being meticulous and choosing a difficult path, but of walking down an entirely wrong path.

-----

Sometimes, I think it is okay not to know the truly precise definition (especially when studying the topic for the first time). To some extent, I think it is fine to start with a vague definition (as long as it is not wrong). This is also consistent with the direction in which mathematics developed. Rigorous definitions of sequences and limits came after differentiation was invented, but the modern definition of differentiation depends on rigorous definitions of sequences and limits.

Of course, the more deeply you study the topic, the more essential a deeper understanding becomes. “Why is this condition necessary?” “What does this actually mean?” Let's not expect to know all of this at the beginning. Especially, especially, when it is not a `vector space`, but something more abstract and difficult, making it easier to fall into the math well...


---

You do not learn a `vector space` by memorizing a definition like this.

> A vector space consists of a set V and a field S, and satisfies the following properties...
> ... (a list of properties)

At least, I did not learn it this way. From what I hear, other people did not learn it this way either. The rigorous definition of a vector space exists for people who know what a vector space is. In other words, perhaps the rigorous definition exists to remind you of something you already know.

Instead, if you want to learn what a vector space is, it is a good idea to buy a simple book on linear algebra and start reading. I just picked up _Linear Algebra and Its Applications (Strang, 1988)_ and brought it to bed, and vector spaces are not even defined in the early part of the book. In Chapter 2, the idea of a vector space is introduced in a non-rigorous way (focusing on $\mathbb{R}^n$, already introduced in Chapter 1), and the important properties of vector spaces are emphasized. It says, “The sum of two vectors is also a vector, and multiplying a vector by some scalar also gives a vector,” and shows several examples.

When studying linear algebra, you do not actually need to know what a field is. You just learn linear algebra.
Object marker
Later, when the time comes to study arbitrary fields, I think it is better to think of it this way: “It is like a vector space, except that arbitrary elements can go in instead of numbers.”

If you are digging an endless math well, it might be worth trying. Ramanujan is said to have learned mathematics that way. But if we are not geniuses like Ramanujan, I think it is better to learn mathematics the way other people who are not Ramanujan do.

### Just my thoughts
This rule seems to apply not only to mathematics but also to computer engineering.
And it seems that the more you rush, the easier it is to fall into the math well. There is no quick way; it seems you just have to keep at it for a long time and do a lot;;

</section>
