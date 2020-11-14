---
layout: article
title: "[WIP] Doubly Robust Estimator for Ranking Metrics with Post-Click Conversions."
category: "recommender systems"
tag: "recommender systems"
mathjax: true
---



### TL;DR;

I rather use English for blogging from now on because I found that writing in Korean is not very helpful as a portfolio, especially when one's aiming for a job abroad. Also, I have no regular opportunity to use English at all. I need one.



There exist geniuses in the world. And I am not. The fact that I didn't realize this tells me I'm not a genius.

A similar flow of thoughts goes a few hours in my mind, but I stop thinking about this because it is not helpful at all. There will, still, be rooms for not-a-genius-at-all like me in the world of academy or industry. At least I hope so.



### Evaluation of Recommender Systems
Recommender Systems engineers resort to Offline evaluation for evaluating recommender systems since online evaluation costs too much. Until recently, engineers used ranking-based metrics, such as Recall and Mean Average Precision (MAP), to evaluate recommender systems. There is a good [reference on ranking metrics](https://medium.com/swlh/rank-aware-recsys-evaluation-metrics-5191bba16832).

Some concerns on evaluation with ranking metrics exist.
Some believe that these metrics are not very closely related to click-through-rate or conversion.
Implicit feedback is biased.

Anywho, now researchers and engineers (1) by counterfactual estimation of CTR/CVR directly in offline-manner and it's active [research area](https://github.com/st-tech/zr-obp).

Counterfactual evaluation estimates values such as CTR as if it happened (but does not happen actually) by weighting events differently(it gives some weights on events that did not happen).

This approach rocks and is [being exploited in many companies such as Spotify and Netflix](https://sites.google.com/view/reveal2019/)

However, I think (1)  is always the solution to every evaluation problem at all. Sometimes we need ranking-metric based evaluations. I don't know why ranking-metric based evaluation is required at all if counterfactual evaluation works very well. Yesterday I ran into a good paper, [doubly robust est for ranking metrics with post-click conversions](https://dl.acm.org/doi/abs/10.1145/3383313.3412262) and I came into a clue why ranking-metric based evalaution is still required.

### Problem Description
It assumes there are two types of implicit feedback.

- Click
- Conversion

For Youtube as an example, I may click a video clip among some exposured videos. After watching several clips, I save a video to my playlist. Conversion can be any desired and predefined behavior, in this case `saving video`. We want to evaluate recommenders based on its conversion rate.

So, we define our ranking based objective as belows
$$
	R_\text{GT}(\hat{Z}) = \frac{1}{|D|} \sum_{u, i \in D} p^{\text{CVR}}_{u,i} \text{score}(u, i)
$$ where $\text{score}(u, i)$ is any score function, e.g. Recall.

However, there is a huge bias called selection bias. User can only converse if they clicked it. Thus we do not know
