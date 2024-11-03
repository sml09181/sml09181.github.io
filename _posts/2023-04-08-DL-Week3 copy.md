---
title: Deep Learning Week4 - Naive Bayes, Linear Algebra
author:
    name: Sujin Kim
date: 2023-04-08 03:55:00 +0800
categories: [MLDL]
tags: [ECC, DL]
pin: false
use_math: true

---
# Review
## 1. 조건부 확률(Conditional Probability)
- The probability of an event will sometimes depend upon other things
- The probability of A, given B <- Suppose event B is true<br>
<div align="center">$P(A|B) = \frac{P(A\cap B)P(A)}{P(B)}$</div>
<br>
> event B is true의 의미가 '사건 B가 발생한다고 가정했을 때'인가?
{: .prompt-warning }

## 2. 독립(Independent)
다음 세 가지 경우 중 하나라도 만족시키면 사건 A, B는 서로 독립이라 할 수 있다.
- $P(A|B) = P(A)$
- $P(B|A) = P(B)$
- $P(A \cap B) = P(A)P(B)$
## 3. Conditional Independent
다음 두 가지 경우 중 하나라도 만족시키면 사건 A, B는 서로 독립이라 할 수 있다.
- $P(A \cap B | C) = P(A|C)P(B|C)$
- $P(A|B, C) = P(A|C)$



# 선형대수학
## Basic Operation
1. 선형대수학
- Vector에 대한 내용을 다루는 학문
- Vector를 변형하고 연산하는 것에 대한 규칙

2. Vector란?
- 기하학적 의미:
- (중요) 보다 일반적인 의미: 서로 더할 수 있고, scalar와 곱하여 새로운 것을 만들 수 있는 특별한 object
- 예시
    * Polynomial도 마찬가지로 Vector라고 볼 수 있음
    * $R^n$(n차원 공간의 실수)의 원소도 Vector라고 볼 수 있음


13p
mxkxn

AB: mnp + mpq
A(BC): npq +mnq -> 결과는 똑같지만 연산량이 다르다. softmax에 행렬의 곱이 사용됨.

16p
증명에서 1) 존재성 2) unique

19p
off-diagonal??



Reference:
- [dependence와 correlation의 차이](https://iamtaehoon.wordpress.com/2015/01/13/dependence%EC%99%80-correlation%EC%9D%98-%EC%B0%A8%EC%9D%B4/)
- [기저의 의미](https://losskatsu.github.io/linear-algebra/basis/)
- [랭크, 차원의 의미](https://losskatsu.github.io/linear-algebra/rank-dim/#%EB%B6%80%EB%B6%84%EA%B3%B5%EA%B0%84subspaces)
- [고유값과 고유벡터](https://darkpgmr.tistory.com/105)