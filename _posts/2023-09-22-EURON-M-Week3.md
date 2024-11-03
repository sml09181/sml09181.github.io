---
title: Deep Learning Specialization 1-4 | Basics of Neural Network Programming
author: Su
date: 2023-09-22 05:55:00 +0800
categories: [DL]
tags: [EURON]
pin: false
use_math: true

---

Learning Source
+ [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning?utm_source=deeplearningai&utm_medium=institutions&utm_campaign=SocialYoutubeDLSC1W1L1#courses)
+ [부스트코스 딥러닝 1단계: 신경망과 딥러닝](https://m.boostcourse.org/ai215/lectures/86249)

<br>

# **얕은 신경망 네트워크**

## **신경망 네트워크 개요**
+ `Logistic Regression`<br>
  <img width="330" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/e0d500b6-a370-451c-a01a-76075eabe8e0">
  + $a$ 와 $z$ 의 계산이 **한 번씩** 이루어진다<br>
  <img width="418" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/58f4a4cb-1136-4fcb-8222-5076288c7af7">
  + $w$ 와 $b$ 의(맞나?) 도함수 계산을 위해 역전파로 $dz$ 를 업데이트 한다.(da도 추가할 수 있나??) 
+ `Neural Network`<br>
  <img width="367" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/492f7a56-3fa2-4c57-98d9-9ecba5e6c87a">
  + $a$ 와 $z$ 의 계산이 **여러 번** 이루어진다<br>
  <img width="767" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/ad287289-5b9b-49e5-bb90-7ab0ec1e4f70">
  + $da^{[2]}$ 와 $dz^{[2]}$ 를 가지고 $dw^{[2]}$ 와 $db^{[2]}$ 를 계산한다

<br>

## **신경망 네트워크의 구성 알아보기**
<img width="500" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/74e8f58d-00f3-48c3-a1d6-af35441c04bb"><br>

+ <code>Input Layer</code>(입력층): 입력 특성들의 층 ➡️ $a^{[0]}$
  + 입력 세트의 표현법: $a^[0]$ 또는 $X$
  + $a^[0]$: 입력층의 활성값
  + a는 활성값을 의미하고 신경망의 충돌이 다음 층으로 전달해주는 값을 의미
+ <code>Hidden Layer</code>(은닉층): 입력층과 출력층 사이에 있는 모든 층
  + $m$ 번째 은닉층은 위첨자 $[m]$ 으로, 그 층의 $n$ 번째 유닛은 아래첨자에 표기한다
    + ex) 첫 번째 은닉층 ➡️ $a^{[l]}$
    + ex) 첫 번째 은닉층의 n번째 유닛 ➡️ $a_n^{[l]}$ 
    + ex) 첫 번째 은닉층에 있는 두 번째 유닛 ➡️ $a_2^{[1]}$
  + 입력층과 출력층의 값은 알 수 있지만 은닉층의 값은 알 수 없다
  + 은닉층에서는 열벡터를 만든다
+ <code>Output Layer</code>(출력층): 출력 특성들의 층
+ 신경망 층의 개수를 셀 때 Input Layer는 고려하지 않는다.
  + ex) 아래 그림: 은닉층 1개, 출력층 1개 ➡️ 2 Layer NN<br>
        <img src="https://github.com/sml09181/sml09181.github.io/assets/105408672/d92230e1-bf18-478b-844e-ff2e587748fe" width="350">
+ shape(# training sample = 1)
  + $X$ : $(n_x, 1)$ (열벡터)
  + $w^{[p]}$(in 은닉층) : (# of $p$ 번째 은닉층의 node, $n_x$)
  + $b^{[p]}$(in 은닉층) : (# of $p$ 번째 은닉층의 node, 1)
  + $w^{[q]}$(in 출력층, 출력층 node 1개) : (1, # of 마지막 은닉층의 node)
  + $b^{[q]}$(in 출력층, 출력층 node 1개) : (1, 1)
    + 위 그림에서 $p=1, q=2$<br>
  <img src="https://github.com/sml09181/sml09181.github.io/assets/105408672/7e7b96aa-3676-494e-bfe3-556bb0ecc071" width="600">


## **신경망 네트워크 출력의 계산**
### Neural Network Representation

+ 입력값이 노드를 통과할 때, 두 가지 과정을 거친다(노드랑 퍼셉트론 차이?)<br>
  <img width="317" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/6b58ded7-ca2c-4eba-9bfd-8055e8c6373a">
  + 1️⃣ $z = w^T + b$
  + 2️⃣ $a=\sigma(z)$<br>
+ 신경망에서의 노드<br>
  <img width="818" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/27df0d8f-24de-4ad8-b204-6ede247dd817">
  + 표기법
    + $a_i^{[l]}$
    + $l$: 몇 번째 층인지 의미
    + $i$: 해당 층에서 몇 번째 노드인지 의미
+ python `hstack`과 `vstack`<br>
  <img src="https://github.com/sml09181/sml09181.github.io/assets/105408672/2a453221-e32b-4c7f-a815-93bd7ca7148a" width="600">
  + `hstack`: 세로축 방향으로 결합한다, 행으로 쌓는다
  + `vstack`: 가로축 방향으로 결합한다, 열로 쌓는다
+ training sample이 하나만 있을 때 예측값 구하는 과정을 행렬로 살펴보기<br>
  <img width="750" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/66ba55c5-08a3-4d8f-b722-4170c356bca5">

  + 입력 벡터 $X$: 각 feature($x_1, x_2, ...x_{n_x}$)를 쌓아 열벡터가 된다
  + 행렬 $W^{[l]}$: $w_i^{T[l]}$ 을 행으로 쌓는다
    + 각 행이 하나의 노드를 나타내며, $w_i^{[l]}$ 이다.
  + 행렬 $a^{[l]}$: $a_i^{[l]}$ 을 행으로 쌓는다
    + 각 행이 하나의 노드를 나타내며, $a_i^{[l]}$ 이다.(열벡터)<br>
  <img width="829" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/3c4b9caa-c9dc-40f6-9139-38d4c44ddbc4">



## 많은 샘플에 대한 벡터화

<img width="836" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/57780f7d-3c04-4b30-9deb-25465f4112da">

+ 표기법
  + $a^{[i](j)}$
  + $i$: 몇 번째 층인지 의미
  + $j$: 몇 번째 훈련 샘플인지 의미<br>

<img width="848" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/2288c3ce-e21c-4f98-ab74-7ec12deb4868">

+ 훈련 샘플과 관련된 모든 변수에 (i) 붙여주면 된다 ➡️ (w, z, a)
+ $x$ 는 훈련 샘플을 열로 쌓은 것이다
+ 행렬 $z$ 와 $a$ 의 가로는 훈련 샘플의 번호가 된다
  + 세로는 신경망의 노드들(은닉 노드의 번호)
+ 행렬 $x$: 가로는 다른 훈련 샘플, 세로는 다른 입력 특성

## 벡터화 구현에 대한 설명

<img width="706" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/d8bb9787-8e0c-4c2e-9050-08c73b5acbbf">

+ 입력값을 열로 쌓는다면 결과값도 열로 쌓인 결과가 나온다.
+ 대칭성: $w^{[1]}a^{[0]}+ b^{[1]}$<br>
<img width="706" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/5b9e9a57-6e2c-435a-b5bf-0e4858962d62">

## 활성화 함수

<img width="703" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/8c92993d-827b-45af-a488-bdff1045cf50">

+ Sigmoid
  + $a=\frac{1}{1+e^{-z}}$
+ Tanh
  + $a=\frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}$
+ Relu
  + $a=\operatorname{max}(0, z)$
+ Leaky ReLU
  + $a=\operatorname{max}(0.01z, z)$
+ Tanh 장점: Tanh 의 값이 [-1, 1] 사이에 있고 평균이 0이기 때문에, 데이터를 원점으로 이동하는 효과가 있습니다. 이는 평균이 0.5인 Sigmoid 보다 더 효율적. 학습이 더 잘 된다.
+ ReLU 장점: 0보다 큰 활성화 함수의 미분값이 다른 함수에 비해 많아 빠르게 학습할 수 있다
  + 실제로는 학습을 느리게 하는 원인인 함수의 기울기가 0에 가까워지는 걸 막아주기 때문에 실제로는 충분한 은닉 유닛의 Z는 0보다 크기 때문에 실제로는 잘 동작한다<br>
<img src="https://github.com/sml09181/sml09181.github.io/assets/105408672/51c3c18a-bb73-4b21-b24c-f1a7fd79ef9c" width="600">


+ 다만 출력층에서은 -1과 1보다는 0과 1이 더 좋음(이진분류할 때) ➡️ 시그모이드
+ z가 굉장히 크거나 작으면 도함수가 0에 가까워져 경사 하강법이 느려질 수 있다. ➡️ ReLU
+ 이진분류의 출력층에는 시그모이드가, 나머지에는 ReLU가 쓰인다
+ ReLU 단점: Z가 음수일때 도함수가 0 ➡️ Leaky ReLU
+ $g^[1]$ 와 $g^[2]$ 가 다를 수 있음
+ LeakyReLU의 0.01 값도 알고리즘의 변수로 넣을 수 있다



## 왜 비선형 활성화 함수를 써야할까요?

<img width="707" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/11c0c15f-7eb0-493d-9894-b2070fa1e851">

+ $g$ 지우면 $g(z)=z$
+ 선형 함수는 $y=x, y=ax, y=ax+b$ 와 같은 함수를 의미한다
  + 선형 활성화 함수, 입력값을 출력값과 같게 내보내기 때문에 항등함수가 더 정확
  + 신경망은 입력의 선형식만을 출력한다 
+ 예를 들 $g(z)=z$ 라는 선형 활성화 함수를 사용한다고 가정했을 때, 3개의 은닉층을 쌓아도 $g(g(g(z)))=z$ 로는 효과가 없다
+ 따라서 은닉층에는 비선형 활성화 함수를 사용해야 한다
+ 비선형 활성화 함수는 ReLU, Sigmoid, Tanh 등의 함수가 있다
+ 선형 활성화함수는 회귀할 때만 쓴다 y가 실수값이면 선형 활성화 함수 써도 된다
+ 출력값인 yhat이 -무한대 무한대이기 대문(ex: 집값 예측)
+ 선형 활성화 함수를 쓰는 곳은 대부분 출력층이다



## 활성화 함수의 미분

<img width="702" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/4a14c419-e4b5-4c7d-b3a5-0834110e738e">

+ Sigmoid
  + $g(z) = \frac{1}{1+e^{-z}}$
  + $g(z) = \frac{d}{dz}g(z) = g(z)(1-g(z))$<br><br>
<img width="691" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/24fd5aaf-88fc-489a-9f41-b3863bdeafab">

+ Tanh
  + $g(z) = \frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}$
  + $g \prime (z) = 1-(g(z))^2$<br><br>

<img width="700" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/cd114b9c-8048-4e89-bf6a-b73de76d50ea">

+ ReLU
  + $g(z)=\operatorname{max}(0, z)$
  + $g \prime (z)=0$ ($z$<0)
  + $g \prime (z)=1$ ($z$>=0)
+ Leaky ReLU
  + $g(z)=\operatorname{max}(0.01z, z)$
  + $g \prime (z)=0.01$ ($z$<0)
  + $g \prime (z)=1$ ($z$>=0)


+ $g \prime $ 는 활성화 함수 $g(z)$ 의 서브 경사이기 때문에 경사하강법이 잘 작동
+ $z$ 가 정확히 0이 될 확률은 매우 작다


## 신경망 네트워크와 경사 하강법
+ 단일층 신경망에서 경사 하강법을 구현하기 위한 방법
  + $dw^{[1]}=\frac{dJ}{\partial w^{[1]}}$
  + $db^{[1]}=\frac{\partial J}{\partial b^{[1]}}$
  + $W^{[1]} = W^{[1]} - \alpha dW^{[1]}$
  + $b^{[1]} = b^{[1]} - \alpha db^{[1]}$
+ 단일층이 아닐 때는 1뿐만 아니라 $1, 2, …,m$ 까지의 계산을 반복하면 된다


+ 이진 분류를 하고 있다고 가정
+ 0이 아닌 값으로 초기화해야 함
+ `keepdims`: 파이썬이 잘못된 1차원 배열을 출력하지 않게 하는 것: (n,)에서 (n,1)로 만든다
+ 정방향 식 4개
+ 역전파 6개

## 역전파에 대한 이해
+ 로지스틱 회귀의 역전파를 구하면 다음과 같습니다.
  + $da=− \frac{y}{a}y​​+​ \frac{1-y}{1-a}​​ $
  + $ dz=a−y$
  + $ dw=dz x$
  + $ db=dz $
  + $x$ 는 고정값이기에 $dx$ 계산하지 않습니다.

+ 실제로는 da[1]과 dz[1]를 한 단계로 계산할 수 있다
+ 차원이 정확히 일치하는지 항상 확인하자


## 랜덤 초기화
+ 신경망에서 $w$ 의 초기값을 0으로 설정한 후 경사 하강법을 적용할 경우 올바르게 작동하지 않습니다.
  + $dw$ 를 계산했을 때 모든 층이 같은 값을 가지게 되기 때문입니다.
+ 따라서 np.random.rand()를 이용해 0이 아닌 랜덤한 값을 부여해줘야 합니다.

+ $b$ 는 모두 0이 되어도 괜찮다, $w$ 만 모두 0이 아니면 된다
+ $w$이 다 0이면 완전 같은 함수를 계산하는 은닉층이 된다

+ 모든 은닉 유닛이 대칭이됨
+ 다른 함수를 계산하기 위한 각각 다른 유닛이 필요함

+ $b$: 대칭 회피 문제

+ 가중치의 초깃값은 매우 작게 하는 것이 좋다
+ w가 큰값을 가지면 z도 엄청 커지고 시그모이드에서 기울기가 커서 
+ 매우 큰 값의 z를 가지고 훈련을 시작하게 되고
+ tanh나 시그모이드가 너무 큰 값 -> 학습 속도가 느려짐



<br><br>



<br><br>

Source:

Image Source:
+ https://medium.com/data-science-365/overview-of-a-neural-networks-learning-process-61690a502fa
+ https://stackoverflow.com/questions/33356442/when-should-i-use-hstack-vstack-vs-append-vs-concatenate-vs-column-stack