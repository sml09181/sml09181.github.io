---
title: Deep Learning Specialization 1-5 | Basics of Neural Network Programming
author: Su
date: 2023-09-29 05:55:00 +0800
categories: [DL]
tags: [EURON]
pin: false
use_math: true

---

Learning Source
+ [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning?utm_source=deeplearningai&utm_medium=institutions&utm_campaign=SocialYoutubeDLSC1W1L1#courses)
+ [부스트코스 딥러닝 1단계: 신경망과 딥러닝](https://m.boostcourse.org/ai215/lectures/86249)

<br>

# **심층 신경망 네트워크**

## **Deep L-layer Nerual network**
<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/c9e5e1b6-6678-4edb-84dd-c7b67da34caa">

+ 참고) 이때 hidden layer의 개수는 하나의 hyper parameter가 된다<br>

<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/7d12587a-b84b-4465-9fde-33dc2ee52fb8">

+ 표기법
  + $L$ : 네트워크 층의 수
  + $n^{​[l]}$: l층에 있는 유닛 개수
  + $a^{​[l]}$: l층에서의 활성값
  + $a^{​[0]}$: 입력 특징($X$)
  + $a^{​[L]}$: 예측된 출력값($\hat y$)
+ 첫 번째 정방향 함수에 대한 input
   + $a^{[0]}$: 한 번에 하나씩(one example) 할 경우의 학습 데이터에 대한 입력 특성
   + $A^{[0]}$: 전체 학습 세트를 진행할 때의 입력 특성


## **Forward Propagation in a Deep Network**
<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/4002941c-6c38-4a98-b5f7-797ef47e48d2">

+ $z$: 해당 층의 parameter에 이전 층의 활성화를 곱하고 편향 벡터를 더해준 값이다.
+ $X$: 서로 다른 열에 저장된 training samples
+ 정방향 전파에서 층 1, 2, 3, 4 각각에 대한 활성화를 계산하는 명시적 반복문을 사용하는 건 괜찮다.
  + 여기서는 반복문을 사용하는 것 외에는 다른 마땅한 방법이 없다.

## **Getting your matrix dimensions right**
<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/aa6808fa-6e82-4207-953f-e16b3c9c1110">

+ one training sample
  + 정방향 전파
    + $W^{​[l]}$ 의 차원: (해당 층의 차원, 그 전층의 차원) ➡️ $(n^{[l]}, n^{[l-1]})$
    + $b^{​[l]}$ 의 차원: (해당 층의 차원, 1) ➡️ $(n^{[l]}, 1)$
  + 역전파
    + $dW^{​[l]}$ 의 차원 ==  $W^{​[l]}$ 의 차원 ➡️ $(n^{[l]}, n^{[l-1]})$
    + $db^{​[l]}$ 의 차원 == $b^{​[l]}$ 의 차원 ➡️ $(n^{[l]}, 1)$
  + $z$ 와 $a$ 의 차원은 같아야 한다.<br>

<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/de034259-f0d4-4a83-a80b-bb8d270cb5bf">

+ Vectorization
  + $W$ 와 $b$ 는 그대로이지만, $Z, A, X$ 의 차원은 조금 달라진다.
  + $Z^{​[l]}$ : $(n^{[l]}, m)$
  + $W^{​[l]}$(그대로) : $(n^{[l]}, n^{[l-1]})$
  + $b^{​[l]}$ : $(n^{[l]}, m)$
    + 원래는 $(n^{[l]}, 1)$ 이지만 broadcasting을 통해 $(n^{[l]}, m)$ 가 된다.
## **Why deep representations?**
<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/9036bb68-1a8a-4433-b928-c50fd520e562">

+ 1️⃣ 직관 1: 네트워크가 더 깊어 질수록, 더 많은 특징을 잡아낼 수가 있다. 낮은 층에서는 간단한 특징을 찾아내고, 깊은 층에서는 탐지된 간단한 것들을 함께 모아 복잡한 특징을 찾아낼 수 있다.
  + 신경망의 첫 번째 층에서 하는 일: 사진을 보고 모서리가 어디에 있는지 파악<br>

<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/95169ca1-9e70-4276-a975-6012c136c8a1">

+ 2️⃣ 직관 2: Circuit theory에 따르면, 상대적으로 은닉층의 개수가 작지만 깊은 심층 신경망에서 계산할 수 있는 함수가 있다. 그러나 얕은 네트워크로 같은 함수를 계산하려고 하면, 즉 충분한 은닉층이 없다면 기하급수적으로 많은 은닉 유닛이 계산에 필요하게 될 것이다. 즉, 얕은 네트워크보다 깊은 네트워크에서 더 계산하기 쉬운 수학적인 함수가 있다.
  + <code>Circuit theory</code>(순환 이론): 로직 게이트의 서로 다른 게이트에서 어떤 종류의 함수를 계산할 수 있을지에 관한 것
  + $\hat y$:$y$ 와 같고, 모든 입력 bit의 배타적 논리합 또는 패리티이다.
  + network의 깊이: $O(\log n)$
## **Building blocks of deep neural networks**
<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/20c79fd4-edb7-4123-b589-312351b47c84">

+ $l$ 번째 층에서 정방향 전파
  + input: 이전 층의 활성화 값인 $a^{​[l-1]}$ 
  + output: $a^{​[l]}$
    + 이때 선형결합된 값인 $z^{​[l]}$ 와 변수 $W^{​[l]}$,$b^{​[l]}$ 값도 캐시로 저장해둔다.
+ $l$ 번째 층에서 역방향 전파
  + input: $da^{​[l]}$ 
  + output: $da^{​[l-1]}$
    + 이때 업데이트를 위한 $dW^{​[l]}$,$db^{​[l]}$ 도 함께 출력
    + 이들을 계산하기 위해서 정방향 계산 때 저장해두었던 캐시를 쓰게 된다.
  + $da^{​[0]}$: 입력 특성에 대응하는 도함수이므로 역전파 단계에서 구하지 않는다
    + 최소한 지도 신경망의 가중치를 학습하는 것에는 유용하지 않다
<br>
<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/f6c4fdf3-42ee-4bb4-a8d2-47effa413823">


## **Forward and backward propagation**
<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/4af0adc3-dfcf-46a4-9fe4-17fec4aeee2b">

* $dz^{​[l]} = da^{​[l]} * {g \prime}^{​[l]} (z^{​[l]})$ 에서 `*`: element-wise product
+ $a^{[l-1]}$ 은 명시적으로 캐시에 저장하지 않았지만 이것 또한 저장해줘야 한다.
+ 역전파는 역함수를 구현하는 과정이라 보면 된다.
<br>
<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/51262aee-b1d4-4aa3-9812-83382de460a9">

+ 역전파를 계산하는 과정에서 캐시에서 $z^{[1]}, z^{[2]}, z^{[3]}$ 을 옮긴다
+ 정방향 반복은 입력 데이터 $X$ 로 초기화한다
+ 역방향 반복은 $da^{[l]}$ 으로 초기화한다. 
  + $-\frac{y}{a} + \frac{(1-y)}{(1-a)}$: 로지스틱 회귀에서 바이너리 분류를 할 때의 값이자 출력, 즉 $y$ 의 예측값에 대응되는 손실 함수의 도함수이기도 하다.

## **Parameters vs Hyperparameters**
<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/87aa426b-6138-4b75-aa5b-ddfb606a942f">

+ <code>parameter</code>(변수): 신경망에서 학습 가능한 $W$ 와 $b$ 를 뜻한다.
+ <code>hyperparameter</code>: 매개변수인 hyperparameter를 결정함으로서 최종 모델의 변수( $W$ 와 $b$)를 통제할 수 있다
  + 학습률(learning rate,  $\alpha$ )
  + 반복횟수(numbers of iteration)
  + 은닉층의 갯수(numbers of hidden layer, $L$)
  + 은닉유닛의 갯수(numbers of hidden units)
  + 활성화 함수의 선택(choice of activation function)
  + 모멘텀항(momentum term)
  + 미니배치 크기(mini batch size)
<br>
<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/b9087c46-55bb-475c-9524-41a97749542d">

+ hyperparameter는 결정된 것이 없으며, 여러 번의 시도를 통해 적합한 hyperparameter를 찾아야 한다.
+ 최적의 hyperparameter를 찾았다고 해도, 일정 시간 뒤에는 다시 업데이트해야 한다.


## **What does this have to do with the brain?**
<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/38a13934-92ec-43a9-aaca-9b15f2cca94d">

+ 신경망과 인간의 뇌 간의 관계는 크지 않다. 다만, 신경망의 복잡한 과정을 단순화해서 뇌 세포의 프로세스로 비유하게 되면, 사람들에게 조금 더 직관적이고, 효과적으로 전달할 수 있다.
+ 그러나 오늘날 신경 과학자들조차도 하나의 뉴런이 무엇을 하는지 거의 모른다. 우리가 신경과학에서 특징짓는 것보다 하나의 뉴런은 훨씬 더 복잡하고 알기 어렵다. 게다가 뉴런이 신경망처럼 역전파를 통해서 학습하는지도 의문이다. 최근에는 이런 비유가 점점 무너져 가고 있다. ➡️더 이상 사용X

## **퀴즈**
+ layer_dims = [n_x, 4, 3, 2, 1]인 레이어들의 배열이 있을 때(layer 1은 4개의 hidden unit이 있고, layer 2는 3개의 hidden unit이 있고 등등...), 이 배열에 𝑛^[𝑙] 의 값을 저장한다고 가정해봅시다. 다음 중 모델의 파라미터를 초기화하는 for-loop는 무엇일까요?
```python
  for (i in range(1, len(layer_dims))):
    parameter['W' + str(i)] = np.random.randn(layers[i], layers[i-1]) * 0.01
    parameter['b' + str(i)] = np.random.randn(layers[i], 1) * 0.01
```