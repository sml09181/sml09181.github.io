---
title: Deep Learning Specialization 2-4 | Basics of Neural Network Programming
author: Su
date: 2023-11-11 05:55:00 +0800
categories: [DL]
tags: [EURON]
pin: false
use_math: true
---

Learning Source
+ [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning?utm_source=deeplearningai&utm_medium=institutions&utm_campaign=SocialYoutubeDLSC1W1L1#courses)
+ [부스트코스 딥러닝 2단계: 심층 신경망 성능 향상시키기](https://m.boostcourse.org/ai216/lectures/132205)

<br>

## **Mini-batch gradient descent**
<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/5eada88c-842e-4f3e-8833-439811324605">

+ 좋은 최적화 알고리즘을 찾는 것은 효율성을 좋게 만든다.
+ 벡터화는 m개의 샘플에 대한 계산을 효율적으로 만들어 준다.
  + 명시적인 반복문 없이도 훈련 세트를 진행할 수 있도록 한다.
  + 하지만 m이 매우 크다면 여전히 느릴 수 있다.
+ <code>Batch Gradient Descent</code>: 전체 훈련 샘플에 대해 훈련 후 경사 하강 진행
  + 경사 하강법을 사용하면 경사 하강법의 작은 한 단계를 밟기 전에 모든 훈련 세트를 처리해야 한다.
  + 배치 경사 하강법은 큰 데이터 세트를 훈련하는데 많은 시간이 들기에 결과적으로 경사 하강을 진행하기까지 오랜 시간이 걸린다.
+ <code>Mini-batch Gradient Descent</code>: 전체 훈련 샘플을 작은 훈련 세트인 mini-batch로 나눈 후, mini-batch 훈련 후 경사 하강 진행
  + 경사 하강법의 각 스텝에서 모든 training sample을 처리하지 않고, 부분 집합(미니 배치)만을 처리하게 되면 더 빨라진다.
  + 따라서 작은 훈련 세트인 mini-batch로 나누어 훈련 후 경사 하강을 진행한다.
  + 하나의 mini-batch $X^{t}, Y^{t}$ 를 동시에 진행시키는 알고리즘 
+ 예를 들어 전체 훈련 세트 크기가 5,000,000이라고 할 때, 이를 사이즈가 1,000인 mini-batch 5,000개로 나누어 훈련 및 경사 하강법을 진행한다.
+ 표기법
  + $(i)$: 몇 번째 trainig sample인지
  + $[l]$ : 신경망의 몇 번째 층인지
  + ${t}$ : 몇 번째 mini-batch인지(=mini-batch의 개수)
+ 배치 경사 하강법에서 전체 훈련 세트를 거치는 한 반복은 오직 하나의 경사 하강 단계만을 할 수 있게 하지만, 미니 배치 경사 하강법의 경우 훈련 세트를 거치는 한 반복은 $t$ 개의 경사 하강 단계를 거치도록 한다.

<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/16f9942c-a082-4693-9b43-f6de1e4c1374">


## **Understanding mini-batch gradient descent**

<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/628dd884-95a4-4515-acb7-46d2a6cbf823">

+ 배치 경사 하강법: 한 번의 반복을 돌 때마다 비용 함수의 값은 계속 감소한다.
+ 미니배치 경사 하강법: 전체적으로 봤을때는 비용 함수가 감소하는 경향을 보이지만 많은 노이즈가 발생한다. 
+ 노이즈(진동)이 발생하는 이유
  + 이전 mini-batch보다 현재 mini-batch가 더 어려운 mini-batch라서
  + 잘못 표시된 샘플이 있다든지의 이유로 비용이 약간 더 높아질 수 있다.<br>

<img width="649" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/6b050284-e399-42e6-b776-42576c8b463b">

+ 미니배치 사이즈를 어떻게 선택하는지에 따라 학습 속도의 차이가 나기에 최적의 값을 찾아내는 것이 중요하다.
+ 비용함수의 등고선
<br>

<img width="545" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/68ff22c1-70c3-42b5-b0bd-d831145953f2">

+ 만약 훈련 세트가 작다면(2,000개 이하) 모든 훈련 세트를 한 번에 학습시키는 배치 경사 하강을 진행합니다.
+ 훈련 세트가 2,000개 보다 클 경우 전형적으로 선택하는 미니배치 사이즈는 64, 128, 256, 512와 같은 2의 제곱수이다.
  + 보통 1024부터는 잘 선택하지 않는다.
  + 또 모든 $X^{t}, Y^{t}$ 가 CPU와 GPU 메모리에 맞는지 확인해야 한다.
+ 배치 경사 하강법은 상대적으로 노이즈가 적고, 큰 단계를 취한다. 그렇게 계속 최솟값으로 나아간다.
+ SGD에서 대부분은 최솟값으로 나아가지만 어떤 경우는 잘못된 곳으로 가기도 한다. 극단적으로 노이즈가 많을 수 있지만 평균적으로는 좋은 방향으로 가게 된다. 따라서 절대 수렴하지 않을 것이다. 진동하면서 최솟값의 주변을 돌아다니게 되지만 최솟값으로 곧장 가서 머물지는 않을 것이다.
  + 노이즈도 작은 학습률로 줄일 수 있다.
  + 그러나 큰 단점은 벡터화에서 얻을 수 있는 속도 향상이라는 이점을 취하지 못한다.
+ mini-batch
  + 학습이 가장 빠르다.
  + 많은 벡터화를 얻는다.
  + 전체 훈련 세트가 진행되기를 기다리지 않고 진행할 수 있다.<br> 



## **Exponentially weighted averages(지수 가중 이동 평균)**
+ 경사 하강법 및 미니배치 경사 하강법보다 더 효율적인 최적화 알고리즘을 이해하기 위해서는, 지수 가중 이동 평균을 먼저 이해해야 한다.
+ 최근의 데이터에 더 많은 영향을 받는 데이터들의 평균 흐름을 계산하기 위해 지수 가중 이동 평균을 구한다. 지수 가중 이동 평균은 최근 데이터 지점에 더 높은 가중치를 준다.

<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/dd80d8b2-0bcc-40e6-97e8-cffbeb7d516c">

+ $\theta_t$ 를 $t$ 번째 날의 기온이라고 했을 때, 지수 가중 이동 평균($v_t$)의 식은 다음과 같습니다.
  + $v_t = \beta v_{t-1} + (1-\beta)\theta_t$
  + 이때 $\beta$: 하이퍼 파라미터 ➡️ 보통 0.9 사용<br>

<img width="625" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/00625af5-c3be-4b93-a89e-048fb453c82b">

+ $v_t$ 는 대략적으로 $\frac{1}{1-\beta}$ 곱하기 일별 기온의 평균과 같다.
  + 즉 $\frac{1}{1-\beta}$ 기간 동안 기온의 평균을 의미힌디.
  + $\beta=0.9$ 일 때 ➡️ 10일의 기온 평균
  + $\beta=0.5$ 일 때 ➡️ 2일의 기온 평균
    + 노이즈가 크고 이상치에 더 민감하게 반응하지만, 기온 변화에 더 빠르게 적응한다.
  + $\beta$ 값이 더 커지면 더 많은 날짜의 기온의 평균을 이용하기 때문에 곡선이 더 부드러워 진다. 그러나 더 큰 범위에서 기온을 평균하기 때문에 곡선이 올바른 값에서 더 멀어진다. 그래서 기온이 바뀔 경우에 지수가중평균 공식은 더 느리게 적응한다. 
  + $\beta$ 값이 더 커지면 이전 값에 많은 가중치를 주고 현재의 기온에는 작은 가중치를 주게 된다.

## **Understanding exponentially weighted averages**
<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/ffb997be-47a9-4dce-8ac0-452c67ebad73">

<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/fce08988-5e0d-4f40-b911-f9bacb301ab3">

+ $\beta=0.9$ 일 때, 어떤 시점에서 앞서 나온 지수 가중 이동 평균 식을 하나의 값으로 정리하여 표현하게 되면 아래와 같다.
  + $v_{​1​​00}=0.1\theta_{100}​​+0.1×0.9\theta_{​99}​​+0.1×(0.9)^​2​\theta_{​98​}​+⋯ $
  + 이를 그림으로 표현 ➡️ 지수적으로 감소하는 그래프($v_{​100}$​ 을 기준으로 보았을 때), 
  + 이유: $v_{​100}$​​ 은 각각의 요소에 지수적으로 감소하는 요소( $0.1×(0.9)​n$​​ )를 곱해서 더한 것이기 때문(요소별 곱셈)
  + 앞에 곱해지는 계수들을 모두 더하면 1 또는 1에 가까운 값이 된다. ➡️ 편향보정
+ 얼마의 기간이 이동하면서 평균이 구해졌는가?
  + $\beta=(1−\epsilon)$ 라고 정의하면
  + $(1−\epsilon)​n​​=​e​\frac{1}{e}$ 를 만족하는 $n$ 이 그 기간이 되는데, 보통 $\frac{1}{\epsilon}$​​ 으로 구할 수 있다.
  + 온도가 감소하기까지(1/3)이 될 때까지 약 10일이 걸린다.
  + 10일 뒤에는 가중치가 현재 날짜의 가중치의 1/3로 줄어든다.
+ 지수 가중 이동 평균의 장점은 구현시 아주 적은 메모리를 사용한다는 것이다.


## **Bias Correction in Exponentially Weighted Average**
<img width="717" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/5c637b73-141d-4408-ba85-f46b8378ea00">

+ 편향 보정으로 평균을 더 정확하게 계산할 수 있다.
+ 저번 시간에 따른 지수평균식대로라면 $t=1$ 일때 $(1−\beta)$ 를 곱한 값이 첫번째 값이 되는데, 이는 우리가 원하는 실제 $v_1$ 값과 차이가 나게 된다.
+ 따라서 $\frac{v_t}{(1-\beta^t)}$ 를 취해서 초기 값에서 실제값과 비슷해지게 한다.
+ 보통 머신러닝에서 구현하지는 않는다. 시간이 지남에 $ (1-\beta^t) $ 는 1에 가까워 져서 우리가 원하는 값과 일치하게 되기 때문이다.
+ 그렇지만 신경이 쓰인다면 구현하는게 옳다.


## **Gradient Descent with Momentum**
<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/893e4d17-d75a-414b-b38b-bf9786a737a0">

<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/893e4d17-d75a-414b-b38b-bf9786a737a0">

+ 알고리즘은 아래와 같다.
  + $V_{dW} = \beta_1 V_{dW} + (1-\beta_1)dW$
  + $w := w-\alpha V_{dW}$
+ 경사 하강법에 Momentum을 추가하면 항상 더 빠르게 동작한다.
+ 기본적인 아이디어: 경사에 대한 지수가중평균 계산하기 ➡️ 그 값으로 가중치를 업데이트한다.
+ 위 아래로 일어나는 진동은 경사 하강법의 속도를 느리게 하고 더 큰 학습률을 사용하는 것을 막는다. 더 큰 학습룰을 사용하게 되면 overfitting의 위험이 있기 때문이다. (수직축 관점)
+ 그런데 수평축에서는 더 빠른 학습을 원한다.
+ 아래로 내려갈 때 가속
+ Momentum 항들은 속도를 나타낸다고 볼 수 있다.
+ Momentum 의 장점: 매 단계의 경사 하강 정도를 부드럽게 만들어 준다.
+ $\beta$ 는 0~1 사이이기 때문에 마찰을 제공하여 공이 제한 없이 빨라지는 것을 막는다.
+ Momentum 알고리즘에서는 보통 편향 추정을 실행하지 않는다.step이 10 단계정도 넘어가면 편향 추정이 더 이상 일어나지 않기 때문이다


## **RMSprop**
<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/93fa4e38-5743-4ce4-821a-e6e5b41d08c5">

+ 알고리즘은 아래와 같다.
  + $S_{dW} = \beta_2 S_{dW} + (1-\beta_2) dW^2 $
  + 업데이트: $w:= w-\alpha \frac{dW}{\sqrt {S_{dW} + \epsilon}} $
  + $dW^2$ 는 요소별 제곱을 뜻한다.
+ RMSProp 의 장점: 미분값이 큰 곳에서는 업데이트시 큰 값으로 나눠주기 때문에 기존 학습률보다 작은 값으로 업데이트된다. 
  + 따라서 진동을 줄이는데 도움이 된다 
  + 반면 미분값이 작은 곳에서는 업데이트시 작은 값으로 나눠주기 때문에 기존 학습률 보다 큰 값으로 업데이트 된다. 이는 더 빠르게 수렴하는 효과를 불러 온다.
+ 수평 방향에서는 학습률이 꽤 빠르게 가기를 원하지만, 수직 방향에서는 느리게 혹은 수직 방향의 진동을 줄이고 싶어 한다.
+ 실제로 수직 방향의 도함수가 수평 방향의 도함수보다 훨씬 큰데, 이렇게 나누면 수직 방향에서는 진동을 줄이는데 도움을 주고, 수평 방향에서는 작은 숫자로 나눴기 때문에 빠르게 학습할 수 있다. 
+ $w, b$ 는 분리를 위한 표현이다. 실제로 $ dw$ 와 $ db$ 는 매우 고차원의 매개변수 벡터이다. 
+ 도함수를 제곱해서 결국 제곱근을 얻기 때문이다. 



## **Adam Optimization Algorithm**
<img width="712" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/5490f999-b154-4ba6-8ca0-0b331eae0b5d">

+ 넓은 범위의 딥러닝 아키텍처 서로 다른 신경망 최적화 알고리즘들은 넓은 신경망에서 일반적으로 잘 작동하지 않는다. 넓은 범위의 딥러닝 아키텍처 
+ Adam 은 Momentum 과 RMSProp 을 섞은 알고리즘이다.
+ 알고리즘은 아래와 같다.
  + $V_{dW} = 0, S_{dW} = 0$ 초기화시킨다.
  + Momentum 항: $V_{dW} \beta_1 V_{dW} + (1-\beta_1)dW$
  + RMSProp 항: $S_{dW} = \beta_2 S_{dW} + (1-\beta_2) dW^2 $
  + Bias correction: $V_{dW}^{correct} = \frac{V_{dW}}{1-\beta^t_1}, S_{dW}^{correct} = \frac{S_{dW}}{1-\beta^t_2}$
  + 업데이트: $w:=w-\alpha \frac{V_{dW}^{correct}}{\sqrt{S_{dW}^{correct}+\epsilon}} $
+ Adam: Adaptive moment estimation의 약자
  + 첫 번재 모멘트: $\beta_1$ 이 도함수의 평균을 계산함
  + 두 번째 모멘트: $\beta_2$ 가 지수가중평균의 제곱을 계산함

<img width="664" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/f7dff801-1f83-42f3-8e2e-f76a629d0214">

+ 보통 $\apha$ 만 튜닝하는 것이 보편적이다. 


## **Learning Rate Decay**
<img width="695" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/721b5109-cbfb-453f-930c-6b2d28299137">

+ 시간에 따라 학습률을 천천히 줄이면 학습 알고리즘의 속도를 높일 수 있다. 
+ 작은 미니배치 일수록 잡음이 심해서 일정한 학습률이라면 최적값에 수렴하기 어려운 현상을 볼 수 있다.
+ 학습률 감쇠 기법을 사용하는 이유는 점점 학습률을 작게 줘서 최적값을 더 빨리 찾도록 만드는 것이다.
  + 최솟값 주변의 밀집된 영역에서 진동하게 된다.
+ 다양한 학습률 감쇠 기법들이 있다.
  + 1 epoch = 전체 데이터를 1번 훑고 지나가는 횟수
  + $ \alpha = \frac{1}{1+ \operatorname{decayrate} \times \operatorname{epoch num}} \alpha_0 $
  + 지수식 감쇠: $ \alpha = 0.95 ^{\operatorname{epoch num}} \alpha_0 $ (exponential decay라 부른다)
  + $ \alpha = \frac{k}{\sqrt{\operatorname{epoch num}}}\alpha_0 $
  + $ \alpha = \frac{k}{\sqrt{\operatorname{batch num}}}\alpha_0 $
  + 이산 계단: step 별로 $\alpha$ 다르게 설정

Source
+ [지수 가중 평균](https://bruders.tistory.com/92)