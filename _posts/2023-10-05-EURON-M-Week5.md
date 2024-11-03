---
title: Deep Learning Specialization 2-1 | Basics of Neural Network Programming
author: Su
date: 2023-10-05 05:55:00 +0800
categories: [DL]
tags: [EURON]
pin: false
use_math: true

---

Learning Source
+ [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning?utm_source=deeplearningai&utm_medium=institutions&utm_campaign=SocialYoutubeDLSC1W1L1#courses)
+ [부스트코스 딥러닝 2단계: 심층 신경망 성능 향상시키기](https://m.boostcourse.org/ai216/lectures/132205)

<br>

## **Train/Dev/Test Sets**
<img width="660" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/3f03a849-627e-4f04-b5a8-e1acfd605b8c">

+ 신경망 훈련시 고려해야 할 요소
  + number of layers
  + number of hidden units
  + learning rates
  + activation functions
  + etc
+ 좋은 hyperparameter 값을 찾기 위해 cycle을 여러 번 반복하여 최적의 값을 선택한다. 
  + 이때 최고의 선택은 가지고 있는 데이터의 양, input feature의 개수, GPU나 CPU 등 다양한 요인에 따라서도 결정된다. 
+ 어떤 분야나 application의 직관이 다른 application 영역에 거의 적용되지 않는다.(ex: 물건 운반)
+ cycle을 얼마나 효율적으로 돌 수 있는지에 대한 고민과 data set을 잘 설정하는 것으로 빠른 진전이 가능하다.<br>

<img width="660" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/739a7f9b-0a66-4762-83a0-944620dd0e60">

+ train, dev, test set 설정
  + <code>train set</code>: 훈련을 위해 사용되는 데이터
  + <code>dev set</code>(=cross validation set): 다양한 모델 중 어떤 모델이 좋은 성능 나타내는지 확인
  + <code>test set</code>: 모델이 얼마나 잘 작동하는지 확인
+ train, dev, test set를 설정하는 것은 반복을 더 빠르게 하고, 또한 알고리즘의 편향과 분산을 더 효율적으로 측정할 수 있도록 한다.
+ 100만 개 이상의 빅데이터 시대: 훨씬 더 작은 비율로 dev set와 test set 크기를 지정하는 게 트렌드
  + dev set의 목표: 서로 다른 알고리즘을 시험하고 어떤 알고리즘이 더 잘 작동하는지 test하는 용도 ➡️ dev 세트는 평가할 수 있을 정도로만 크면 된다.
  + test의 목표: 최종 분류기가 어느 정도 성능인지 신뢰있는 추정치를 제공
  + 즉 train/dev/test은 98만/1만/1만 이렇게 해도 충분하다. 
  + 데이터 크기가 더 커지만 dev set과 test set의 비율은 더 작아진다. <br>

<img width="656" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/e161c3cb-9c67-4d13-8f76-2e159b10f549">

+ 두 가지 데이터셋의 분포가 다르다. ➡️ 🌟이때 dev set과 test set은 반드시 같은 분포에서 와야 한다.🌟
+ 비편향 추정이 필요 없는 경우에는 test set이 없어도 된다.
+ ML에서 별도의 test set 없이 train set과 dev set만 있는 경우, dev set를 test set라고 부르는 게 대다수이다. ➡️ test set에 과적합되기 때문에 완벽히 좋은 용어는 아니다.



## **Bias/Variance**
<img width="658" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/eac0c426-8505-4f14-a249-223f0f7e7ef8">

+ 위와 같은 2차원의 예제에서는 데이터를 나타내고 편향과 분산을 시각화할 수 있다. 
+ 하지만 높은 차원의 문제에서는 데이터를 나타내거나 결정 경계를 시각화할 수 없다. 

+ bias-variance tradeoff
  
|높은 편향(high bias)|underfitting|
|알맞음(just right)|-|
|높은 분산(high variance)|overfitting|

<br>
<img width="659" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/01e38ca7-6891-4caa-978b-972ff37fde8b">

+ train set과 dev set의 관계
|_|높은 var(overfitting)|높은 bias(underfitting)|높은 bias & 높은 var|낮은 bias & 낮은 var|
|---|---|---|---|---|
|train set|1%|15%|15%|0.5%|
|dev set|11%|11%|30%|1%|

+ 가정: 기본적으로 인간 수준의 성능이 되어야 한다. 
  + 위 예제: 개와 고양이를 분류할 때 인간 수준의 성능은 0% 에 가까울 것이다. 
  + 조금 더 일반적: 베이지안 최적 오차가 0% 라는 가정이 깔려 있다.
  + 최적 오차 혹은 베이즈 오차에 따라 bias 또는 varianc가 높은지 낮은지가 결정된다.


+ train set에 과적합되어 dev set가 있는 교차 검증 set에서 일반화되지 못한 경우: high variance
+ 🌟checking bias: train set의 error로 확인
+ 🌟checking variance: train set에서 dev set으로 갈 때 오차가 얼마나 커지는지(일반화의 정도)로 확인
+ 이 모든 것은 베이즈 오차가 꽤 작고 train set와 dev set가 같은 확률 분포에서 왔다는 가정 하에 이루어진다. 
<br>

<img width="659" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/1f7ea628-4ca0-4483-888e-770434b4cb10">

+ 높은 bias와 높은 variance: 일부 데이터에 대해 과대적합(overfitting)


## **Basic "recipe" for machine learning**
<img src="https://github.com/sml09181/sml09181.github.io/assets/105408672/48102b09-e4d5-4d4a-939d-1be417c901b4" width="500">

+ bias가 클 때 할 수 있는 것
  + 신경망을 더 깊게 만든다.
  + 더 오래 훈련시킨다.
  + 은닉층 안의 unit 수를 증가시킨다.
    + 참고) bias 문제가 있을 때 data를 더 얻는 것은 크게 도움이 되지 않는다.
+ variance가 클 때 할 수 있는 것
  + regularization을 추가한다.
  + data를 더 많이 추가한다. 

<br>
<img width="656" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/56d4d9c2-745d-4841-835a-5e9dc1935a77">

+ 초기 ML 시대에는 bias-variance tradeoff에 대한 많은 논의가 있었지만, 지금(빅데이터 시대)은 그렇지 않다. 
  + 더 큰 네트워크를 갖는 것이 대부분 분산을 해치지 않고 편향만을 감소시킨다. (정규화를 올바르게 했다면)
  + 데이터를 더 얻는 것도 대부분 편향을 해치지 않고 분산을 감소시킨다.
  + 즉 biggger network나 more data는 편향만을 감소시키거나 분산만을 감소시키는 도구가 된다. 