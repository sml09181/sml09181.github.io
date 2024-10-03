---
title: Deep Learning Specialization 2-2 | Basics of Neural Network Programming
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


## **Regularization**
+ overfitting이 의심된다면(=분산이 높다면) 가장 먼저 정규화를 시도해야 한다. training data를 더 추가해도 되지만 cost가 크다. <BR>
<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/4ba9492e-c446-4bc2-b9d6-a29ac28270fd">

+ 비용 함수: $ J(w, b) = \frac{1}{m}\sigma^m_{i=1}L(y^{(i)}, h^{(i)}) + \frac{\lambda}{2m} { \left \| w \right \| }^2 $
  + $w$ 만 정규화하고 $b$ 는 정규화하지 않는다.
  + $w$ 가 high-dimensional ➡️ parameter vector이기 때문
  + 특히 높은 분산을 가질 때 $w$ 는 많은 매개변수를 가진다. 
  + 반면 $b$ 는 하나의 실수이기 때문에 거의 모든 매개변수는 $b$ 가 아닌 $w$ 에 있다.
+ $w: (n^{[l-1]}, n^{[l]})$
  + $n^{[l]}$ : l번째 hidden layer의 은닉 유닛 개수
+ $m$ 앞의 2: 그저 scaling 상수일 뿐
+ <code> L1 정규화 </code>: $\left \| w \right \|_1 = \sigma^{nx}_{j=1}\lvert w_j \lvert$
  + make vector more sparse a lot of zeros in it
+ <code>L2 정규화</code>: $\left \| w \right \|^2_2 = \sigma^{nx}_{j=1}w^2_j$
  + Frobenuius norm: 행렬의 원소 제곱의 합이라는 뜻
    + $ \left \| w^{[l]}\right \| ^2_F = \sigma^{n^{[l-1]}}_{i=1} \sigma^{n^{[l]}}_{j=1} (w^[l]_{ij})^2$
+ L2 정규화는 weight decay라고 불린다.
  + $w^{[l]} = (1-\frac{\alpha \lambda}{m}) w^{[l]} - \alpha$ (역전파에서 온 값들)
  + 위의 식을 보면 기존 경사 하강법과 달리 weight에 1보다 작은 값인 $(1-\frac{\alpha \lambda}{m})$ 이 곱해져 행렬 $w^{[l]}$ 의 값보다 더 작아지게 된다.
+ dev set 또는 cross validation set을 사용한다.
+ 강의에서 lambda를 lambd(mbd)라 표기한 이유: 파이썬 명령어와 충돌하지 않기 위해서<br>

<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/5f239749-c50f-4462-a5ef-721ca616cc70">



## **Why regularization reduces overfitting**
<img width="628" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/684d09dd-cadd-46ea-b602-df9aa81512db">

+ $\lambda$ 값을 크게 만들어서 가중치행렬 $w$를 0에 가깝게 설정 가능 ➡️ 많은 hidden unit을 0에 가까운 값으로 설정할 수 있다. 
  + 즉, 로지스틱 회귀에 가까운 네트워크를 만든다.(더 간단하고 작은 신경망)
  + $J(w^{[l]}, b^{[l]}) = \frac{1}{m} \sigma^{m}_{i=1}L(\hat y^{(i)}) + \frac {\lambda}{2m} \sigma ^L_{l=1} \left \| w^{[l]}\right \| ^2_F$
<br>
<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/bfb39032-0996-48ba-b15c-67f26b4462f1">

 + tanh 활성화 함수를 사용했을 경우  
   + $\lambda$ 값 증가 ➡️ 비용함수에 의해 $w$ 는 감소
   + 이때 $z^{[l]} = w^{[l]}a^{[l-1]} + b^{[l]}$ ➡️ $z$ 도 감소
+ $z$ 가 작을 때 $g(z)$ 는 선형 함수가 되고, 전체 네트워크도 선형이 되기에 과대적합과 같이 복잡한 결정을 내릴 수 없다.<br>
<img width="400" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/219f62c0-496a-49df-9428-2fdc94f1085c">

+ cost function 구현 팁
  + 경사 하강법의 반복의 수에 대한 함수로 비용함수를 설정해라
  + 꼭 정규화 항을 추가하여 타겟으로 하는 비용함수의 단조감소를 보자. 
  + 비용함수의 예전 정의인 첫 항만을 쓰고 있지는 않는지 확인하자.

## **Dropout regularization**
<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/a047f214-1823-4221-975d-12f82e7025cb">

+ <code>dropout</code>
  + 신경망의 각각의 층에 대해 노드를 삭제하는 확률을 설정 삭제할 노드를 랜덤으로 선정
  + 삭제된 노드의 들어가는 링크와 나가는 링크를 모두 삭제합니다.
  + 경사 하강법의 하나의 반복마다 0이 되는 은닉 유닛들이 달라진다. 
  + 같은 훈련 세트를 두 번째 반복할 때는 0이 되는 은닉 유닛의 패턴이 달라진다.
+ 효과: 더 작고 간소화된 네트워크가 만들어진다.<br>

<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/088543ad-19a8-4ce4-8621-e700a7cdaecc">

+ <code>Inverted dropout</code>(역 드롭아웃): 위와 같이 노드를 삭제후에 얻은 활성화 값에 `keep_prob`(삭제하지 않을 확률)을 나눠 주는 것
+ `keep_prob`로 나눠주는 이유: $a3$ 의 기댓값을 유지하기 위해서(기존에 삭제하지 않았을 때 활성화 값의 기대값으로 맞춰주기 위함)
+ $d3$: 어떤 노드를 0으로 만들지 결정
  + 정방향 전파와 역전파 모두
+ scaling 문제가 작기 때문에 test하기 쉽다. <br>

<img width="557" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/1b50e095-6a15-48eb-8327-09e02e26b5b6">

+ test에서는 드롭아웃을 사용하지 않는다. ➡️ 노이즈만 증가시킬 뿐
+ 즉 드롭아웃을 적용하지 않고, 무작위로 유닛을 제거하지 않으며, 훈련 시 사용했던 `1/keep_prob` 인자를 유지하지 않는다.
+ 
## **Understanding dropout**
<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/7864469b-bd1b-40b3-b89b-6653e9ea031e">

+ dropout: 랜덤으로 노드 삭제 ➡️ 하나의 특성에 의존X ➡️가중치를 다른 곳으로 분산시키는 효과
+ 드롭아웃의 `keep_prob` 확률은 층마다 다르게 설정 할 수 있습니다.
+ 모든 반복에서 잘 정의된 비용함수가 하강하는지 확인하는게 어려워집니다. 따라서 우선 드롭아웃을 사용하지 않고, 비용함수가 단조감소인지 확인 후에 사용해야 합니다.
 
+ dropout은 L2 regularization과 비슷한 효과를 보인다.
  + L2가 다른 가중치에 적용된다는 것과 서로 다른 크기의 입력에 더 잘 적응한다만ㅇ 다른 부분 
+ 매개변수가 많은 층, 즉 과대적합의 우려가 많은 층은 더 강력한 형태의 드롭아웃을 위해 `keep_prob`를 작게 설정한다. 
  + 단점: 교차 검증을 위해 더 많은 hyperparameter가 생긴다. 
+ dropout 단점: 비용함수 J가 더이상 잘 정의되지 않는다. 그래프 그리기 어려워진다. 
+ 보통 `keep_prob`를 1로 설정하여 dropout 효과를 멈추고 코드를 실행시켜 J가 단조감소하는지 확인 그 이후에 드롭아웃을 주고 코드를 수정하지 않는다. 

## **Other regularization methods**
### Data augmentation(데이터 증식)
+ 이미지의 경우 더 많은 훈련 데이터를 사용함으로서 overfitting 해결 가능
+ 보통 이미지를 대칭, 확대, 왜곡 혹은 회전시킴
+ 완전히 새로운 독립적인 sample을 얻는 것보다 더 많은 정보를 추가해주지는 않지만, 컴퓨터적인 비용이 들지 않는다는 게 장점이다. <br>
<img width="575" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/56c072be-e087-4bcf-8593-279b516b8d3f">

### Early stopping
<img width="597" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/2bbb32a5-ec63-4249-b50d-e54f68bf4814">

+ 단조하강함수인 test set error와 dev set error을 동시에 그린다.
  + $w$ : 작은 값으로 무작위 초기화 
+ dev set error가 어느 순간부터 하락하지 않고 증가하기 시작하는 것이라면 overfitting이 시작하는 시점이다. ➡️ 이때 stop
+ 단점
  + training시 작업은 크게 2가지로 나뉜다.
  + 1️⃣: training 목적인 cost function를 최적화 시키는 작업
  + 2️⃣: overfitting 하지 않게 만드는 작업이 있습니다. 
  + 두 작업은 별개의 일이라서 두 개의 다른 방법으로 접근해야 한다.
  + 그러나 early stopping은 혼합된 방법이기 때문에 최적의 조건을 찾지 못할 수도 있다.



<br>

<br>
<br>
Latex 절댓값 기호(|)로 고생 ing...

+ $\lvert \rvert$:\lvert \rvert
+ $\left \| \right \|$: \left \| \right \|