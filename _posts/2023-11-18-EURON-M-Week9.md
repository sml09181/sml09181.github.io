---
title: Deep Learning Specialization 2-5 | Basics of Neural Network Programming
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

## **Tuning Process**
<img width="304" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/afec93d3-e091-47cf-9e2f-0794e19179b1">

+ 딥러닝에는 다양한 hyperparameter가 존재한다. 상황에 따라 다를 수도 있지만, 보통 우선 조정하는 순서로 나열되었다.
  + 학습률( $\alpha$ )
  + Momentum 알고리즘의 $\beta$
  + 은닉 유닛의 수
  + 미니배치 크기
  + 은닉층의 갯수
  + 학습률 감쇠(learning rate decay) 정도
  + Adam 알고리즘의 $\beta_1$, $\beta_2$, $\epsilon$ <br>

<img width="625" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/771e064b-855a-4241-b9e3-cf920843c312">

+ <code>무작위 접근 방식</code>
  + 매우 좋은 접근 방식
  + 어떤 하이퍼파라미터가 문제 해결에 더 중요한지 미리 알 수 없기 때문에 사용<br>

<img width="530" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/026a7355-4b6f-4424-a8ea-01d5083f2711">

+ <code>정밀화 접근</code>
  + 다른 일반적 접근 방식 중 하나
  + 우선 전체 하이퍼파라미터 공간에서 탐색하여 좋은 점을 찾은 후, 그 근방에서 더 정밀하게 탐색
+ 정리
  + 1️⃣ Random something, not a grid search
  + 2️⃣ 원한다면 정밀화 접근을 이용할 수 있다.


## **Using an appropriate scale to pick hyperparameters**
<img width="509" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/5d242802-3c9a-40ec-80bc-0610bc7c8023">

+ 선형 척도가 합리적인 parameters
  + Random 추출 in 균일 분포(Uniform distribution)
  + 특정 Layer의 hidden unit 개수
  + 신경망의 Hidden Layer 개수<br>

<img width="626" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/7103baa5-d3c9-4e42-b67b-40e5e7425468">

+ 로그 척도가 합리적인 parameters
  + 학습률<br>
    + 1 과 0.0001 사이의 값중에 균일하게 무작위 값을 고르게 되면, 90%의 값이 1 과 0.1 사이에 존재하기 때문에 공평하다고 할 수 없다.
    + 위 예시에 따르면 0과 -4 사이에 균일하게 무작위로 고르고 10의 지수로 바꿔주는 것
    
    ```python
    r = -4 * np.random.rand() # r in [-4, 0]
    alpha = 10**r             # alpha in [10^(-4), 1]
    ```
<br>
<img width="625" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/876f9419-73a6-40f7-84b8-892592d0ec58">

  + $\beta$ in 지수 가중 이동 평균
    + 마찬가지로 0.9 와 0.999 사이의 값을 탐색하는 것은 비합리적이기 때문에 $1-\beta$  를 취해준 후, 위의 예시와 마찬가지로 로그척도에서 무작위 값을 선택하여 탐색한다.
    + 선형 척도를 사용하지 않는 이유: 위의 값들은 1에 가까울수록 $\beta$ 가 아주 조금만 바뀌어도 알고리즘 결과에 더 큰 영향을 끼치기 때문
      + $\beta$ = 0.9 ➡️ 0.9005: 대략 10개의 값을 평균내는 것
      + $\beta$ = 0.999 ➡️ 0.9995: 1000개에서 2000개를 가중평균 내는 것으로 바뀜
    + $\beta$ 가 1에 가까운 곳에서 더 조밀하게 뽑는다.

## **Hyperparameters tuning in practive: Pandas Vs.Caviar**
<img width="614" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/70bd99ff-916e-4fc0-8918-889a094b5d21">

+ 하이퍼파라미터 튜닝 방법 <br>
  <img width="625" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/975ac2f7-beea-4122-9140-de55e75ca874">
+ 1️⃣ <code>Baby Sitting One Model</code> = `판다 접근`
  + 데이터는 방대하지만 CPU나 GPU 등 컴퓨터 자원이 많이 없어서 적은 숫자의 모델을 한 번에 학습시킬 수 없을 때 사용
  + 하나의 모델로 매일 성능을 지켜보면서, 학습 속도를 조금 씩 바꾸는 방식
+ 2️⃣ <code>Training Many Models in Parallel</code> = `캐비어 접근`
+ 컴퓨터의 자원이 충분히 많아 여러 모델을 한번에 학습 시킬 수 있을 때 사용

