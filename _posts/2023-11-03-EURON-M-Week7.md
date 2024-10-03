---
title: Deep Learning Specialization 2-3 | Basics of Neural Network Programming
author: Su
date: 2023-11-03 05:55:00 +0800
categories: [DL]
tags: [EURON]
pin: false
use_math: true
---

Learning Source
+ [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning?utm_source=deeplearningai&utm_medium=institutions&utm_campaign=SocialYoutubeDLSC1W1L1#courses)
+ [부스트코스 딥러닝 2단계: 심층 신경망 성능 향상시키기](https://m.boostcourse.org/ai216/lectures/132205)

<br>

## **Normalizaing inputs**
<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/e1224a23-14e6-478f-bde1-aebf02683264">


+ 정규화 단계
  + 1️⃣ 평균을 0으로 만든다.
    + $ \mu = \frac{1}{m} \sum_{i=1}^m x^{(i)} $
    + $ x := x - \mu $
  + 2️⃣ 분산을 1로 만든다.
    + $ \sigma^2 = \frac{1}{m} \sum_{i=1}^m {x^{(i)}}^2 $
    + $ x := \frac{x}{\sigma} $
    + 위 강의 사진에서 $\sigma^2$ 는 오타이며, $\sigma$ 가 맞다.
    + `$**$`: 요소별 곱셈
+ test set을 정규화할 때 train set와 같은 $\mu$ 와 $\sigma$ 를 사용해야 한다.
+ input을 정규화하지 않으면 매우 구부러진 활처럼 가늘고 긴 모양의 비용함수가 된다.
+ 정규화를 통해 비용함수의 모양은 더 둥글고 최적화하기 쉬운 모습(대칭적)이 된다. ➡️ 학습 알고리즘이 더 빨리 실행된다.




## **Vanishing/exploding gradients**
<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/d4c0f5e1-d96e-4bb3-8360-f10b7b6af979">


+ <code>Vanishing/exploding gradients</code>: 매우 깊은 신경망을 훈련시킬 때 나타나는 문제점이다.
+ 예를 들어 $g(z)=z, b^{[l]}=0$ 이라고 가정했을 때 $\hat y = w^{[l]}w^{[l-1]}...w^{[2]}w^{[1]}x$ 가 된다.
+ 이때 모든 가중치 행렬 $ w = 1.5E$ 라고 가정하면($E$ 는 단위행렬) $\hat y = 1.5^{(l-1)} Ex$ 가 된다.
+ 더 깊은 신경망일수록 $ \hat y $ 의 값은 기하급수적으로 커진다. 
+ 반대로 모든 가중치 행렬 $w=0.5E$ 라고 가정하면 $\hat y = 0.5^{(l-1)}Ex$ 가 되고 더 깊은 신경망일수록 $\hat y$ 의 값은 기하급수적으로 감소한다. 
+ 이를 토대로 생각하면 경사 하강법에서 $w$ 의 값이 단위행렬보다 큰 값이라면 경사의 폭발, $ w$ 의 값이 단위 행렬보다 작은 값이라면 경사의 소실 문제점이 생긴다.
+ 경사의 소실과 폭발로 인해 학습시키는 데 많은 시간이 걸리기에 가중치 초기화 값을 신중하게 해야한다.
+ 다음 중 경사 소실이 발생활 확률이 가장 높은 모델은?
  + w=0.5E, 5개의 layer ❌
  + w=1.5E, 10개의 layer ❌
  + w=0.3E, 5개의 layer ❌
  + w=0.3E, 10개의 layer ⭕️



## **Weight initialization for deep networks**
<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/bb709895-eb73-4ecf-af4d-1b140726ce88">

+ 가중치 초기화 방법
  + $w_i$ 의 분산을 $\frac{1}{n}$ 으로 설정한다. ($n$ : 입력 특성의 개수)
  + ReLU 활성화 함수를 사용하는 경우 $w$ 의 분산을 $\frac{2}{n^{[l-1]}}$ 으로 설정한다.
  + tanh 활성화 함수를 사용하는 경우 $w$ 의 분산을 $\frac{1}{n^{[l-1]}}$ 또는 $\frac{2}{n^{[l-1]}+n^{[l]}}$ 으로 설정한다.
  + ReLU 활성화 함수를 사용하는 경우 분산을 $\frac{1}{n}$ 보다는 $\frac{2}{n}$ 를 쓰는 게 더 낫다.
+ 일반적인 경우 층 $l$ 은 해당 층의 각 유닛에 대해 $ n^{[l-1]} $ 의 입력을 갖는다.


## **Numerical approximaation of gradients**
<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/8fc0f416-2f31-4b3e-bff1-8373759a78a8">

<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/95ac4213-3228-4a7e-9a35-5c0d3279f10f">

+ 경사 검사를 하는 이유: 역전파를 알맞게 구현했는지 확인하기 위함
+ 중심 차분법의 정의:
  + $f \prime (\theta) = \lim_{\epsilon \to \inf} \frac{f(\theta-\epsilon) - f(\theta+\epsilon)}{2 \epsilon} $
  + $\epsilon$ : 굉장히 작은 수
+ 이 수치 미분은 중심 차분법보다 오차가 더 높다.




## **Gradient Checking**

<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/e886c71d-6c6b-451c-8fa4-0fa1ae17da7d">

<img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/c654ba32-5480-4faf-80bc-51db5868976f">


+ 역전파를 구현할 때 경사 검사 test를 진행한다.
+ 더 큰 삼각형에서 너비 분의 높이를 구하는 것이 $\theta$ 에서의 도함수를 근사하는데 더 나은 값을 제공하기 때문이다. 

+ 우선, 모델 안에 있는 모든 변수($W, b$ )를 하나의 벡터( $\theta$ )로 concatenate한다.
+ 그러면 비용 함수는 $J(W, b)$ 에서 $J( \theta)$ 로 변한다.
+ 그후, 수치 미분을 구한다.
  + $d\theta^{[i]}_{\operatorname{approx}} = \frac{J(\theta_1, ..., \theta_i+\epsilon, ...)-J(\theta_1, ..., \theta_i-\epsilon, ...)}{2\epsilon} $
  + $ W^{[1]}$ 과 $dW^{[1]}$ 의 shape는 같다.
+ 최종적으로 수치 미분과 일반 미분을 비교한다.
  + $d\theta^{[i]}_{\operatorname{approx}} \approx d\theta $
  + 유사도를 계산하는 방법은 유클리디안 거리를 사용한다.
  + $ \frac{\lVert d\theta^{[i]}_{\operatorname{approx}}-d\theta \rVert_2}{\lVert d\theta \rVert_2}$
  + 보통 거리가 $10^{-7}$ 보다 작으면 잘 계싼되었다고 판단한다.
+ 구현 코드
  + 순전파 구현 코드
    ```python
    # GRADED FUNCTION: forward_propagation

    def forward_propagation(x, theta):
      """
      Implemenet the linear forward propagation(compute J) presented in Figure 1 (J(theta) = theta * x)

      Arguments:
      x -- a real-valued input
      theta -- our parameter, a real number as well

      Returns:
      J -- the value of function J. computed using the formula J(theta) = theta * x
      """

      J = np.dot(theta, x)

      return J

      x, theta = 2,4
      J = forward_propagation(X, theta)
      print("J ="k, str(J)) # J = 8
    
    ```
  + 역전파 구현 코드
    ```python
    # GRADED FUNCTION: backward_propagation

    def backward_propagation(x, theta):
      """
      Computes the derivative of J with respect to theta (see Figure 1).

      Arguments:
      x -- a real-valued input
      theta -- our parameter, a real number as well

      Returns:
      dtheta -- the gradient of the cost with respect to theta
      """

      dtheta = x # answer
      return dtheta
    
    x, theta = 2, 4
    dtheta = backward_propagation(x, theta)
    print("dtheta = " + str(dtheta)) # dtheta = 2
    
    ```


## **Gradient Checking implementation notes**
<img width="715" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/a946ffd0-4d82-46b8-8abf-8f6c143ee63d">

+ 속도가 굉장히 느리기 때문에 훈련시에는 절대 사용 하지 않고 디버깅 할때만 사용한다.
+ 알고리즘이 경사 검사에 실패 했다면, 어느 원소 부분에서 실패했는지 찾아본다. 특정 부분에서 계속 실패했다면, 그 경사가 계산된 층에서 문제가 생긴것을 확인할 수 있다.
+ $d\theta$ 는 $\theta$ 에 대응하는 $J$ 의 정규화 항($\frac{d}{dm}\sum_m{\lVert w^{[l]} \rVert^2_F})$ 도 포함하기 때문에 경사 검사 계산시 같이 포함해야 한다.
+ 드롭아웃에서는 무작위로 노드를 삭제하기 때문에 적용하기 쉽지 않다. 따라서 통상 드롭아웃을 끄고 알고리즘이 최소한 드롭아웃 없이 맞는지 확인하고, 다시 드롭아웃을 켠다.
+ 마지막으로 거의 일어나지 않지만 가끔 무작위 초기화를 해도 초기에 경사 검사가 잘 되는 경우이다. 이때는 훈련을 조금 시킨 다음에 경사 검사를 다시 해보는 방법이 있다.