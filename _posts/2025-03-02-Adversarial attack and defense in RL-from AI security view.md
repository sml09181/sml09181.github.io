---
title: 2025-03-02-Adversarial attack and defense in RL-from AI security view
author: Su
date: 2025-03-02 11:00:00 +0800
categories: [Paper Review]
tags: [RL, Security]
pin: false
use_math: true
---


> Springer 2019. [Paper](https://cybersecurity.springeropen.com/counter/pdf/10.1186/s42400-019-0027-x.pdf) Chen, T., Liu, J., Xiang, Y. _et al._

## 🍒 Key Takeaways

## Abstract

-   AI security 관점에서 RL에서의 Adversarial attack에 대한 최초의 comprehensive survey
-   existing adversarial attacks에 대한 대표적인 defense technologies against existing adversarial attacks에 대해서도 간략히 소개

## 1. Introduction

-   Huang et al. (2017)
    -   RL은 input에 작은 perturbation를 추가한 adversarial attack하다는 것을 처음으로 발견하였다.
    -   **cross-dataset transferability** in RL: 동일한 task를 위한 policy 간에 adversarial examples이 전이될 수 있음을 확인하였음.
-   주요 연구 분야
    -   `Atari Game`: 매 time step마다가 아닌, 특정 step에서만 independently하게 adversarial examples를 생성하여 공격하는 방식
    -   `Auto Path Planning`

<img width="593" alt="Image" src="https://github.com/user-attachments/assets/22ad6477-59ed-4473-a625-d6689730598f"/>

## 2. Preliminaries

### Definition

-   `Adversarial Example`
	- information carrier (such as image, voice or txt) with small perturbations added
	-  can remain imperceptible to human vision system
	- 1️⃣ **Implicit Adversarial Example**: pixel 수준에서 사람에게 보이지 않는 perturbations을 추가하여 global information를 수정함 
	- 2️⃣ **Dominant Adversarial Example**
		- a modified version of clean map
		- physical-level obstacles을 추가하여 e local information를 변경하였다. 
-   `Transferability`: 한 모델에서 잘못 분류되는 adversarial example가 동일한 task를 해결하도록 훈련된 다른 모델에서도 잘못 분류되는 성질
-   `Threat Model`
	- Finding system potential threat to establish an adversarial policy
	- policy의 raw input에 small perturbations를 넣을 수 있는 adversary를 고려한다.
-   `Target Agent`
	- adversarial examples에 의해 공격받는 대상 주체
	- 일반적으로 RL policy으로 훈련된 NN

### UNREAL 알고리즘 
- RL algorithm 자체는 아님 
-   적용 분야:
	- `Atari Pong Game` : 8.8 times against human performance
	- `Labyrinth` (1인칭 3D 미로): reached 87% of human level
-   RL algorithm: `A3C` 알고리즘 기반

UNREAL에는 2가지 auxiliary tasks가 있다.
- 1️⃣ **Control Task**
	- Pixel Control: 픽셀 단위로 변화를 추적하여 environment를 더 잘 이해하도록 학습한다. 
	-   Hidden Layer Activation Control: 학습 안정성 강화에 도움이 된다. 
- 2️⃣ **Back Prediction Task**
	-   일부 시나리오에서는 feedback $r$ 을 얻을 수 없다. 
	-   NN이 다음 단계의 feedback value 을 예측하도록 학습 → 표현력 향상

-   UNREAL은 과거 연속된 다중 프레임 이미지 입력을 사용해 다음 단계의 피드백 값을 예측하고 이를 학습 목표로 설정합니다.
    
historical continuous multi-frame image input으로 다음 단계의 feedback value를 예측한다. 그리고 그 feedback value를 training target으로 설정한다. 또한 history information를 활용하여 Value Iteration Task를 강화한다.  

찾아보니 UNREAL Engine과 Python Deep RL algorithm 간의 TCP 통신을 지원하는 미완성 [plugin](https://docs.isaratech.com/ue4-plugins/drl-unreal) 이 있었다. bridge environment로 Unreal과 Python 사이의 data를 송수신하는 게 가능하다는 걸 알게 되어 같이 넣어 보았다. 이 plugin의 구성은 아래 그림과 같다. 

![Image](https://github.com/user-attachments/assets/24e5ef5c-b6af-498d-98cd-85c989882009)


## 3. White-box Adversarial attack in RL

### 🎯 Fast gradient sign method (FGSM)

-   적용 분야: `Atari Pong Game`
    -   agent가 공의 이동 방향을 제대로 판단하지 못함
-   RL algorithm
    -   `DQN`(Deep Q-Network): 가장 취약 → 높은 공격 성공률
    -   `TRPO/A3C`: 상대적으로 높은 resistance

<img width="770" alt="Image" src="https://github.com/user-attachments/assets/c2a15090-9e94-4a5f-8425-ca376eca79f2" />

원본 input $x$ 에 충분히 작은 perturbation $η$ 를 element-wise로 더해 $x$ 에 대한 adversarial example $\tilde x$ 를 만들 수 있다. 즉, $\tilde x = ω^T\tilde x = ω^Tx + ω^Tη$ 이다. 이때 $η$ 를 이루는 값들은 $-ϵ$ 보다 크고, $ϵ$ 보다 작아야 한다. 이로써 classifier는 $x$ 와 $\tilde x$ 를 같은 class로 분류하게 된다. 

$$ η=ϵ⋅sign(ω), ∥η∥_{∞}<ϵ$$

   -   $ω$: weight vector
   -   $η$ maximizes the change in output for the adversarial example $\tilde x$
   - $ϵ$ : perturbation 크기를 조절하는 hyper-parameter

FGSM은 cost function $J$ 를 linearization하여 classifier의 misclassification을 일으키는 $η$ 를 계산한다. $∇_xJ(θ,x,y)$ 는 $x$ 에 대한 $J$의 변화율을 나타낸다. 
즉, $η$ 는 GT인 $y$ 에 반대되는 방향으로 그 부호를 얻게 된다. 
$$ η=ϵ⋅sign(∇_xJ(θ,x,y)) $$ 

    
### 🎯 Start Point-based Adversarial Attack on Q-learning (SPA)

-   적용 분야: `Automatic Path Finding`
	- **Key Point** ($k$)
		- strategically important location
		- e.g.) starting point, target point, 경로 상의 주요 분기점
	- **Key Vector** ($v$)
		- Key Point($k$)에서 목표 지점($t$)까지의 directional vector 
		- $v=(t_c−k_c,t_r−k_r)$
		- $t_c, t_r$: Coordinates of the target point
-   RL algorithm: `Q-learning`
	- Q-learning 기반 Automatic Path Finding에서 처음으로 adversarial example를 성공적으로 찾아냈다(precision: 70%).
	-   BUT limited, fixed map size (28×28)
- probabilistic output model
	- **STEP1**: 4가지 factor를 계산
	- **STEP2**: 각 adversarial point ($a_i$)이 agent의 경로 탐색을 inference할 가능성을 각 factor와의 가중합으로 계산한다. 
	- **STEP3**: 모든 $p_{a_i}$ 값 중 TOP 10을 선택한다.

<img width="739" alt="Image" src="https://github.com/user-attachments/assets/b717b440-c11b-4269-b4e4-ddadab65a346" />

<br>

STEP1의 4가지 factor는 다음과 같다. 

1️⃣ **Energy Point Gravitation**
adversarial point  $k$ 가 key vector  $v$ 상에 위치할수록 공격 성공 가능성이 높아진다. row와 column 에서 각각 값이 하나씩 나온다. 이때 key point $k$ 와 adversarial point  $k$ 가 다른 것에 유의하자. 

$$ \begin{cases}
e_{ic} = k_c + i \cdot d' \cdot \frac{k'_c - k_c}{\sqrt{(k'_c - k_c)^2 + (k'_r - k_r)^2}} \\
e_{ir} = k_r + i \cdot d' \cdot \sqrt{1 - \left( \frac{k'_c - k_c}{\sqrt{(k'_c - k_c)^2 + (k'_r - k_r)^2}} \right)^2}
\end{cases} $$

2️⃣ **Key Point Gravitation**
adversarial point가 이 key point $k$에 가까울수록 방해 가능성이 높아짐:

$$ d_{1i} = |a_{ic} - k_c| + |a_{ir} - k_r| \\
\text{where } (k_c, k_r) = k, \ (a_{ic}, a_{ir}) = a_i \in A $$

3️⃣ **Path Gravitation**
적대적 점이 초기 경로 $Z_1$ 에 가까울수록 방해 가능성이 높아짐
$$ d_{2i} = \min \left\{ d_2 \ \bigg| \ d_2 = |a_{ic} - z_{jc}| + |a_{ir} - z_{jr}|, \ z_j \in Z_1 \right\} \\
\text{where } (z_{jc}, z_{jr}) = z_j, \ (a_{ic}, a_{ir}) = a_i \in A $$ 

4️⃣ **Included Angle**
key point $k$에서 adversarial point $a_i$로 향하는 벡터와 목표 지점 $t$ 로 향하는 벡터(key vector) 간의 각도 $θ_i$ 를 나타낸다. 각도 $θ_i$ 가 작을수록(key vector와 유사한 방향) 공격 효과가 증가하는 경향이 있다. 여기서부터는 adversarial point의 표기가 $a_i$ 로 바뀌고, key vector의 표기가 ${v}_{kt}$로 바뀐 것에 유의하자.

$$ \begin{aligned}
\mathbf{v}_{ka} &= (a_{ic} - k_c, \ a_{ir} - k_r) \\
\mathbf{v}_{kt} &= (t_c - k_c, \ t_r - k_r) \\
\cos \theta_i &= \frac{\mathbf{v}_{ka} \cdot \mathbf{v}_{kt}}{|\mathbf{v}_{ka}| \ |\mathbf{v}_{kt}|} \\
\theta_i &= \arccos(\cos \theta_i)
\end{aligned} $$ 

STEP2, 즉 각 adversarial point  $a_i$의 확률은 다음과 같이 계산한다.

$$\begin{equation}
p_{a_i} = \sum_{j=1}^{4} \omega_j f_j(a_i) = \omega_1 \cdot a_{ie} + \omega_2 \cdot d'_{1i} + \omega_3 \cdot d'_{2i} + \omega_4 \cdot \theta'_i
\end{equation}
$$
    
- $ω_j$: the weight for each factor respectively
	- PCA를 통해 계산된다. 
    
마지막으로 STEP3에서는 모든 $p_{a_i}$ 값 중 TOP 10을 선택한다. 

### 🎯 White-box based adversarial attack on DQN (WBA)
- SPA의 확장 버전 
- 적용 분야: `Automatic Path Finding`
-  RL algorithm: `Q-learning`
- SPA 알고리즘을 확장하여 DQN의 Q-table 변동 패턴을 분석하고, 경로 탐색 과정에서 취약점(vulnerable points)을 식별하는 방법


### 🎯 Common dominant adversarial examples generation method (CDG)

## 4. Black-box Adversarial attack in RL

### 🪇 Policy induction attack (PIA)

### 🪇 Specific time-step attack

### 🪇 Adversarial attack on VIN (AVI)

## 5. Defense technology against adversarial attack

## 6. Conclusion and discussion

## 🍋 After Read
