---
title: Federated Learning-FedSGD, FedAvg
author: Su
date: 2024-12-02 11:00:00 +0800
categories: [Paper Review]
tags: [Audio]
pin: false
use_math: true
---

[Paper Link](https://arxiv.org/pdf/1602.05629)

## 🥑 Key Takeaways
- 1️⃣ FL 개념을 처음으로 제시하였다.  
- 2️⃣ IID, Non-IID 상관없이 **각 모델의 intialization point가 같아야** updated local weight들을 average하여 global weight에 반영하는 FedAVG학습 방식이 잘 작동한다.


# Federated Learning
- Background: 개인정보 보호가 중요하거나, 데이터 양이 많은 경우 traditional한 data center에서의 학습이 어렵다.
- **Federated Learning**: 학습 데이터를 모바일 기기에 분산시킨 채로, local에서 계산된 updated를 모아서 공유 모델을 학습하는 방법

## 장점
- raw training data에 직접 접근할 필요 없이 학습을 진행할 수 있다.
- 서버에 대한 신뢰는 여전히 필요하지만, 학습 목표가 각 클라이언트의 데이터로 정의될 수 있다면, Federated Learning은 개인정보 보호와 보안 risk를 크게 줄일 수 있음.
-   공격 표면이 기기만 국한되므로, 클라우드와의 연결을 최소화할 수 있음.

## Federated Optimization
- **Federated Optimization**: Federated learning에서 암묵적으로 발생하는 최적화 문제
- 분산 최적화와의 차이:
![](https://blog.kakaocdn.net/dn/S4G51/btsGn1w4t3J/0XRwvL0fbmLCoNYUPYkgbK/img.png)
	- 1️⃣ Non-IID: 각 클라이언트의 학습 데이터는 특정 사용자의 기기 사용에 따라 달라져 해당 클라이언트의 데이터셋은 전체 분포를 대표하지 않음.
	- 2️⃣ Unbalanced: 일부 사용자는 다른 사용자보다 서비스를 훨씬 더 많이 사용하여, 로컬 데이터 양이 다름.
	- 3️⃣ Massively distributed: 참여하는 클라이언트의 수가 각 클라이언트의 예시 수보다 훨씬 많음.
	- 4️⃣ Limited communication: 모바일 기기는 종종 오프라인 상태거나 느리거나 비싼 연결을 사용.
- 이 논문에서는 **non-IID**와 **unbalanced** 특성에 중점을 두고, **communication cost**의 중요성을 살펴 본다. 

## Communication cost
- communication cost: client와 server 간의 자료 송수신(communication)에 드는 비용
- parameter 크기, # device, 그리고 server와 client간의 거리의 영향을 받는다.
- Data center optimization
	- communication costs가 상대적으로 작고, computation costs가 크다.
	- 최근 GPU의 발달로 computation costs가 줄어들었다. 
- federated optimization
	- communication costs가 지배적이다. 보통 upload bandwidth는 1MB/s 이하로 제한된다
	- client는 주로 충전 중이고, 유료가 아닌 **Wi-Fi** 연결에 있을 때만 최적화에 참여한다.
	- 각 client는 하루에 몇 번만 update rounds에만 참여할 수 있다.
	- 반면, 각 클라이언트의 데이터셋은 전체 데이터셋에 비해 작기 때문에, 현대 smartphone의 processor(GPU 포함)는 계산 비용을 거의 무시할 수 있을 정도로 빠르다.
- 따라서 모델 학습을 위한 communication rounds 수를 줄이기 위해 추가적인 계산을 사용하는 것이 목표이다. 이를 위한 두 가지 주요 방법이 있다:
	- 1️⃣ Increased parallelism: 각 communication round 간에 더 많은 클라이언트를 독립적으로 작업하게 한다.
	- 2️⃣ Increased computation per client: 각 클라이언트가 간단한 계산(예: gradient 계산)을 수행하는 대신, 더 복잡한 계산을 수행한다.
	-   실험에서는 두 가지 접근 방식을 모두 조사하지만, 클라이언트 간 최소한의 병렬성을 사용할 경우 얻는 속도 향상은 주로 각 클라이언트에서 더 많은 계산을 추가하는 데(2️⃣)서 나온다.

# 1. FedSGD(Stochastic Gradient Descent)
- FL 개념 설명을 위해 나온 baseline 방법(Naïve algorithm)이며, 실제로 잘 쓰이지 않는다. 
- Synchronous update 방식을 가정하여 comminucation rounds로 진행된다.

![image](https://github.com/user-attachments/assets/db3b0527-b63e-4025-aa45-f7d645a1aa4b)

- 학습 과정
	- 1️⃣ Global weight initialization
	- 2️⃣ Client sampling with client fraction hyper-parameter $C$
		- ex. $C=0.75$: client 4개 중 3개 sampling
	- 3️⃣ Local learning
		- Server: global weight을 client로 보낸다.
		- Client: 각자 가진 local data로 새로운 gradient를 계산한다. 
	- 4️⃣ Update parameter
		- Client: 계산된 gradient를 server에 보낸다.
		- Server: 받은 gradients를 가중평균 내어 새롭게 global weight를 update한다. 
	- 5️⃣ 2~4번 반복 
- Hyper-parameter
	- client fraction hyper-parameter $C$


# 2. FedAvg
- FedAVG = FedSGD + mini-batch 개념 
- FedSGD = $B=\infty \land E=1$ 인 FedAVG
- 주요 Hyper-parameter
	- client fraction hyper-parameter $C$
	- mini-batch size $B$
	- epoch $E$
- Synchronous update 방식을 가정하여 comminucation rounds로 진행된다.
- 학습 과정
	- 1️⃣ Global weight initialization
	- 2️⃣ Client sampling with client fraction hyper-parameter $C$
		- ex. $C=0.75$: client 4개 중 3개 sampling
	- 3️⃣ Local learning
		- Server: global weight을 client로 보낸다.
		- Client: 각자 가진 local data로 새로운 gradient를 **mini-batch 단위($B$)**로 계산한다. 모든 mini-batch를 다 돌면 하나의 epoch가 진행된 것이다. 하나의 client당 하나의 weight가 나오게 된다. 
	- 4️⃣ Update parameter
		- Client: 계산된 gradient를 server에 보낸다.
		- Server: 받은 gradients를 가중평균 내어 새롭게 global weight를 update한다. 
	- 5️⃣ 2~4번 반복 
- Algorithm의 pseudo-code는 다음과 같다. 
![image](https://github.com/user-attachments/assets/e18b4d1c-b48f-4702-8c7a-4ea57c8aa7a3)

FedAVG는 communication round를 크게 줄여, 분산 데이터에서 모델을 학습하는 데 드는 시간을 대폭 단축시켰다.
또한 저자들은 averaging process가 마치 dropout과 같은 regularization처럼 작용하는 것으로 추측한다.


## Robust to imbalanced, Non-IID data distribution

일반적인 Non-convex objectives에서는 파라미터 공간에서 모델을 평균화하는 게 모델에 좋지 않다. 하지만 이 논문은 FedAVG를 적용하면 모델을 크게 개선시킬 수 있다고 주장하였다. 저자들은 MNIST 데이터셋을 사용하였다. 


![image](https://github.com/user-attachments/assets/337b53de-d150-4685-9e21-77f585260601)

두 MNIST 숫자 인식 모델 $w$와 $w'$는 각각 MNIST 훈련 세트에서 600개의 서로 다른 IID 샘플로 훈련되었다. 훈련은 SGD 방식으로, fixed learning rate 0.1로 240번의 업데이트를 거쳐 이루어졌다. 각 mini-batch 크기는 50이며, mini-dataset 크기 600에 대해 20번의 패스를 수행했다. 이 정도 훈련이 진행되면, 모델은 각자의 로컬 데이터셋에 과적합되기 시작한다.

- 왼쪽 Figure: 두 모델을 서로 다른 random initialization에서부터 훈련한 후 평균화한 모습. 좋지 않은 결과를 보인다. 
- 오른쪽 Figure(FedAVG): 두 모델을 same random initialization에서 시작하고, 각 모델을 데이터를 서로 다른 부분 집합에 대해 독립적으로 훈련시킨 후 모델을 평균화한 모습

즉, **각 모델의 intialization point가 같아야** 나중에 업데이트되어 나온 weight들을 평균내었을때 이 학습 방식이 잘 작동한다!(IID, Non-IID 상관없이) 이 실험을 통해 불균형적이고 비독립적이지 않은 데이터 분포에도 robust함을 입증하였다.

## Hyper-parameter tuning 순서
- $C \rightarrow B \rightarrow E$
- $E$를 늘리는 것은 $B$를 줄이는 것보다 더 많은 시간을 필요로 함.

## Reference

- [[AISTATS 2017] FedSGD, FedAvg - (1)](https://federated-learning.tistory.com/entry/AISTATS-2017-FedSGD-FedAvg-1)
- 🌟 [[연합학습 기본] FedSGD & FedAVG](https://hello-world-jhyu95.tistory.com/entry/%EC%97%B0%ED%95%A9%ED%95%99%EC%8A%B5-%EA%B8%B0%EB%B3%B8%EA%B0%9C%EB%85%90-3)
- [Federated Learning](https://jaehong-data.tistory.com/79)
