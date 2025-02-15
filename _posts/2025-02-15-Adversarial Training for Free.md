---
title: Adversarial Training for Free!
author: Su
date: 2025-02-15 11:00:00 +0800
categories: [Paper Review]
tags: [CV, Security]
pin: false
use_math: true
---

> NeurIPS 2019. [Paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/7503cfacd12053d309b6bed5c89de212-Paper.pdf) [Github](https://github.com/mahyarnajibi/FreeAdversarialTraining)
Ali Shafahi, Mahyar Najibi, Amin Ghiasi, Zheng Xu, John Dickerson, Christoph Studer, Larry S. Davis, Gavin Taylor, and Tom Goldstein. 


## 🍒 Key Takeaways
+ 하나의 backward pass에서 NN parameter 뿐만 아니라 input image에 대한 loss의 gradient도 계산하여 cost 없이 Adversarial Examples을 생성하였다. 
+ 동일한 input image에 대해 여러 번 update하기 위해, 동일한 minibatch로 연속 $m$ 번 훈련하도록 하였고, 전체 훈련 반복 횟수를 유지하기 위해 전체 epoch 수를 $m$ 으로 나누었다. 


## 1. Introduction

이 논문은 Adversarial Examples Generation 분야를 다룬다. Adversarial Examples로 Neural Network를 학습시키는 것을 *Adversarial Training*이라 한다. 

<aside> 💡
**adversarial examples** 이란? 
<BR> 모델을 의도적으로 속이거나 오분류하게 만들기 위해 설계된 input data. Noise나 Perturbation을 추가하여 만들 수 있다. 이로 인해 모델은 높은 confidence로 잘못된 예측을 하게 된다.
</aside>

### contributions
   + eliminates the overhead cost (기존 연구: high cost)
   + updating model parameters에 쓰이는 gradient information을 image를 변형시킬 때에 재사용한다.
   + 기존 방법과 비슷하거나 약간 더 높은 성능을 보인다. 


## 2. Related Work

기존 연구는 adversarial examples를 생성하는 cost가 너무 크다. gradient computation은 NW parameter 업데이트할 때도 필요하지만, 각 SGD iteration에서 adversarial example generation할 때도 여러 번 쓰인다. 따라서 후자에 쓰이는 # of gradient steps에 따라 slowdown factor가 결정되며, non-robust model보다 3-30배 더 많은 시간이 소요된다. 
Adversarial training · defense 기법들은 너무 time-consuming해 large-scale problems에 적용하기 어렵다. 


### Non-targeted adversarial examples
Adversarial examples에는 두 가지 종류가 있다. 이중 이 논문은 generation과 evaluation 모두에서  non-targeted examples을 사용하였다. 
+ `non-targeted`: image를 특정 class로 이동시킨다.
+ `targeted`: natural class를 벗어나게 한다. 

기존 유명한 non-targeted generation method는 다음과 같다.
+ `Fast Gradient Sign Method (FGSM)`
	+ 한 번의 iteration으로 gradients 부호를 사용한다.
	+ non-iterative attack
+ `Basic Iterative Method(BIM)`: FGSM의 반복 버전
+ `PGD(Projected Gradient Descent)` 공격
	+ a variant of BIM with uniform random noise as initialization
	+ \# of iterations $K$ 가 중요 
	+ In each iteration, 각 이미지에 대한 loss의 gradient를 계산하기 위해 a complete forward and backward pass가 필요하다. 

### Adversarial training
adversarial training의 robustness는 사용된 adversarial examples의 strength에 따라 결정된다. fast non-iterative attack인 FGSM과 Rand+FGSM는 PGD와 같은 iterative attacks에는 효과가 없다. 

+ **natural training**
	+ Inner Loop: 없음 
	+ Outer Loop: only requires $∇_θ l(x, y, θ)$
+ **K-PGD adversarial training algorithm**
	+ min-max formulation of adversarial training
	+ Inner Loop: PGD-K로 adversarial examples 구성
		+ $∇_xl(x_{adv}, y, θ)$
	+ Outer Loop: minibatch SGD로 모델 업데이트
		+  $∇_θl(x_{adv}, y, θ)$
		+ needs roughly $K + 1$ times more computation


## 3. Method

K-PGD adversarial training은 느리기 때문에 대규모 데이터셋에 사용하기는 힘들다. 예를 들어 CIFAR-10에서 WideResNet 7-PGD 훈련은 Titan X GPU에서 약 4일 소요되었다. 

### “Free” adversarial training
+ computes the ascent step by re-using the backward pass needed for the descent step. 
+ 1️⃣ 현재 training minibatch를 NN로 forward pass 
+ 2️⃣ backward pass로 NN parameter에 대한 gradient 계산 
+ 3️⃣ 동일한 backward pass에서 input image에 대한 loss의 gradient도 계산 

하지만 여러 backward pass 없이 동일한 input image에 대한 multiple update가 불가하다는 한계가 있었다. 따라서 저자들은 training 방식을 수정했다. 

동일한 minibatch로 연속 $m$ 번 훈련하도록 하였고, 전체 훈련 반복 횟수를 유지하기 위해 전체 epoch 수를 $m$ 으로 나누었다. 그리고 새 minibatch 가 form 될 때, 이전 minibatch 에서 생성된 perturbation을 새 minibatch의 perturbation 초기값으로 사용했다.

완성된 최종 알고리즘은 다음과 같다. 

<img width="634" alt="Image" src="https://github.com/user-attachments/assets/03f2b16a-46d7-4ed2-9d9c-db284a1c19fc" />

## 🍋 After Read
+ image를 넘어 일반 tabular data에도 잘 적용될지 궁금하다. 
+ backward pass의 gradient를 재사용하여 다른 기능을 추가할 수 있다는 것을 깨달았다. 


## Reference
+ https://velog.io/@seoyeon/Pytorch-tutorial-%EC%A0%81%EB%8C%80%EC%A0%81-%EC%98%88%EC%A0%9C-%EC%83%9D%EC%84%B1ADVERSARIAL-EXAMPLE-GENERATION