---
title: InstructTTS: Modelling Expressive TTS in Discrete Latent Space with Natural Language Style Prompt
author: Su
date: 2024-12-02 15:00:00 +0800
categories: [Paper Review]
tags: [Audio]
pin: false
use_math: true
---

> TASLP 2023 [Paper](https://arxiv.org/pdf/2301.13662) [Page](https://dongchaoyang.top/InstructTTS/)
> Dongchao Yang*, Songxiang Liu*, Rongjie Huang, Chao Weng, Helen Meng, Fellow, IEEE


*%% VQ, VAE 복습하고 다시 읽기*

## 🍒 Key Takeaways
1️⃣ SSL과 cross-modal metric learning을 사용하여 robust sentence embedding model을 얻기 위해 3단계 훈련 과정을 제시하였다.
2️⃣ 일반적인 mel-spectrogram 대신 vector-quantized acoustic token을 사용하기 위해 discrete latent space를 모델링하고 discrete diffusion probabilistic model을 활용하였다.
3️⃣ Style-speaker와 style-content의 mutual information minimization을 통해 style prompt에서 content와 speaker의 information leakage를 방지하였다. 


# Overview
![image](https://github.com/user-attachments/assets/1bdc52d8-9183-4567-a535-66dfec955add)
- 1️⃣ **Content Encoder**: content prompt에서 content representation을 추출
- 2️⃣ **Style Encoder**: Style prompt에서 style representation을 추출
- 3️⃣ **Speaker Embedding Module**
- 4️⃣ **Style-Adaptive Layer Normalization (SALN) Adaptor**
- 5️⃣ **Discrete Diffusion Decoder**

# InstructTTS
- **ExpressiveTTS**: 자연어 prompt를 사용한 expressive TTS 모델링을 처음으로 연구하여 사용자 제어가 가능한 expressive TTS를 실현하는 데 한 걸음 더 나아갔음.  

![image](https://github.com/user-attachments/assets/eb9b0dea-d7a8-43e1-ae1a-a3a2d8b13c7b)

-   **Sentence Embedding Model Training**:  
    스타일 prompt에서 의미 정보를 효과적으로 캡처할 수 있도록 하기 위해, 3단계 훈련 전략을 도입하여 sentence embedding model을 훈련함.  
    또한, speech synthesis를 sequence-to-sequence language modeling task로 보고, discrete latent space에서 acoustic features를 모델링. 이를 위해 novel discrete diffusion model을 훈련하여 VQ acoustic features를 생성함.
-   **Types of VQ Acoustic Features**:  
    두 가지 유형의 VQ acoustic features를 모델링함:
    -   Mel-spectrogram 기반 VQ features
    -   Waveform 기반 VQ features
-   **Effective Modeling with Discrete Diffusion Model**:  
    두 가지 VQ features는 모두 discrete diffusion model을 사용하여 효과적으로 모델링 가능. Waveform 기반 모델링은 한 번의 훈련 단계로 이루어지며, non-autoregressive 모델로 VALL-E와 MusicLM과 차별화됨.
    
-   **Mutual Information (MI) Estimation & Minimization**:  
    Acoustic model 훈련 중에 MI 추정 및 최소화를 적용하여 style-speaker와 style-content 간의 MI를 최소화. 이를 통해 스타일 prompt로부터 콘텐츠 및 화자 정보 유출을 방지하고, 보다 정확한 스타일 제어를 달성함.

## Algorithm Pseudo-code

- Training phase
![image](https://github.com/user-attachments/assets/a7909f69-a233-40d5-b735-5030ebbf2471)

- Inference phase
![image](https://github.com/user-attachments/assets/c7f09b1e-e6a5-438a-b16f-f1a7fb471ec0)


# Method
![image](https://github.com/user-attachments/assets/1bdc52d8-9183-4567-a535-66dfec955add)

Overview에서 언급했던 다섯 가지 구성 요소를 하나씩 살펴 보자. 

## A. Content Encoder
-   Goal: Content Prompt에서 내용 표현 추출
-   FastSpeech2 기반
    -   4개의 Feed-Forward Transformer 사용
    -   FFT block의 hyper-parameter:
        -   Hidden size: 256
        -   Attention heads: 2
        -   Kernel size: 9
        -   Filter size: 1024
    -   Variance Adaptor 사용: Duration, Pitch 예측 (발화 스타일과 관련)

## B. Style Prompt Embedding Model
-  Goal: style prompt에서 스타일 표현 추출
-  RoBERTa 모델 사용
    -   Style prompt sequence: $S = [S_1, S_2, ..., S_M]$ ($M$은 시퀀스 길이)
    -   `[CLS]` 토큰을 시퀀스 앞에 추가 후 RoBERTa에 입력
    -   style representation: `[CLS]` 토큰의 표현을 스타일 표현으로 사용
-   Importance of style prompt embedding quality:
	- 1️⃣ style prompt 공간은 중요한 의미 정보를 포함해야 함
	- 2️⃣ prompt embedding 공간은 균일하고 부드러워야 하며(relatively uniform and smooth), 훈련에 보지 못한 스타일 설명에도 일반화 가능해야 함

### Training 과정
   - 1️⃣ 기본 언어 모델 학습 (중국어)
       -   기존 영어 기반 모델을 중국어 데이터로 학습
   -  2️⃣ labeled data로 model fine-tuning
       -   적은 양의 자연어 추론(NLI) 데이터로 학습
   -  3️⃣ style prompt와 음성 간의 교차 모드 표현 학습
       -   style prompt vector와 음성에서 추출한 스타일 표현 벡터를 공유된 의미 공간으로 mapping
       -   교차 모드 표현 학습: Style-Prompt와 Audio Pair 기반의 오디오-텍스트 검색 작업 수행
       -   Metric Learning 사용, InfoNCE Loss 적용하여 모델 학습


## C. Style Encoder
-   구성 요소:
	- Prompt Encoder: Pretrained된 Robust Style Prompt Embedding Model
	- Adaptor Layer: Style embedding을 새로운 잠재 공간으로 매핑
	-  Audio Encoder: Mel-Spectrogram에서 스타일 정보 추출
-   Training:
    -   Style Prompt embedding과 Audio embedding 간의 거리 최소화
    -   스타일-화자(MI) 및 스타일-내용(MI) 상호 정보(MI) 최소화
    -   목표: Audio Encoder가 스타일과 관련된 정보만 인코딩하도록 학습
-   Mutual Information(MI):
    -   MI: 무작위 변수 간의 상관 관계 측정
    -   고차원 확률 변수의 MI 계산은 어려움
    -   MINE과 InfoNCE는 MI 하한을 추정, CLUB은 MI 상한을 추정
    -   본 연구에서는 CLUB 방법을 사용하여 스타일-화자 및 스타일-내용 간의 MI 최소화


## D. modeling Mel-spectrograms in Discrete Latent Space

- 저자들은 Mel-spectrogram을 discrete latent space에서 모델링하는 것이 expressive TTS에 적합하다고 주장하였다. 
- 따라서 저자들은 VQ-VAE를 중간 표현으로 사용하여 mel-spectrogram 모델링을 도와주며, non-autoregressive mel-spectrogram 토큰 생성 모델을 제안하였다.
	- discrete diffusion models 기반
- 기존 TTS의 문제점
	- 대부분의 TTS : 텍스트에서 mel-spectrogram을 연속적인 공간에서 직접 학습하고, 예측된 mel-spectrogram을 HiFiGAN vocoder로 복원하여 waveform을 생성
	-   하지만 mel-spectrogram의 주파수 bin은 시간과 주파수 축을 따라 복잡하게 상관관계를 가지므로, 특히 감정과 말하는 스타일이 강한 음성을 생성할 때 모델링이 어려움
	-   Ground-truth mel-spectrogram과 예측된 mel-spectrogram 간의 차이가 합성 성능에 영향을 미침

- Mel-spectrogram을 여전히 discrete latent space에서 모델링하지만 HiFiGAN vocoder를 사용하여 mel-spectrogram에서 waveform을 복원.
	- 1️⃣ 대규모 음성 데이터셋으로 VQ-VAE를 사전 훈련시켜 Mel-VQ-VAE가 언어, pitch, 에너지, 감정 정보를 latent code에 인코딩
	- 2️⃣ vector quantized latent codes를 예측하는 것을 목표로 삼고 mel-spectrogram을 discrete latent space에서 모델링

### Mel-VQ-VAE
![image](https://github.com/user-attachments/assets/50a65057-c5ce-4037-88e0-87f93e963ef1)

- Mel-VQ-VAE는 3개의 부분으로 구성됨:
	- 1️⃣ Mel-encoder $E_{mel}$
	- 2️⃣ Mel-decoder $G_{mel}$
	- 3️⃣ codebook $Z = \{z_k\}_{k=1}^{K} \in \mathbb{R}^{K \times n_z}$ \
		- Codebook size: $K$
		- code dim: $n_z$
- 입력으로 mel-spectrogram $s \in \mathbb{R}^{F_{bin} \times T_{bin}}$을 받으면, - mel-spectrogram은 latent representation $\hat{z} = E_{mel}(s) \in \mathbb{R}^{F'_{bin} \times T'_{bin} \times n_z}$로 인코딩됨.
- $F'_{bin} \times T'_{bin}$은 축소된 주파수 및 시간 차원 - Quantizer $Q(.)$를 사용하여 각 특징 $\hat{z}_{ij}$을 가장 가까운 codebook 항목 $z_k$에 매핑하여 discrete spectrogram 토큰 $z_q$ 생성
$$ z_q = Q(\hat{z}) := \arg \min_{z_k \in Z} ||\hat{z}_{ij} - z_k||_2^2 $$
- 이후 decoder를 사용하여 mel-spectrogram을 재구성: $\hat{s} = G_{mel}(z_q)$
- 성능 향상을 위해 VQGAN을 참고하여 adversarial loss를 훈련 단계에 추가


### Mel-VQ-Diffusion Decoder
- Mel-VQ-VAE의 도움을 받아 mel-spectrogram 예측 문제를 quantization tokens 예측 문제로 전환
- 고품질의 mel-spectrogram 토큰을 빠른 추론 속도로 생성하기 위해 Mel-VQ-Diffusion decoder를 제안

#### 아이디어 소개
- 훈련 데이터 $(x_0, y)$에서 $y$는 phone features, style features, speaker features의 조합.
	- $x_0$는 ground truth mel-spectrogram tokens
- 확산 과정(Diffusion process)을 구축하여 $p(x_0)$의 분포를 제어 가능한 stationary 분포 $p(x_T)$로 변환 
- Transformer 기반 신경망을 사용하여 $p(x_0)$를 조건부로 복원하기 학
- 전이 행렬 $Q_t \in \mathbb{R}^{(K+1) \times (K+1)}$는 다음과 같이 정의됨: $$ Q_t = \begin{bmatrix} \alpha_t + \beta_t & \beta_t & \beta_t & \beta_t & \cdots & 0 \\ \beta_t & \alpha_t + \beta_t & \beta_t & \beta_t & \cdots & 0 \\ \beta_t & \beta_t & \alpha_t + \beta_t & \beta_t & \cdots & 0 \\ \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\ \gamma_t & \gamma_t & \gamma_t & \gamma_t & \cdots & 1 \end{bmatrix} $$
- 확산 과정은 $q(x_t | x_0)$를 다음과 같이 계산: $$ Q_t c(x_0) = \alpha_t c(x_0) + (\gamma_t - \beta_t) c(K+1) + \beta_t $$
- Stationary distribution $p(x_T)$는: $$ p(x_T) = [\beta_T, \beta_T, \cdots, \gamma_T] $$
- Decoder Training Target
	- Goal: 네트워크 $p_{\theta}(x_{t-1} | x_t, y)$를 학습하여 $q(x_{t-1} | x_t, x_0)$의 posterior 전이 분포를 추정 
	- variational lower bound(VLB)을 최소화하는 방식으로 훈련: $$ L_{diff} = \sum_{t=1}^{T-1} \left[ D_{KL}[q(x_{t-1} | x_t, x_0) || p_{\theta}(x_{t-1} | x_t, y)] + D_{KL}[q(x_T | x_0) || p(x_T)] \right] $$ 

- Enhancing the Connection Between $x_0$ and $y$ 
	- 네트워크가 훈련 마지막 단계에서 조건부 정보 $y$를 무시하는 문제를 해결하기 위해 classifier-free guidance 도입 - 목표 함수 최적화: $$ \log(p(x|y)) + \lambda \log(p(y|x)) $$
	- $\lambda$: posterior 제약의 강도를 제어하는 하이퍼파라미터 - Bayes's theorem을 사용하여 최적화된 목표 함수는: $$ \arg \max_x \left[ \log p(x) + (\lambda + 1)(\log p(x|y) - \log p(x)) \right] $$ 
	- 무조건적인 mel-spectrogram token 예측을 위해, 훈련 시 10% 확률로 null vector $n$을 사용.
	- 추론 단계에서는 조건부 mel-spectrogram 토큰의 logits $p_{\theta}(x_{t-1} | x_t, y)$을 먼저 생성한 뒤, 무조건적인 mel-spectrogram 토큰의 logits $p_{\theta}(x_{t-1} | x_t, n)$을 예측
	- 최종 샘플링 확률은 다음과 같이 계산: $$ p_{\theta}(x_{t-1} | x_t, n) + (\lambda + 1)(p_{\theta}(x_{t-1} | x_t, y) - p_{\theta}(x_{t-1} | x_t, n)) $$


## E. Modelling Waveform in Discrete Latent Space Via Multiple Vector Quantizers

### Neural Audio Codec Models
-   대규모로 사전 훈련된 neural codec models을 활용하여 discrete latent space에서 직접 waveform을 예측하는 방법 연구
-   최근에는 AudioLM과 같은 방법들이 제안됨. AudioLM은 자기 지도 학습 모델의 k-means 토큰과 신경망 코덱 모델의 음향 토큰을 사용하여 음성-음성 변환 언어 모델을 훈련시켜, 고품질의 음성-음성 생성 수행
-   VALL-E는 텍스트 입력과 참조 오디오를 기반으로 음성을 합성하기 위해 두 단계 모델을 훈련시킴. 하지만 VALL-E는 두 단계 훈련 전략을 요구하며, 첫 번째 단계는 autoregressive language model로, 합성 속도에 큰 영향을 미침
-   이 논문에서는 discrete diffusion model을 기반으로 한 non-autoregressive model을 제안하여 합성 속도를 크게 개선하면서 고품질 합성 성능을 유지하였다.
    

### Neural Audio Codec Model vs  Mel-VQ-VAE
![image](https://github.com/user-attachments/assets/87b9774e-c73f-42d7-9e4b-caaf32db3a90)

-   Neural Audio Codec 모델**은 Mel-VQ-VAE보다 더 많은 codebook을 포함한다.
	-   더 많은 codebook을 사용하면 재구성 성능이 향상되지만 긴 시퀀스를 Transformer로 모델링하는 과정에서 메모리 문제를 발생시킨다.
	- ex. 10초 길이의 음성을 24k 샘플링 속도로 처리하면 8개의 codebook을 사용하고, 인코더에서 240배 다운샘플링을 설정하면 8000개의 토큰이 발생

###  U-Transformer Architecture:

![image](https://github.com/user-attachments/assets/897dc597-85c0-4e0c-98c1-833941b4f692)

-   저자들은 여러 codebook을 동시에 모델링하기 위해 U-Transformer를 제안하였다. 
- 1️⃣ 여러 convolution 레이어를 사용하여 입력 codebook 행렬을 codebook 차원에서 다운샘플링.
- 2️⃣ denoising transformer로 잠재 공간에서 토큰 간의 관계를 모델링
- 3️⃣ 그 후 여러 convolution 및 upsampling 레이어를 사용하여 codebook 차원을 복원
- 4️⃣ 각 codebook에 대해 예측 결과를 동시에 출력하는 다양한 output layer 사용


### Wave-VQ-Diffusion:
- Mel-VQ-Diffusion과의 세 가지 차이점
	- 1️⃣ 여러 codebook을 동시에 모델링하기 위해 U-transformer architecture 채택
		- Mel-VQ-Diffusion과 동일한 Transformer 아키텍처 사용
	- 2️⃣ 서로 다른 codebook에 대해 서로 다른 embedding table을 사용
		- 서로 다른 codebook에서 나온 토큰들이 다른 데이터 분포를 따르기 때문이다
	- 3️⃣확산 과정에서 개선된 mask 및 uniform 전략
        -   첫 번째 residual vector quantization layer에서 마지막 layer로 갈수록 codebook에 포함된 정보가 점차 줄어듦
        -   첫 번째 layer의 codebook은 텍스트, 스타일, 화자 정보 대부분을 포함하고, 이후 layers는 주로 미세한 음향 세부 정보를 포함
        -   첫 번째 layer의 codebook의 토큰은 $y$를 조건으로 쉽게 복원 가능하고 이후 layers의 토큰은 $y$와의 명확한 연결이 없기 때문에 복원하기 어려움
        -  easy-first-generation 원칙에 따라, 전방 과정에서 마지막 layer의 codebook(예: codebook Nq)을 마스크하고, 후방 과정에서 첫 번째 layer의 codebook(ex. codebook 1)을 마스크
        -   기존의 마스크 및 uniform 전략은 시퀀스의 모든 토큰이 동일한 중요도를 가진다고 가정했으나, 이는 easy-first-generation 원칙에 어긋남
        -   이를 해결하기 위해 개선된 마스크 및 uniform 전략 제안





## Reference

- [[Paper 리뷰] InstructTTS: Modelling Expressive TTS in Discrete Latent Space with Natural Language Style Prompt](https://randomsampling.tistory.com/198)
