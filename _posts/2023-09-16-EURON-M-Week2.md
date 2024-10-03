---
title: Deep Learning Specialization 1-3 | Basics of Neural Network Programming
author: Su
date: 2023-09-16 05:55:00 +0800
categories: [DL]
tags: [EURON]
pin: false
use_math: true

---

Learning Source
+ [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning?utm_source=deeplearningai&utm_medium=institutions&utm_campaign=SocialYoutubeDLSC1W1L1#courses)
+ [부스트코스 딥러닝 1단계: 신경망과 딥러닝](https://m.boostcourse.org/ai215/lectures/86249)

# **파이썬과 벡터화**
## Vectorization
### What is vectorization?
+ $z=w^Tx +b$ 을 구해보자.<Br>
  $w= \begin{bmatrix}
      w_1 \\
      w_2 \\
      ... \\
      w_{n_x} \\ \end{bmatrix}$ (열벡터) 
  <br>

  $x= \begin{bmatrix}
      x_1 \\
      x_2 \\
      ... \\
      x_{n_x} \\  \end{bmatrix}$ (열벡터)
+ 이때 $w \in \mathbb{R}^{n_x}, x \in \mathbb{R}^{n_x}$
+ 1️⃣ **Non-vetorized**
  ```python
    z=0
    for i in range(n_x):
      z+= w[i]*x[i]
    z+=b
  ```
+ 2️⃣ **Vectorized**
  ```python
    z = np.dot(w, x) + b
  ```
+ <code>SIMD</code>(Single Instruction Multiple Data)
  + 병렬 프로세서의 한 종류로, 하나의 명령어로 여러 개의 값을 동시에 계산하는 방식
  + 벡터화 연산을 가능하게 한다.
  + for문으로 하나의 값을 연산하는 것보다 vectorized하여 한 번에 연산하는 것이 더 빠르고 효율적이다.
  ```python
    import numpy as np
    a = np.array([1, 2, 3, 4])
    print(a) # [1, 2, 3, 4]
  ```
  ```python
    import time
    a = np.random.rand(100000)
    b = np.random.rand(100000)

    tic = time.time()
    c = np.dot(a, b)
    toc = time.time()

    print(c)
    print("Vectorized version: " + str(1000*(toc-tic)) + "ms")
    # 24960.839226052667
    # Vectorized version: 1.783132553100586ms

    c=0
    tic = time.time()
    for i in range(100000):
      c += a[i]*b[i]
    toc = time.time()

    print(c)
    print("For loop:" + str(1000*(toc-tic))+"ms")
    # 24960.839226052893
    # For loop:145.95532417297363ms
  ```

## More Vectorization Examples
+ Whenever possible, avoid explicit for-loops
  + 1️⃣ **Non-vetorized**
    + $ u = Av$
    + $u_i = \sum_i \sum_j A_{ij}v_{ij}$
    + $u = \operatorname{np.zeros}((n, 1))$
  ```python
  u = np.zeros((n, 1))
  for i ...
    for j ...
      u[i] += A[i][j] * v[j]
  ```
  + 2️⃣ **Vectorized**
    ```python
      u = np.dot(A, v)
    ```
+ Vectors and matrix valued functions
  + Say you need to apply the exponential operation on every element of a matrix/vector. <br>

  $ v = \begin{bmatrix} v_1 \\ ... \\ v_n \\ \end{bmatrix} \to u = \begin{bmatrix} e^{v_1} \\ ... \\ e^{v_n} \\ \end{bmatrix}$ (둘다 열벡터)  
  

  ```python
      u = np.zeros((n, 1))
      for i in range(n):
        u[i]=math.exp(v[i])
    ```
    ```python
    import numpy as np
    u = np.exp(v)
    u = np.log(v)
    u = np.abs(v)
    np.maximum(v, k)

    np.maximum(np.eye(2), [0.5, 2]) # broadcasting
    '''
    array([[ 1. ,  2. ],
        [ 0.5,  2. ]])
    '''
  ```

+ Logistic regression derivatives
  + $dw$ 를 vectorized 시킴으로써 가장 안쪽의 for문은 제거했지만, 여전히 가장 바깥의 for문이 남아 있다.<br>
  <img width="550" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/16ab9414-2968-4530-9798-d912bc3f3c02">

  + 자주 쓰는 넘파이(numpy) 함수:
    + `log`
    + `abs`
    + `maximum`
    + `**`
    + `zeros`


## Vectorizing Logistic Regression
+ 아래의 식은 for문을 이용해 i의 값을 변화시키며 계산해야 한다.
  + $z^{(i)} = W^Tx^{(i)}+b$
  + $a^{(i)} = \sigma(z^{(i)})$
+ 하지만 계산의 효율성을 증가시키기 위해 벡터를 이용하면 다음과 같이 계산할 수 있습니다.
  + `Z = np.dot(np.transpose(W), X) + b`
+ 위의 코드에서 $(1,m)$ 크기의 matrix와 상수 $b$ 를 더하기에 오류가 날 것 같지만, 파이썬이 자동적으로 상수를 $(1,m)$ 크기의 행렬로 broadcasting 해주기 때문에 발생하지 않는다.
- $ X = \begin{bmatrix} \mid & \mid &  & \mid \\ X^{(1)} & X^{(2)} & ... & X^{(m)} \\ \mid & \mid &  & \mid \\ \end{bmatrix} $
+ $Z = [Z^{(1)}, Z^{(2)}, ..., Z^{(m)}]=w^TX+[b, b, ..., b] \\ = w^TX^{(1)}+b, w^TX^{(2)}+b, ..., w^TX^{(m)}+b$
+ 즉, `Z = np.dot(W.T, X) + b`이다(이때 $b$는 실수)
+ $A=[a^{(1)}, a^{(2)}, ..., a^{(m)}] = \sigma(Z)$


## Vecotorizing Logistic Regression's Gradient Computation

### Vectorizing Logistic Regression
+ 이미 겉의 for문을 제거해서 $dw$ 는 벡터이므로 $dw_1, dw_2, ..., dw_3$ 를 각각 계산하지 않는다.
<br>

  $A=[a^{(1)}, a^{(2)}, ..., a^{(m)}], Y = [y^{(1)}, y^{(2)}, ..., y^{(m)}]$ <BR>
  $dz =[dz^{(1)}, ..., dz^{(m)}] = A-Y $<BR>
  $ dz^{(1)} = a^{(1)} - y^{(1)}, ..., dz^{(m)} = a^{(m)} - y^{(m)} $

+ 1️⃣ **Non-vetorized**
  + $dw$ 구하기<br>
    $ dw = 0$ <br>
    $dw += X^{(1)}dz^{(1)}$<br>
    $dw += X^{(2)}dz^{(2)}$<br>
    $...$<br>
    $dw += X^{(m)}dz^{(m)}$<br>
    $ dw /= m$
  + $db$ 구하기<br>
    $ db = 0$ <br>
    $db += dz^{(1)}$<br>
    $db += dz^{(2)}$<br>
    $...$<br>
    $db += dz^{(m)}$<br>
    $ db /= m$
+ 2️⃣ **Vectorized**
  + $dw$ 구하기<br>
    $dw = \frac{1}{m} Xdz^T$
    $  =\frac{1}{m}$
  + $db$ 구하기 <br>
    $db = \frac{1}{m} \sum_{i=1}^{m} z^{(i)}=\frac{1}{m} \operatorname{np.sum}(dz)$ <br>
    $=\frac{1}{m} \begin{bmatrix} \mid & \mid & 0 & \mid \\ X^{(1)} & X^{(2)} & ... & X^{(m)} \\ \mid & \mid & 0 & \mid \\  \end{bmatrix} \begin{bmatrix} dz^{(1)} \\  ... \\ dz^{(m)} \\  \end{bmatrix} $
+ $dz$ 는 column vector이다.

### Implementing Logistic Regression
+ 1️⃣ **Non-vetorized** 
  $ J=0, dw_1=0, dw_2=0, db=0$<br>
  $ \operatorname{for} i = 1 \operatorname{to} m:
      \\ Z^{(i)} = w^Tx^{(i)} + b
      \\ a^{(i)} = \sigma(z^{(i)})
      \\ J += -[y^{(i)}\log a^{(i)} + (1-y^{(i)})\log (1-a^{(i)})]
      \\ dz^{(i)} = a^{(i)} - y^{(i)}
      \\ dw_1 += {x_1}^{(i)}dz^{(i)}
      \\ dw_2 += {x_2}^{(i)}dz^{(i)}
      \\ ...
      \\ dw_{n_x} += {x_{n_x}}^{(i)}dz^{(i)}
      \\ db += dz^{(i)}
    \\ J = J / m, dw_1 = dw_1 / m, dw_2 = dw_2 / m
    \\ db = db/m $
+ 2️⃣ **Vectorized**
  for iter in range(1000):
  $ Z = w^TX + b
    \\  = \operatorname{np.dot}(w.T, X) + b
    \\ A = \sigma(Z)
    \\ dZ = A - Y
    \\ dw = \frac{1}{m}XdZ^T
    \\ db = \frac{1}{m} \operatorname{np.sum}(dz)
    \\ w := w - \alpha dw
    \\ b := b - \alpha db$

## Broadcasting in Python

### numpy.random 관련 정리
+ <code>np.random.rand()</code>: 0~1 사이 균일 분포 추출 함수
  + 괄호 안을 비워두면 값 1개가 추출된다
  + 괄호 내에 dimension을 적으면, 해당 dimension의 numpy array가 생성된다
  ```python
    np.random.rand() # 0.6320642827185263
    np.random.rand(3, 2)
    '''
    array([[0.59950361, 0.81513226],
        [0.0775172 , 0.33192424],
        [0.2379349 , 0.84078282]])
    '''
  ```
+ <code>np.random.random()</code>: 0~1 사이 균일 분포 추출 함수
  + rand 함수와 거의 동일하나, 원하는 차원의 형태를 튜플 자료형으로 넣어주어야 한다
  + 괄호 안을 비워두면 값 1개가 추출된다
  + 괄호 내에 dimension을 적으면, 해당 dimension의 numpy array가 생성된다
  ```python
    np.random.random() # 0.7256115181737749
    np.random.random((3, 2))
    '''
    array([[0.21562737, 0.69082549],
        [0.81584948, 0.61564714],
        [0.85217779, 0.70106302]])
    '''
  ```
+ <code>np.random.randn</code>: 표준정규분포 추출 함수
  + 평균 0, 표준편차 1을 가지는 표준정규분포 내에서 임의 추출한다
  ```python
  np.random.randn() # -0.13049944410660697
  np.random.randn(4) # array([ 0.68319741, -1.37804659,  1.25332479, -1.21334304])
  np.random.randn(4, 1)
  np.random.randn(3, 2)
  '''
  array([[-0.2243001 ],
       [ 0.03057147],
       [-1.26080828],
       [-1.94113049]])
  array([[ 0.39715935,  0.27487829],
       [-0.9506768 , -0.0807032 ],
       [-0.97761767,  0.92817657]])
  '''
  ```
+ <code>randint()</code>: 정수 임의 추출 함수
  + 인자: (lowerbound, highbound, size)
  + lowerbound ~ highbound-1 사이 범위에서 정수를 임의 추출한다
  ```python
    np.random.randint(-2, 2, 3) # array([-1, -2, -2])
    np.random.randint(1, 8, 5) # array([3, 5, 4, 5, 3])
  ```


### 파이썬 행렬 곱연산 정리
+ 1️⃣ 행렬 A, B가 `np.array()` 형태일 때
  + 별연산(`*`): 스칼라곱
    + (n,m)*(n,m) = (n,m)
    + 브로드캐스팅시
      (1,m)*(n,m) = (n,m)
      or (m,1)*(m,n) = (m,n)
      or (m,n)*(m,1) = (m,n)
      or (n,m)*(1,m) = (n,m)
  + 내적(`dot`): 선형대수의 내적과 같다
    + (n,m).dot((m,k)) = (n,k)
  + 행렬곱(`@`): 행렬의 곱셈
    + 단, 2차원 공간(행렬)에서는 내적과 같은 역할을 한다
    + 3차원부터는 tensor 곱(외적)을 하게 된다
    + (n,m)@(m,k) = (n,k)
+ 2️⃣ 행렬 A, B가 `np.matrix()` 형태일 때



### Broadcasting
+ <code>Broadcasting</code>: an operation of matching the dimensions of differently shaped arrays in order to be able to perform further operations on those arrays
+ 차원의 크기가 1이거나, 차원에 대해 축의 길이가 동일할 때 가능하다
+ code execution time을 줄일 수 있는 기법이다
+ dimension이 auto expanded 된다
<img src="https://github.com/sml09181/sml09181.github.io/assets/105408672/887eab24-ed9b-463e-a761-e191c9f7aa9e">

### Broadcasting Examples
+ Calories from Carbs, Proteins, Fats in 100g of different foods<br>
  <img width="427" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/452858cf-b559-421b-9adc-3cddbbdaec9e">
  ```python
    A = np.array([[56.0, 0.0, 4.4, 68.0],
                [1.2, 104.0, 52.0, 8.0],
                [1.8, 135.0, 99.0, 0.9]])
    cal = A.sum(axis=0) 
    print(cal) # [ 59.  239.  155.4  76.9]
    print(cal.shape) # (4,)
    percentage = 100*A/cal.reshape(1,4)
    print(percentage)
    '''
    [[94.91525424  0.          2.83140283 88.42652796]
    [ 2.03389831 43.51464435 33.46203346 10.40312094]
    [ 3.05084746 56.48535565 63.70656371  1.17035111]]
    '''
  ```
+ 덧셈<br>
  <img width="398" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/2338425f-0ccf-4f7b-bbcf-6595d3f49682">
+ General Principle<br>
  <img width="400" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/fdf5707f-f5a8-49b9-8489-ddbf149a4835">
+ 퀴즈
  + 퀴즈1
  ```python
    a = np.random.randn(2, 3) # a.shape = (2, 3)
    b = np.random.randn(2, 1) # b.shape = (2, 1)
    c = a * b
    print(c)
    print(c.shape)
    '''
    [[-0.24956183  0.31327135  1.00597402]
    [ 1.13608675 -1.01161003  0.14922026]]
    (2, 3)
    '''
  ```
  + 퀴즈2
  ```python
    a = np.random.randn(4, 3)
    b = np.random.randn(3, 2)
    c = a * b
    '''
    사이즈가 매치되지 않기 때문에 에러가 난다.
    ValueError: operands could not be broadcast together with shapes (4,3) (3,2) 
    '''
  ```
  + 퀴즈3
  ```python
    a = np.random.randn(3, 3)
    b = np.random.randn(3, 1)
    c = a * b
    print(c)
    print(c.shape)
    '''
    [[ 1.62479277  1.82880562  0.94845195]
    [ 0.01090171 -0.26786701  0.3190328 ]
    [-0.40082135 -0.24469142  0.55287189]]
    (3, 3)
    '''
  ```
  + 퀴즈4
  ```python
    a = np.random.randn(2, 3)
    print(a)
    print(a.shape)
    b = np.random.randn(2, 1)
    print(b)
    print(b.shape)
    c = a + b
    print(c)
    print(c.shape)
    '''
    [[-1.40759913  0.70449155 -0.53921712]
    [-0.03529759  0.25307406  2.94068672]]
    (2, 3)
    [[-0.01688486]
    [-0.19458319]]
    (2, 1)
    [[-1.42448398  0.68760669 -0.55610198]
    [-0.22988078  0.05849087  2.74610353]]
    (2, 3)
    '''
  ```

## A Note on Python
+ 결론: <mark style='background-color: #f5f0ff'>🌟Rank가 1인 형태를 쓰지 말자🌟</mark>
  + `(n, )` 같이 Rank 1인 형태의 자료구조는 행벡터도 아니고 열벡터도 아니므로, 직관적이지 않은 결과를 도출한다
+ <code>shape</code>: 행렬의 형태를 볼 수 있다(행, 열)
+ <code>assert</code>: 해당 코드 부분에서 어떤 조건이 참임을 확고히 하는 것
  + 형식: `assert [조건식], [메시지]`
  + 조건문이 일치하지 않으면 [메시지]를 담아 AssertionError 발생시킨다
  + 행렬과 배열의 차원을 확인할 때 사용하자
  + rank1를 얻게 되면 <code>reshape</code>으로 행벡터 또는 열벡터로 바꿔주자
  + Examples
    ```python
      assert image.shape == [3, 224, 224]
      assert image.shape[0] == 3
      assert array1.shape[0] == array2.shape[0]
      assert len(array2.shape) == 3
    ```
    ```python
    # 출처: https://hbase.tistory.com/398
    def test(age):
      assert type(age) is int, 'age 값은 정수만 가능'
      assert age > 0, 'age 값은 양수만 가능'

    age = 1
    test(age)

    age = -10
    test(age)

    # Traceback (most recent call last):
    #   File "./test.py", line 12, in <module>
    #     test(age)
    #   File "./test.py", line 6, in test
    #     assert age > 0, 'age 값은 양수만 가능'
    # AssertionError: age 값은 양수만 가능
    ```


## Explanation of Logistic Regression Cost Function

### Logistic regression cost function
+ 1️⃣ Logistic Regression에서 배운 손실함수
  + $y$ 값이 1이 될 확률: $P(y=1 \mid x) = \hat y$
  + $y$ 값이 0이 될 확률: $P(y=0 \mid x) = 1- \hat y$
+ 2️⃣ 위 두 등식을 하나의 수식으로 합쳐보자
  + $P(y \mid x) = \hat y^y(1-\hat y)^{(1-y)}$
  + if $y=1$: $P(y \mid x) = \hat y^1(1-\hat y)^0$
  + if $y=0$: $P(y \mid x) = \hat y^0(1-\hat y)^1$
+ 3️⃣ 로그함수는 단조 증가 함수이므로 위 식은 아래와 동일하다<br>
  $\log P(y \mid x) = \log(\hat y^y(1-\hat y)^{(1-y)})
  \\ = y\log \hat y + (1-y)\log(1-\hat y) $
+ 4️⃣ Our Goal: 확률 $\log P(y \mid x)$ 을 최대화시키는 것 ▶️ 손실함수 $-\log P(y \mid x)$ 을 최소화시키자
+ 5️⃣ training sample 하나의 손실함수는 다음과 같이 정의된다
  $ L(\hat y, y) = -\log P(y \mid x) = y\log \hat y + (1-y)\log(1-\hat y) $


### Cost on m examples
+ training samples이 `IID`(independently and identically distributed)라 가정한다
  + `IID`: 서로 독립이고 각각 동일한 확률분포를 따르는 다차원 확률변수
  + 동일한 조건 아래에서 수행되는 실험이나 관측을 여러 번 반복하여 데이터를 얻는 것
+ scale을 맞추기 위해 `m`으로 나눠준다
+ Cost: $J(w, b)= \frac{1}{m} \sum_{i=1}^{m} L(\hat y^{(i)}, y^{(i)})$
<img width="629" alt="image" src="https://github.com/sml09181/sml09181.github.io/assets/105408672/386c5802-6084-4f31-806c-9fb6f093c75a">

<br><br>

Latex로 열벡터 입력하는 거 너무 어렵다..

<br><br>

Source:
+ [🌟 넘파이 랜덤 추출 함수 정리 : rand, random, randn, randint, choice, seed](https://jimmy-ai.tistory.com/60)
+ [🌟 Python assert 사용법 및 예제](https://hbase.tistory.com/398)
+ [Assertion of arbitrary array shapes in Python](https://medium.com/@nearlydaniel/assertion-of-arbitrary-array-shapes-in-python-3c96f6b7ccb4)
+ [9-1. 독립동일분포(독립성, 합의 분포)](https://data-science-note.tistory.com/entry/9-1-%EB%8F%85%EB%A6%BD%EB%8F%99%EC%9D%BC%EB%B6%84%ED%8F%AC%EB%8F%85%EB%A6%BD%EC%84%B1-%ED%95%A9%EC%9D%98-%EB%B6%84%ED%8F%AC#:~:text=%EB%8F%85%EB%A6%BD%EB%8F%99%EC%9D%BC%EB%B6%84%ED%8F%AC(i.i.d.%3B%20independently,%ED%95%98%EC%97%AC%20%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%A5%BC%20%EC%96%BB%EB%8A%94%20%EA%B2%83.))
+ [[Numpy] Broadcasting(브로드캐스팅)](https://seong6496.tistory.com/54)
+ [[Numpy]행렬곱(@)과 내적(dot) 그리고 별연산(*)](https://seong6496.tistory.com/110)
+ [🌟 [Numpy] 파이썬 내적, 행렬곱 함수 np.dot() 사용법 총정리](https://jimmy-ai.tistory.com/75) -> 나중에 다시 공부