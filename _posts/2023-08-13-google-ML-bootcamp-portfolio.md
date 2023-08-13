---
layout: post
title: Seulin Jeong
subtitle: Valuable Experiences I've had
categories: 
tags: []
use_math: true
type: hidden
---

안녕하세요! 인공지능 개발자 정슬인입니다.

<br>
새로운 경험은 항상 가치 있는 일이라 생각하여 웹/앱/AI 등 다양한 분야를 공부해보았습니다. 다양한 경험은 어떤 업무가 주어지든 빠르게 적응할 수 있는 바탕이 되었다고 생각합니다.

<br>
또한, 스타트업 및 당근마켓에서 근무하며 데이터 분석을 통해 서비스를 개선한 경험이 있습니다. 데이터 분석을 통해 새로운 가치를 발견하고 이를 활용해 서비스를 개선했을 때 보람을 느낍니다.

<br>
:heavy_check_mark:&nbsp; 다양한 협업 경험을 통해 어떤 환경에서도 빠르게 적응합니다.

:heavy_check_mark:&nbsp; 눈앞의 쉬운 길보다는 어렵지만 가치 있는 길을 좋아합니다.

## :pushpin:&nbsp; Profile

### :link:&nbsp; Contacts
- Email: ths06123@gmail.com
- GitHub: [Seulin](https://github.com/Seulin)

### :mortar_board:&nbsp; Education
- 대구경북과학기술원 (DGIST)
    - 기초학부 공학 학사, 컴퓨터 트랙
    - 2018.02 - 2023.02
    - GPA: 4.09/4.3

### :books:&nbsp; Stacks
- Web Development: `SQL`, `Vue.js`, `PHP`, `Node.js`, `AWS`, `Docker`
- Data Analysis: `Python`, `PyTorch`, `NumPy`, `pandas`, `Matplotlib`

### :globe_with_meridians:&nbsp; Language Skills
- TOEFL 97 (2023.03)

## :book:&nbsp; Projects

<details markdown="1">
<summary class="h3-title">
&nbsp; 다양한 포맷의 악성코드 탐지를 위한 Transformer 모델
</summary>
> 2022.06 - 2022.10 
> 
> AI보안 기술개발 교육과정 - 한국정보보호산업협회 (KISIA)

#### 프로젝트 개요
- 문제 제기 및 솔루션
    - 고정된 feature로 학습한 악성코드 탐지 모델은 다양한 포맷의 악성코드에 적용할 수 없음
    - 동일한 포맷의 악성코드에서도 추출 가능한 feature가 다양함
    - 유용한 feature를 모두 활용하기 위한 Transformer 모델로 악성코드 탐지를 하고자 함
- 과정
    - Virus Total에 악성코드 샘플을 요청하여 전달받은 샘플을 연구에 활용
    - PE, APK, ELF 악성코드에서 추출 가능한 feature 분석 및 추출
    - Transformer를 악성코드 탐지 모델로 선정 및 개발
    - Evasion Attack을 막기 위한 SHAP (XAI) 활용 방안 제시
- 결과
    - 모델
        - train data: 1000, test data: 300
        - accuracy: 0.69

#### 성과
- [과학기술정보통신부 장관상](/assets/pdf/awards/sec-individual.pdf) 수상 *- 과학기술정보통신부*
    - AI보안 기술개발 교육과정 개인 부문 최우수상
    - 팀 프로젝트 성과 및 개인 평가 포함
- [AI보안 기술개발 교육과정 우수 활동팀](/assets/pdf/awards/sec-team.pdf) 수상 *- 한국정보보호산업협회*
    - AI보안 기술개발 교육과정 팀 프로젝트 부문 공동 1등 상

#### 역할
- 팀장: 주간 회의 진행, 일정 조율, 발표
- 팀 구성: 보안 2, AI 2
- 인공지능 모델 분석 및 Transformer 제안
    - Vision Transformer (ViT, 2020)가 등장한 이후로 Transformer가 자연어처리 외에도 좋은 성능을 보인다는 것을 증명함
    - 서로 다른 크기의 input을 받을 수 있어서 악성코드의 모든 feature를 활용할 수 있음
    - Attetion mechanism으로 병렬 처리가 가능함
- PE, 공통 feature 분석 및 추출
- Transformer 모델 개발

#### 기술 및 라이브러리
- Transformer Implementation: `Python`, `PyTorch`, `NumPy`, `pandas`
- Malware Feature Extraction: `pefile`

#### 관련 자료
[GitHub](https://url.kr/6g4ixm)

[Weekly Meeting Minutes](https://docs.google.com/document/d/1m5AwoPpe4Jtsu6hwGuRbnX8VqkdvTgynbwP2VijkqKg/edit?usp=sharing)

[Report & Presentation](https://docs.google.com/presentation/d/1_dTFK6WphDtymrteWm--TwfOYkCw0CzKasFkNM0yErw/edit?usp=sharing)

</details>

<details markdown="1">
<summary class="h3-title">
&nbsp; 긴급차량의 골든타임 확보 및 교통체증 감소를 위한 Multi-Agents 신호 시스템
</summary>

> 2022.04 - 2022.06 
> 
> '강화학습' 강의 프로젝트 - DGIST

#### 프로젝트 개요
- 문제 제기 및 솔루션
    - 긴급차량 통행을 위해 수동으로 신호 시스템을 제어하는 것에는 한계가 있음
    - 고정적인 신호 시스템은 교통체증을 증가시키는 원인임
    - 긴급차량 통행을 우선시하고 부차적으로 교통체증까지 줄이는 신호등을 만들고자 함
- 모델
    - Algorithm: Q-learning
    - Agent: 신호등
    - Action: 가능한 초록신호 phase
    - State: 긴급차량 위치 및 속도, 교통체증 수치 등을 포함한 크기 26의 vector 
    - Reward Model
        - WT: average waiting time of vehicles <br>
        - EVS: emergency vehicle's speed <br>
        - $\alpha$: Weight to prioritize emergency vehicles
        - $Reward = -WT + EVS*\alpha$
- 결과
    - 학습 영상
    ![road](/assets/images/posts/road.gif){: width="50%"}
    <br>
    - 긴급차량 통행 시간 60% 감축

        |: Fixed traffic light :|: Trained traffic light :|
        | -- | -- |
        | ![EV_travel](/assets/images/posts/evt-fixed.png) | ![EV_travel2](/assets/images/posts/evt-learned.png) |
        |: Avg: 145.3s :|: Avg: 61.4s :|
        {: .no-space}

    - 일반차량 대기 시간 15% 감축

        |: Fixed traffic light :|: Trained traffic light :|
        | -- | -- |
        | ![Watiting time](/assets/images/posts/awt-7000-fixed.png) | ![Waiting time](/assets/images/posts/awt-7000-learned.png) |
        |: Avg: 1552.3s :|: Avg: 1331.2s :|
        {: .no-space}

#### 역할
- 팀장
- 팀 구성: 2인
- State, Reward model design
- Model Training & Simulation
- Results Analaysis & Visualization
- 학습시간 86% 단축 (182m → 26m)
    - Q-table의 크기가 매우 커 학습시간이 오래 걸림
    - $Q-table \, size = (num\;states) * (num \; actions) * (num \; agents) = 3.6*10^{15}$
    - Decaying ε-greedy의 ε와 decay 값 최적화로 학습시간을 대폭 줄임

#### 기술 및 라이브러리
- Reinforcement Learning: `Python`, `OpenAI gym`
- Simulation: `SUMO (Simulation of Urban MObility)` - [Official Site](https://www.eclipse.org/sumo/)

#### 관련 자료
[GitHub](https://url.kr/iwt945)

[Report](https://docs.google.com/document/d/1fVuZjVmgpYKKvaU3rci1VCDs7f1Hm2K93z4LSrYdBBc/edit?usp=sharing)

[Presentation](https://docs.google.com/presentation/d/1_isdUFmhiau14VszPiuuSGmSbYC4R34T5BFDYWlzVOA/edit?usp=sharing)

</details>

<details markdown="1">
<summary class="h3-title">
&nbsp; AI 프레임워크 없이 딥러닝 모델 구현
</summary>

> 2020.10 - 2020.12 
> 
> '딥러닝 개론' 강의 프로젝트 - DGIST

#### 프로젝트 개요
- 모델
    - PyTorch에서 제공하는 함수와 동일한 Abstarct Data Type를 가지도록 작성함
    - DNN, CNN Model: MNIST 숫자 예측
    - RNN, LSTM Model: 문장의 감정 예측
- 구현 내용
    - Layer: Linear, Convolutaion, Max-pooling, Relu, LeakyRelu, Softmax, Dropout
    - RNN, LSTM Cells
    - Cross Entropy Loss & Backpropagation
    - SGD optimizer & ADAM optimizer
    - Confusion Matrix & Loss graph & Accuracy graph
    - Hyperparameter Optimizing
- 결과
    - DNN
        - Linear → R/L → Linear → R/L → Linear → Softmax
        - Average Accuray: 0.902

        |: R: Relu :|: L: LeakyRelu :|
        | -- | -- |
        |![Relu](https://raw.githubusercontent.com/Seulin/Deep-Learning-Models-without-Frameworks/main/DNN/figure/relu_cf.png)|![Leaky Relu](https://raw.githubusercontent.com/Seulin/Deep-Learning-Models-without-Frameworks/main/DNN/figure/lrelu_cf.png)|
    - CNN
        - Convolution → Convolution → Linear → Softmax
        - Average Accuray: 0.836

        ![CNN](/assets/images/posts/CNN.png){: width="70%"}

    - RNN
        - Test Accuray: 0.679

        |: Accuracy :|: Loss :|
        | -- | -- |
        | ![RNN Accuracy](https://raw.githubusercontent.com/Seulin/Deep-Learning-Models-without-Frameworks/main/RNN-LSTM/A%20accuracy%20graph.png) | ![RNN Loss](https://raw.githubusercontent.com/Seulin/Deep-Learning-Models-without-Frameworks/main/RNN-LSTM/A%20loss%20graph.png)|

    - LSTM
        - Test Accuray: 0.643

        |: Accuracy :|: Loss :|
        | -- | -- |
        | ![lstm accuracy](https://raw.githubusercontent.com/Seulin/Deep-Learning-Models-without-Frameworks/main/RNN-LSTM/E%20accuracy%20graph.png) | ![lstm loss](https://raw.githubusercontent.com/Seulin/Deep-Learning-Models-without-Frameworks/main/RNN-LSTM/E%20loss%20graph.png) |

#### 기술 및 라이브러리
- Model Implementaion: `Python`, `NumPy`
- Visualization: `Matplotlib`, `searborn`, `tensorboard`

#### 관련 자료
[GitHub](https://url.kr/34ce69)

</details>

<details markdown="1">
<summary class="h3-title">
&nbsp; 소스코드 유사도 분석 알고리즘 및 이를 이용한 교육 보조 프로그램 
</summary>

> 2020.03 - 2020.12 
> 
> Undergraduate Group Research Program (UGRP) - DGIST

#### 프로젝트 개요
- 문제 제기 및 솔루션
    - 코딩을 처음 배우는 학생들의 코드는 길이가 짧아 구조적 유사성은 적합한 표절 기준이 아님
    - 코딩 스타일을 기준으로 표절도를 검사하되 학습 과정에 따른 코딩 스타일의 가변성을 고려함
    - 이 알고리즘을 적용한 코딩 교육 사이트를 만들고자 함
- 결과
    - 코딩 스타일 기반의 소스코드 유사도 계산 알고리즘 연구 (Python, tokenizer)
        - 변수명, 클래스명, 연산자 주위 공백 등을 기준으로 코딩 스타일 vector를 생성
        - 지금까지 작성해온 코딩 스타일과 현재의 코딩 스타일을 비교하여 표절도를 계산함
        - 코딩 스타일에 급격한 변화가 있을 경우 높은 표절도를 가짐
    - 코드 표절 검사 기능을 활용한 코딩 교육 사이트 (MEVN stack)
        - 로그인 및 회원가입
        - 강좌 및 과제 등록
        - 개인 학습현황
        - 실습 과제를 위한 실시간 Python interpreter
            - 샌드박스 및 에디터 화면 분할 기능
            - 화면
                ![editor](https://user-images.githubusercontent.com/52347271/230982511-6f7663ba-234a-4570-bc8e-0cea3ecfaf0f.jpg)
            
#### 역할
 - 팀 구성: 4인
 - 알고리즘 연구
    - 연산자 및 괄호 주위 공백, 평균 함수 길이 등의 코딩 스타일 vector 추출
    - 표절도 계산식 정립
    - K-means Clustering을 이용한 군집 내 편차 계산
 - 사이트 개발
    - Docker와 Socket을 사용한 실시간 Python interpreter 구현
    - 강의 캘린더 및 일정 조율 기능
    - AWS EC2 서버 및 MongoDB Atlas 관리


#### 기술 및 라이브러리
- Data Analysis: `Python`, `tokenizer`, `pandas`, `NumPy`
- Web Development: `Vue.js`, `Node.js`, `AWS EC2(Ubuntu)`, `MongoDB`, `GitHub`, `Vuetify`
- Extra: `Docker`, `Socket.io`

#### 관련 자료
[GitHub](https://url.kr/nsvhta)

</details>


## :computer:&nbsp; Work Experiences

<details markdown="1">
<summary class="h3-title">
&nbsp; 당근마켓 썸머테크 인턴십 - (주)당근마켓
</summary>

> 2023.06 - 현재 
> 
> 검색 품질 향상을 위한 머신러닝 모델 설계

#### 역할
- 게시글 카테고리 추정을 위한 자연어 처리 모델 개발
    - 100만 건 이상의 데이터셋 활용
    - 퀄리티 지표 분석 및 이를 반영한 모델 학습
    - 모델 결과 분석 및 개선
- 검색 랭킹모델 고도화
    - Hyper-parameter opitmization
    - 새로운 Feature 도입
    - CTR 분석 및 모델 개선


#### 기술 및 라이브러리
Data Analysis: `PyTorch`, `pandas`, `SQL`, `BigQuery`

</details>


<details markdown="1">
<summary class="h3-title">
&nbsp; 스타트업 소프트웨어 엔지니어 - Robinhood Plane Inc.
</summary>

> 2021.04 - 2023.05
> 
> 글로벌 마켓 리서치 웹사이트 개발 및 리서치 데이터 분석

#### 서비스 개요
- Small business owner의 해외 진출을 위한 글로벌 마켓 리서치 서비스
    - 제품 판매 전 소비자들의 반응을 파악하기 위한 마켓 리서치
    - target country, recommended price, strategy 등의 인사이트 제공

#### 역할
- 사이트 내의 Funnel Analysis
    - Google Analytics와 hotjar을 통한 user behavior 로그 분석
    - user flow 개선을 통한 이탈률 4% 감축
        - 회원가입 시 입력 정보 최소화 및 소셜 로그인 도입
        - 간편한 리서치 참여를 위한 질문 형식 변경 (주관식 → 객관식)
- 1500여 건의 마켓 리서치 데이터 분석
    - Recommended Price logic 구현
    - Google Trend API 활용
- 마켓 리서치 웹사이트 개발
    - Eximbay 결제 시스템 도입
    - PM: radiansys 사와 개발 외주 진행 (2021.09~)
    - AWS EC2 서버 및 MySQL 관리

#### 성과
- 2022 창업진흥원 글로벌창업사관학교 입교팀 선정
- 한국 대표 스타트업으로 선정 및 2022 [Slush](https://www.slush.org/about/) (Start-up Conference in Finland) 참여

#### 기술 및 라이브러리
- Web Development: `PHP`, `SQL`, `AWS EC2(Centos)`, `GitHub`, `Figma`
- Data Analysis: `Python`, `pandas`

</details>


<details markdown="1">
<summary class="h3-title">
&nbsp; 유펜솔루션 인턴십 - UpennSolution Co., Ltd.
</summary>

> 2021.02 - 2021.04
> 
> 쉽게 데이터 처리/분석을 할 수 있는 프로토타입의 웹사이트 개발

#### 서비스 개요
- 데이터 처리/분석을 쉽게 할 수 있도록 도와주는 서비스
    - 프로그래밍 없이 사용자가 직접 데이터 처리와 분석이 가능함
    - Table 형태의 데이터를 다룸

#### 역할
- PoC 웹사이트 제작
    - Frontend
        - 파일 업로드 및 버전 관리(변경 히스토리) 기능 구현
        - 필드 병합/분리/타입변환, 테이블 병합 등 기능 구현
        - 대용량 파일의 경우, 병렬 처리를 통해 렌더링 속도를 80%(50s → 10s) 감축
    - Backend
        - Isolation Forest를 통한 이상치 제거
        - K-NN을 통한 결측치 대체
        - 데이터 시각화

#### 기술 및 라이브러리
- Web Development: `Vue.js`, `Vuetify`, `axios`, `Django`, `Gitlab`
- Data Analysis: `Python`, `pandas`, `scikit-learn`, `Matplotlib`

</details>








## :trophy:&nbsp; Awards
- [대구광역시장상](/assets/pdf/awards/daegu.pdf) *- 대구광역시 (2023.02)*
- [과학기술정보통신부 장관상](/assets/pdf/awards/sec-individual.pdf) 수상 *- 과학기술정보통신부 (2022.11)*
- [AI보안 기술개발 교육과정 우수 활동팀](/assets/pdf/awards/sec-team.pdf) 수상 *- 한국정보보호산업협회 (2022.11)*
- [2020 Dean's List](/assets/pdf/awards/dean.pdf) *- DGIST (2020)*
- [대한민국 국회의원상](/assets/pdf/awards/assembly.pdf) *- 대한민국 국회 (2018.02)*
- [DGIST 총장상](/assets/pdf/awards/president.pdf) *- DGIST (2018.02)*


## :page_with_curl:&nbsp; Certificates
- [NVIDIA](/assets/pdf/certificates/NVIDIA.pdf) *- NVIDIA*
    - [Building Conversational AI Applications](https://www.nvidia.com/en-us/training/instructor-led-workshops/building-conversational-ai-apps/)
    - [Building Transforemr-Based Natural Language Processing Applications](https://www.nvidia.com/en-us/training/instructor-led-workshops/natural-language-processing/)
- [AWS](/assets/pdf/certificates/AWS.pdf) *- AWS*
    - [Developing on AWS](https://aws.amazon.com/ko/training/classroom/developing-on-aws/?ct=sec&sec=rolesol)
    - [Deep Learning on AWS](https://aws.amazon.com/ko/training/classroom/deep-learning-on-aws/?ct=sec&sec=rolesol)
    - [Building Data Lakes on AWS](https://aws.amazon.com/ko/training/classroom/building-data-lakes/?ct=sec&sec=rolesol)
    - [Architecting on AWS](https://aws.amazon.com/ko/training/classroom/architecting-on-aws/?ct=sec&sec=rolesol)
- [Unlocking Information Security Program](/assets/pdf/certificates/edX.pdf) *- IsraelX*
    - [From Cryptography to Buffer Overflows](https://www.edx.org/course/unlocking-information-security-i-from-cryptography-to-buffer-overflows)
    - [An Internet Perspective](https://www.edx.org/course/unlocking-information-security-ii-an-internet-perspective)
- [2021 DGIST 하계 인턴 프로그램](/assets/pdf/certificates/lab.pdf) *- 대구경북과학기술원*