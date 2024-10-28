# 재활용 품목 분류를 위한 Object Detection

## 1. 📖 프로젝트 소개

![Head Image](https://github.com/boostcampaitech7/level2-objectdetection-cv-20/blob/main/etc/Recycle1.png)

우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다.분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다.

따라서 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 사진 데이터셋을 활용할 예정이며, 이를 통해 구축된 모델은 분리수거를 돕거나 어린아이들의 분리수거 교육 등에 사용 될 수 있습니다.

프로젝트 기간 : 24.09.30 ~ 24.10.28

```
부스트코스 강의 수강 및 과제 : 24.09.30 ~ 24.10.06
데이터 EDA / 데이터 전처리 / 베이스라인 모델 학습 : 24.10.07 ~ 24.10.13
데이터 증강 및 모델 성능 개선 : 24.10.14 ~ 24.10.18
하이퍼 파라미터 튜닝 / 앙상블 : 24.10.19 ~ 24.10.24
최종 자료 정리 및 문서화 : 24.10.25 ~ 24.10.28
```
<br/>

## 2.🧑‍🤝‍🧑 Team ( CV-20 : CV Up!!)

<table>
    <tr height="160px">
        <td align="center" width="150px">
            <a href="https://github.com/kaeh3403"><img height="110px"  src="https://avatars.githubusercontent.com/kaeh3403"></a>
            <br/>
            <a href="https://github.com/kaeh3403"><strong>김성규</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/sweetpotato15"><img height="110px"  src="https://avatars.githubusercontent.com/sweetpotato15"/></a>
            <br/>
            <a href="https://github.com/sweetpotato15"><strong>김유경</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/jeajin"><img height="110px"  src="https://avatars.githubusercontent.com/jeajin"/></a>
            <br/>
            <a href="https://github.com/jeajin"><strong>김재진</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/SuyoungPark11"><img height="110px" src="https://avatars.githubusercontent.com/SuyoungPark11"/></a>
            <br />
            <a href="https://github.com/SuyoungPark11"><strong>박수영</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/uddaniiii"><img height="110px" src="https://avatars.githubusercontent.com/uddaniiii"/></a>
            <br />
            <a href="https://github.com/uddaniiii"><strong>이단유</strong></a>
            <br />
        </td>
</table> 

|Name|Roles|
|:----------:|:------------------------------------------------------------:|
|김성규| 타임라인 관리, EDA 분석, Cascade r-cnn, EfficientDet, ATSS 모델 성능 실험, 앙상블(WBF) 적용 |
|김유경| Git 관리, small object 증강 실험, DDQ, H-Dino 모델 성능 실험 |
|김재진| EDA 분석, YOLO11, EfficientDet, DINO, ATSS 모델 성능 실험 |
|박수영| 회의록 작성 및 서버 사용 관리, 증강 실험|
|이단유| EDA 분석, faster r-cnn 모델 성능 실험, 증강 실험, 보고서 관리|

</div>

wrap up 레포트 : [wrap up report](https://github.com/boostcampaitech7/level2-objectdetection-cv-20/blob/main/Object%20Det_CV_%ED%8C%80%20%EB%A6%AC%ED%8F%AC%ED%8A%B8(20%EC%A1%B0).pdf)

<br/>
<br/>

## 3. 💻 프로젝트 수행 

### 3.1. 개발 환경 및 사용 툴
> 서버      : V100 GPU
>
> 버전 관리 : Github
> 
> 기록 관리 : Notion, Google Docs
>
> MLOps    : WandB 
>
> 기타      : Streamlit, Zoom
<br/>

### 3.2. 프로젝트 pipeline

> EDA 

![image](https://github.com/user-attachments/assets/f1fc2f31-ae32-484b-a25e-b20704b3fa70)
- 전체 데이터에서 Paper, Plastic bag, General trash 의 비율이 높으며, 클래스 당 객체 수의 분포가 고르지 않고 불균형이 심함을 확인했습니다. -> Oversamplig 기법 활용 

- 또한, 바운딩 박스의 크기가 이미지 면적의 5% 이하에 집중되어 있음을 확인했습니다. -> Mosaic, Small Object 기법 활용
<br/>  

> Preprocessing


- 데이터 시각화를 바탕으로 클래스 불균형 문제를 해결하기 위해 Oversampling 기법을 진행했습니다. 
- Detectron2의  RepeatFactorTrainingSampler, mmDetection v3의 ClassAwareSampler를 사용
  

| **Model** | **Backbone** | **Oversampling** | **mAP_50** |
|:-----------:|:---------:|:------------:|:----------:|
|   faster R-CNN  |   resnext101_fpn   |    X    |        0.4523       |
|   faster R-CNN  |    resnext101_fpn  |    O    |        0.4872       |
<br/>

> Augmentation

아래와 같은 3가지 증강 기법을 Faster R-CNN 모델에 적용해보았습니다. 

- 객체 탐지 논문에서 가장 많이 사용되는 기본적인 증강 기법들 (Horizontal/Vertical Flip, Random Crop, Color transform)
- 여러 장의 이미지를 하나로 결합하여 다양한 형태의 이미지를 생성하는 Mosaic 기법
- 작은 객체를 여러번 복사하여 붙여넣어 새로운 이미지를 생성하는 Small object 기법

![image](https://github.com/user-attachments/assets/81d5888f-56f8-499a-8f04-2d76f57ac669)
(좌측부터 원본 이미지, Basic Aug, Mosaic, Small Object 증강 기법 적용)

|  | **No Aug** | **Basic Aug** | **Mosaic** | **Small Object** |
|:---------:|:---------:|:---------:|:---------:|:---------:|
|   **mAP_50** |   0.3815   |   0.4640  |   0.4554  |   0.4645  |
<br/>

> Models

![image](https://github.com/user-attachments/assets/fcb28b54-5d51-4dd6-9336-56705328eca8)

- 기본적인 1-stage, 2-stage 모델부터 최신 SOTA 논문에 등재된 모델들까지 다양하게 실험했습니다. 
- 1-stage model : EfficientDet, YOLO11, 
- 2-stage model : Cascade R-CNN, faster R-CNN, Cascade mask R-CNN
- SOTA model : ATSS, DDQ-DETR, DINO, H-DINO
<br/>

> TTA (Test Time Augmentation)

- 추론 과정에서 모델이 다양한 입력을 통해 일반화 성능을 높이기 위해 TTA를 도입했습니다. 
- Resize와 RandomFlip 기법 적용

| **Model** | **Backbone** | **TTA** | **mAP_50** |
|:-----------:|:---------:|:------------:|:----------:|
|   H-DINO  |   ResNet50  |    X    |       0.3924       |
|   H-DINO  |   ResNet50  |    O    |       0.5488       |
<br/>

> Ensemble

- 성능이 가장 높았던 모델들을 활용하여, WBF, Soft-NMS, NMS 세 가지 기법의 앙상블을 적용하였습니다. 
- 다만, 단일 모델 성능이 앙상블 방식보다 높았습니다.

| **Method** | **Model1** | **Model2** | **Model3** | **mAP_50** |
|:-----------:|:---------:|:------------:|:----------:|:----------:|
|   WBF  |   DDQ*  |   DINO*  |     |   0.6597  |
|   WBF  |   DDQ(Mosaic)*  |   DINO*  |     |   0.6566  |
|   WBF  |   ATSS  |   DINO*  |   DDQ*  |   0.6581  |
|   NMS  |   ATSS  |   DINO*  |  DDQ*   |   0.6421  |
|   Soft-NMS  |   ATSS  |   DINO*  |   DDQ*  |   0.6258  |

(* : TTA 적용)
<br/>
<br/>

## 📈 4. 프로젝트 결과 

### 4-1. 프로젝트 결과

- 최종 선택된 모델은 아래와 같습니다. 

| **Model** | **Backbone** | **Preprocessing** | **score threshold** | **etc** | **mAP_50** |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
|   DINO |   Swin-l   |   Oversampling  |   0.05  |   feature map level : 6 <br/> TTA  | 0.6660 |

- 실제 feature map level 을 5에서 6으로 변경한 이후 EDA는 다음과 같습니다. 
- 이미지 크기 정확도
![image](https://github.com/user-attachments/assets/5da7331d-2a8c-45de-a3d8-780569671add)
- 작은 객체 정확도
![image](https://github.com/user-attachments/assets/7c99df36-136f-47c6-acf6-ab2bd1a85a3c)
<br/>

### 4-2. 프로젝트 실행방법

> DINO 실행 방법 및 코드

1. `model/mmdetection_v3` 내로 디렉토리 변경
```
cd model/mmdetection_v3
```
2. `train.py` 실행
```
python tools/train.py {config 폴더 경로}

# DINO Train:

python tools/train.py my_config/dino_swin_recycle_v3.py
```

3. `test.py` 실행
```
python tools/test.py {config 폴더 경로} {model checkpoint 경로} [tta 여부]

# DINO Test:

python tools/test.py my_config/dino_swin_recycle_v3.py my_config/work_dirs/dino_recycle/epoch_num.pth --tta
```

4. 생성된 json file을 `submission` 형태로 변환
```
python ../../submission/json_to_submission.py {생성된 json file 경로} {output 디렉토리 경로} {output file 이름} {score_threshold} [train 여부]

# Convert 

python ../../submission/json_to_submission.py my_config/work_dirs/dino_recycle/submission.json ../../submission dino_output_file 0.05
```
<br/>

## 5. 프로젝트 구조
프로젝트는 다음과 같은 구조로 구성되어 있습니다. 
```
📦level2-imageclassification-cv-20
 ┣ 📂etc # 프로젝트 설명을 위한 기타 파일 폴더
 ┣ 📂experiments # 실험 관련 폴더
 ┃ ┣ 📂EDA # 모델 학습 전, 후 EDA 폴더
 ┃ ┣ 📂ensemble # 앙상블 관련 폴더
 ┃ ┣ 📂mosaic # mosaic 증강 관련 폴더
 ┃ ┗ 📂small_object # 작은 객체 탐지 관련 폴더
 ┣ 📂model # 모델 라이브러리 관련 폴더
 ┃ ┣ 📂detectron2 
 ┃ ┣ 📂mmdetection 
 ┃ ┣ 📂mmdetection_v3 
 ┃ ┗ 📂yolo 
 ┗ 📂submission # csv output 폴더
```

<br/>


## 6. 기타사항

- 본 프로젝트에서 사용한 데이터셋의 적용 저작권 라이선스인 CC-BY 2.0([link](https://creativecommons.org/licenses/by/2.0/kr/))의 가이드를 준수하고 있습니다.

