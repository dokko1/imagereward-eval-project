# ImageReward 기반 Text-to-Image 모델 평가 프로젝트

본 프로젝트는 ImageReward 모델을 활용하여 다양한 Text-to-Image(T2I) 모델의 
이미지 생성 품질을 정량적으로 평가하기 위한 실험 코드를 포함합니다.  
MS-COCO 프롬프트 기반으로 이미지 생성 → ImageReward 점수 평가 → 결과 저장의 
전체 파이프라인을 구현하였습니다.

---

## 📌 주요 내용

- ImageReward 점수를 활용한 T2I 모델 품질 평가
- Concept Erasure 모델(SA, SAFREE, UCE, MACE, ESD 등) 비교 가능
- COCO 기반 대규모 프롬프트(30k 중 10k 샘플링) 평가 환경 제공
- GPU 환경에서 자동화된 평가 스크립트 제공
- 이미지 생성 → 평가까지 end-to-end 파이프라인 구현

---

## 📁 디렉토리 구조
```
ImageReward/
├── generate_images.py        # 이미지 생성 스크립트
├── test_coco.py              # ImageReward 평가 코드
├── scripts/
│   ├── test-benchmark.sh     # 여러 모델의 결과를 배치 평가
│   ├── test.sh               # 개별 모델 테스트
│   └── test_coco.sh          # COCO 기반 ImageReward 평가
├── benchmark/
│   ├── coco_30k_10k.csv      # 평가 프롬프트
│   └── generations/          # 생성된 이미지 저장 경로
├── utils.py                  # 유틸리티 함수
```
---

## ⚙️ 설치 방법

필요한 패키지를 아래 명령으로 설치합니다.
```
pip install -r requirements.txt
```
---

## 🖼 이미지 생성 방법

`generate_images.py`에서 다음 항목을 설정할 수 있습니다:

- 사용할 모델 이름  
- batch size  
- 프롬프트 파일 경로  
- GPU device index  
- 출력 이미지 경로  

설정 후 아래와 같이 실행합니다:

```
python generate_images.py
```
이미지는 다음 경로에 저장됩니다:
```
benchmark/generations/<모델이름>/
```
---

## 🧪 ImageReward 평가 실행

이미지 생성이 끝난 뒤, 다음 스크립트로 ImageReward 점수를 계산합니다.

### 1) 전체 벤치마크 평가
```
bash scripts/test-benchmark.sh
```
### 2) 개별 평가 실행
```
bash scripts/test.sh
```
### 3) COCO 기반 평가
```
bash scripts/test_coco.sh
```

평가 결과는 CSV 파일 형태로 저장되며, 모델 간 점수 비교가 가능합니다.