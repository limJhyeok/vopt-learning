# vopt-learning

Vision 모델 및 파이프라인 최적화(TensorRT 등) study 노트. 

로컬에 GPU가 없어서 **노트북은 Colab GPU 런타임에서 실행**하는 것을 전제로 합니다.

## 노트북
| 노트북 | 내용 | Colab |
|---|---|---|
| [`tensorrt/01_basic.ipynb`](notebooks/tensorrt/01_basic.ipynb) | PyTorch → ONNX → TensorRT 변환과 추론, 정확도·속도 비교 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/limJhyeok/vopt-learning/blob/main/notebooks/tensorrt/01_basic.ipynb) |

## 구조
```
notebooks/            # study 노트북 (각 노트북 첫 셀 = Colab setup)
  tensorrt/           #   TensorRT 시리즈
src/vopt/             # study 본문과 무관한 헬퍼
  env.py              #   실행 환경 출력, GPU 태그
  cache.py            #   빌드 산출물 캐시 (로컬/Drive → GitHub Release → 빌드)
  trt_utils.py        #   TensorRT build / load / TRTModule(torch 텐서 입출력)
  bench.py            #   CUDA event 기반 latency 벤치마크
scripts/
  publish_artifact.sh # 빌드된 engine을 GitHub Release(tag: artifacts)에 업로드
```

## 사용법
1. VS Code에서 노트북을 열고 **Select Kernel → Colab → GPU**를 선택합니다.
   (또는 위 표의 **Open In Colab** 링크로 열고 **런타임 → 런타임 유형 변경 → GPU**를 선택합니다.)
2. 첫 셀(Setup)을 실행합니다. 저장소 clone, 필요한 패키지 설치, `vopt` import까지 진행됩니다.
3. `src/vopt`를 수정했다면 push한 뒤 Setup 셀을 다시 실행합니다(autoreload 적용).

## Prebuilt engine
`trt_utils.get_engine()`은 아래 순서로 engine을 찾습니다.
1. 캐시 디렉토리 (`$VOPT_CACHE` → Drive `MyDrive/vopt-cache` → `/content/vopt-cache`)
2. GitHub Release `artifacts`
3. 위 두 곳에 없으면 직접 빌드

파일명은 `{name}-{hash}-{gpu}-sm{cc}-trt{ver}.engine` 형식이며, GPU 아키텍처와 TensorRT 버전이 같아야 재사용할 수 있습니다.
업로드: `Colab: Download...`로 로컬 `artifacts/`에 받은 뒤 `scripts/publish_artifact.sh artifacts/*.engine`을 실행합니다.
