다음은 프로젝트의 논문과 코드를 바탕으로 작성한 README.md 내용입니다.

-----

# Jeju-dialect-speech-to-text

[cite\_start]이 프로젝트는 제주 방언 음성을 표준 한국어 텍스트로 변환하는 모듈형 음성-텍스트 번역 시스템을 다룹니다[cite: 15]. [cite\_start]시스템은 Whisper 인코더, 커넥터 모듈(MLP, Q-Former, STE), 그리고 T5 디코더로 구성됩니다[cite: 15].

## 프로젝트 소개

[cite\_start]최신 음성 인식 기술은 대부분 표준 언어를 기반으로 개발되어, 제주 방언과 같은 다양한 언어 변형을 처리하는 데 한계가 있습니다[cite: 22]. [cite\_start]제주 방언은 유네스코에 의해 '소멸 위기 언어'로 지정될 만큼 문화적, 언어적 가치가 높지만 [cite: 24][cite\_start], 현재의 자동 음성 인식(ASR) 시스템에서는 낮은 성능을 보입니다[cite: 25].

이 연구는 Whisper와 같은 대규모 사전 훈련 모델을 제주 방언에 적용할 때 발생하는 문제를 해결하고자 합니다. [cite\_start]단순한 파인튜닝만으로는 제주 방언과 표준 한국어 사이의 상당한 어휘적, 문법적, 음운적 차이를 극복하기 어렵습니다[cite: 27]. [cite\_start]따라서 이 프로젝트에서는 Whisper 인코더와 T5 디코더를 결합한 End-to-End 음성 번역 아키텍처를 제안하고, 두 모델을 연결하는 다양한 커넥터 모듈의 성능을 실험합니다[cite: 35, 36].

## 모델 아키텍처

모델은 크게 세 부분으로 구성됩니다:

1.  [cite\_start]**ASR Encoder (Whisper)**: 입력된 음성에서 음향적 특징을 추출하는 인코더입니다[cite: 36].
2.  **Connector**: Whisper 인코더와 T5 디코더의 임베딩 공간을 연결하는 모듈입니다. [cite\_start]이 프로젝트에서는 MLP, Q-Former, STE 세 가지 커넥터를 실험했습니다[cite: 16].
      * [cite\_start]**Q-Former**: 학습 가능한 쿼리 임베딩을 사용하여 ASR 출력에서 정보를 집계하고 변환하는 경량 트랜스포머 기반 커넥터입니다[cite: 82, 83, 86].
      * [cite\_start]**STE (Subsampler-Transformer Encoder)**: 1D Convolution으로 구성된 서브샘플링 모듈과 트랜스포머 인코더 블록을 결합하여 음성 임베딩을 처리합니다[cite: 88].
3.  [cite\_start]**LLM Decoder (T5)**: 커넥터를 통해 전달된 임베딩을 바탕으로 최종적인 표준 한국어 텍스트를 생성하는 디코더입니다[cite: 36, 63].

[cite\_start]*그림 1: 제안된 Whisper-Connector-T5 모델 아키텍처 개요 [cite: 34]*

## 데이터셋

[cite\_start]AI Hub에서 제공하는 '한국어 방언 발화(제주도)' 데이터셋을 사용했습니다[cite: 110]. [cite\_start]이 데이터셋은 제주 지역 발화자의 음성 녹음과 해당 표준 한국어 텍스트로 구성되어 있습니다[cite: 111]. [cite\_start]리소스 제약으로 인해 전체 데이터 중 약 70GB를 훈련, 13GB를 검증, 5GB를 테스트용으로 사용했습니다[cite: 113].

## 실험 결과 및 분석

[cite\_start]실험 결과, MLP, Q-Former, STE를 사용한 모든 커넥터 기반 모델이 단순 파인튜닝한 Whisper 모델보다 성능이 저조했으며, BLEU 점수는 0에 가까웠습니다[cite: 16, 151]. [cite\_start]이는 Whisper의 음향-음성적 표현과 T5의 구문-의미론적 기대치 사이에 불일치가 존재하기 때문으로 분석됩니다[cite: 17].

[cite\_start]UMAP 시각화 결과, Whisper, 커넥터, T5의 임베딩 공간이 명확하게 분리되어 있어, 커넥터가 두 표현 공간을 효과적으로 정렬하지 못하고 있음을 확인했습니다[cite: 179, 180, 183].

[cite\_start]흥미롭게도, 한국어 T5 디코더를 영어 T5 디코더로 교체했을 때 BLEU 점수가 향상되었습니다[cite: 18]. [cite\_start]이는 언어 호환성 때문이 아니라, 영어 T5 모델이 더 강력한 사전 훈련을 거쳤고, Whisper의 음절 리듬과 잘 맞는 바이트 수준 토큰화를 사용하기 때문입니다[cite: 18, 194, 213].

## 코드 실행 방법

### 의존성 설치

```bash
pip install torch torchaudio pandas transformers matplotlib jiwer
```

### 훈련

훈련을 실행하려면 `whisper_t5_ddp_connector.py` 스크립트를 사용합니다. DDP(DistributedDataParallel)를 사용한 다중 GPU 학습을 지원합니다.

```bash
python whisper_t5_ddp_connector.py [arguments]
```

**Arguments:**

  * `--train_csv`: 훈련 데이터 CSV 경로 (기본값: `/home/aikusrv02/dialect/data/train_valid.csv`)
  * `--valid_csv`: 검증 데이터 CSV 경로 (기본값: `/home/aikusrv02/dialect/data/valid_valid.csv`)
  * `--test_csv`: 테스트 데이터 CSV 경로 (기본값: `/home/aikusrv02/dialect/data/test_valid.csv`)
  * `--num_epochs`: 학습 에폭 수 (기본값: 5)
  * `--learning_rate`: 학습률 (기본값: 5e-5)
  * `--warmup_ratio`: 웜업 비율 (기본값: 0.1)
  * `--weight_decay`: 가중치 감쇠 (기본값: 0.01)
  * `--batch_size`: 배치 크기 (GPU당) (기본값: 1)
  * `--model_save_path`: 모델 저장 경로 (기본값: `./saved_models_ddp_all/`)
  * `--connector`: 커넥터 종류 (`mlp`, `qformer`, `ste`) (기본값: `qformer`)

## 결론

이 프로젝트를 통해 모듈형 음성 번역 시스템에서 인코더와 디코더 간의 표현 공간을 정렬하는 것이 매우 중요하며, 단순히 두 모델을 연결하는 것만으로는 충분하지 않다는 것을 확인했습니다. [cite\_start]특히 디코더의 사전 훈련 강도와 토크나이저의粒度(granularity)가 전체 시스템 성능에 큰 영향을 미친다는 점을 발견했습니다[cite: 212, 216].
