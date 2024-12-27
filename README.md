# CGM: Copy Mechanism GPT with Mask for Ellipsis and Anaphora Resolution in Dialogue

## I. Index

> 1. 디렉토리 구조 
> 2. 실험 설정
> 3. 코드 실행 



## II. 디렉토리 구조 

### 1. 전체 디렉토리 

```shell
.
├── checkpoint_gpt
├── config
│   ├── gpt_test.yaml
│   └── gpt_training.yaml
├── data
│   ├── camRest676_annotated
│   │   ├── tagged
│   │   │   ├── test_tagged.jsonl
│   │   │   ├── training_tagged.jsonl
│   │   │   └── valid_tagged.jsonl
│   │   ├── test.jsonl
│   │   ├── total.jsonl
│   │   ├── training.jsonl
│   │   └── valid.jsonl
│   ├── generation_E
│   │   └── test.jsonl
│   ├── generation_E-f_change
│   │   └── test.jsonl
│   └── generation_f
│       └── test.jsonl
├── README.md
├── requirements.txt
├── run.py
└── src
    ├── agent
    │   └── gpt_agent.py
    ├── datamodule
    │   └──  datamodule.py
    └── model
        ├── backbone
        │   └── gpt_w_pgn_weight_adding_softmax.py
        └──  gpt.py
```

### 2. 각 디렉토리 별 구성 

#### 1) config

- 각 모델 학습 및 추론 별 설정을 나타낸 `.yaml` 파일 
- 생성 모델(GPT)과 학습 또는 추론 여부에 따라 2개 설정으로 구분 

#### 2) data

##### ① camRest676_annotated

- CamRest676 데이터에 대해 생략 / 대용 발생 추가 수정을 거친 데이터
- `camRest676_annotated/README.jsonl` 필수 참조 

##### ② generation_E

- **질문 복원 모델**(GPT, EEVE) 학습을 위해 split한 IT 도메인 데이터

##### ③ generation_E-f_change

- **도메인 변경 환경 내 질문 복원 성능 측정**을 위해 IT 도메인 데이터와 금융 도메인 데이터를 shuffle하여 구축한 데이터

#### 3) src

##### ① agent

- 전체 학습, 검증, 평가 과정을 실행하는 코드 

##### ② datamodule

- 모델에 제공할 데이터의 전처리를 실행하는 코드 

##### ③ model

- 실제 복원을 진행하는 모델 코드 



## III. 실험 설정 

### 1. config 파일 설정

#### 1) config 파일 구성 

| key                     | 설명                       | key                              | 설명                               |
| ----------------------- | -------------------------- | -------------------------------- | ---------------------------------- |
| seed                    | 모델 초기화 시드 설정      | name                             | 사용 LM 이름 (gpt 고정) |
| mode                    | 학습 또는 추론             | save_dir                         | 모델 가중치 저장 경로              |
| predict_file_path       | 예측 결과 파일 경로        | model.path                       | 사용할 모델 초기 가중치 경로       |
| model.num_beams         | 빔 서치 후보 개수          | model.generator                  | 사용 모델 구조                     |
| optimizer.lr            | 학습률                     | tokenizer.path                   | 토크나이저 경로                    |
| datamodule.batch_size   | 배치 크기                  | datamodule.shuffle               | 데이터 셔플 여부                   |
| datamodule.num_workers  | 데이터 로드 병렬화         | datamodule.data_dir              | 사용할 데이터 경로                 |
| datamodule.check_length | 각 데이터 파일 length 체크 | datamodule.max_{src/tgt}\_length | 최대 입출력 길이                   |



#### 2) config 파일 설정

- 진행할 실험에 맞추어 config를 작성해야 함. 

- 다음은 몇 가지 경우에 대한 예시임.

  - **추론 진행 시**

    - `mode`를 `predict`로 설정 후 `predict_file_path` 수정 

      **※ `predict_file_path`를 수정하지 않을 시 다른 설정으로 추론을 진행할 때마다 디폴트 경로에 덮어씌워짐**

    - `model.path`를 학습된 가중치 경로로 설정 (디폴트 `checkpoint_{model}/trained_model/`)

  - **데이터 경로 변경 시**

    -  `datamodule.data_dir`를 수정. 파일 명까지 다르다면 각각 파일 이름 또한 명시



### 2. 데이터 설정

- **디폴트**: `data/camRest676_annotated`

- 입력 데이터는 `jsonl` 형식이어야 하며, **새 데이터 사용 시** `jsonl` 파일 내 **`source`, `target`** 항목이 포함되어 있어야 함 
  - `source`: 입력 프롬프트 문장
  - `target`: 정답 문장 



## IV. 코드 실행 

### 1. 코드 실행 

- trainer 디렉토리 아래에서 다음 명령어를 실행 

  ```shell
  python run.py --config {config.yaml}
  ```

  - `*.yaml` 파일은 `--config` 인자를 통해 설정할 수 있음 



### 2. 실행 예 

```shell
$ python run.py --config config/gpt_training.yaml
INFO:__main__:COMMAND: python run.py --config config/gpt_training.yaml
INFO:__main__:SET seed 42
INFO:__main__:<CONFIG>
... (이하 config.yaml 내용) ...

INFO:src.datamodule.datamodule:LOAD data/camRest676_annotated/training.jsonl
INFO:src.datamodule.datamodule:LOAD data/camRest676_annotated/valid.jsonl
INFO:src.datamodule.datamodule:LOAD data/camRest676_annotated/test.jsonl
Some weights of PgnGPT were not initialized from the model checkpoint at openai_community/gpt2-xl and are newly initialized: ['gate.bias', 'gate.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

...

[TRAIN] Epoch0-L1.987-A0.682:   5%|████                                      | 16/326 [03:43<41:53,  8.11s/it]
```

