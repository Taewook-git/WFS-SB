# A100 원클릭 실험 실행

이 문서는 A100 80GB 한 장에서 기본 논문 준비 실험을 바로 시작하는 경로다. 기본 실행은 VideoMME의 첫 20개 고유 영상, 실제 1 FPS sampling origin 5개, DWT/SWT, K=16, BLIP-2 ITM, Qwen2.5-VL-7B 조합이다.

## 1. 현재 작업을 개인 GitHub fork에 게시

Windows 작업 PC의 저장소 루트에서 한 번 실행한다.

```powershell
.\scripts\publish_fork.ps1
```

GitHub 로그인이 없으면 브라우저 인증을 열고, `MAC-AutoML/WFS-SB` 개인 fork를 만든 뒤 `phase-stable-icassp` branch를 push한다. 스크립트는 dirty worktree나 다른 URL을 가리키는 `fork` remote를 발견하면 push하지 않고 중단한다.

## 2. A100 서버에서 clone 후 시작

`<GITHUB_ID>`만 본인 계정으로 바꾼다.

```bash
git clone --branch phase-stable-icassp https://github.com/<GITHUB_ID>/WFS-SB.git
cd WFS-SB
bash scripts/run_a100_experiment.sh
```

처음 실행할 때 다음 작업을 순서대로 수행한다.

1. 현재 checkout이 깨끗한지 확인하고 origin branch를 fast-forward한다.
2. Python 3.10 venv `~/.venvs/wfs-sb-a100`을 만들고 의존성을 설치한다.
3. `lmms-eval`을 고정 commit `bb1ebe76...`에 checkout하고 저장소 patch를 정확히 적용한다.
4. Hugging Face 로그인이 없으면 공식 `hf auth login`을 실행한다.
5. 필요한 VideoMME archive chunk를 내려받아 요청한 20개 MP4가 모두 있는지 검증한다.
6. sampling manifest, BLIP-2 전처리, DWT/SWT 분석, Uniform/Top-K baseline, keyframe export를 실행한다.
7. DWT/SWT × origin 5개의 Qwen evaluation을 독립 cell로 실행한다.
8. `lmms-eval`의 공식 answer parser 결과를 병합하고 video-cluster paired bootstrap 결과를 만든다.

Hugging Face token을 환경변수로 주고 싶다면 shell history에 남지 않게 입력한다.

```bash
read -rsp 'HF token: ' HF_TOKEN && echo
export HF_TOKEN
bash scripts/run_a100_experiment.sh
unset HF_TOKEN
```

token은 명령행 인자나 실험 로그에 기록하지 않는다. bootstrap은 인증 cache를 만든 뒤 MLLM child process에서 `HF_TOKEN` 환경변수를 제거한다.

## 3. 중단 후 재개

동일한 명령을 다시 실행하면 된다.

```bash
bash scripts/run_a100_experiment.sh
```

Stage-0 marker는 명령·입력·코드 fingerprint와 모든 산출물 checksum을 확인한다. MLLM marker도 keyframe checksum, model/task 설정, 결과 및 sample-log checksum을 확인한다. 값이 달라진 cell만 다시 실행한다.

장시간 SSH 연결에는 `tmux` 사용을 권장한다.

```bash
tmux new -s ti-dwt
bash scripts/run_a100_experiment.sh 2>&1 | tee a100_experiment.log
```

## 4. 자주 쓰는 변형

이미 VideoMME가 `/data/VideoMME`에 있다면 다운로드를 생략한다. `/data/VideoMME/data/*.mp4`와 annotation JSON이 있어야 한다.

```bash
bash scripts/run_a100_experiment.sh \
  --dataset-root /data/VideoMME \
  --no-download-data
```

keyframe export까지만 먼저 확인한다.

```bash
bash scripts/run_a100_experiment.sh --skip-mllm
```

Uniform/Top-K까지 MLLM grid에 포함한다.

```bash
bash scripts/run_a100_experiment.sh --include-baselines
```

완료된 Stage-0를 재사용해 DWT와 SWT 모두 경계 4개로 고정한 반사실적
keyframe/MLLM 실험만 실행한다. 기존 일반 DWT/SWT 산출물은 수정하지 않고
`matched_cardinality/b04/` 아래에 새 10개 cell을 만든다.

```bash
bash scripts/run_a100_experiment.sh \
  --skip-bootstrap \
  --no-download-data \
  --matched-only \
  --matched-count 4
```

이 명령은 기존 `origin_signals.jsonl`이 있어야 하며, 재실행하면 완료 marker가
유효한 matched cell을 건너뛴다. 원래 Stage-0가 기록한 effective config도
읽기 전용으로 재사용한다. `B=4`는 Stage-0 계획에 미리 적힌 진단값이다.
논문 test set에서는 별도 calibration data로 count를 정한 뒤 고정해야 한다.

전체 VideoMME로 전환한다. 이 옵션은 모든 원본 영상을 받고 strict export와 10,000회 bootstrap을 사용하므로 충분한 디스크와 실행 시간을 확보해야 한다.

```bash
bash scripts/run_a100_experiment.sh --full
```

다른 GPU를 선택하거나 BLIP batch를 낮춘다.

```bash
bash scripts/run_a100_experiment.sh \
  --cuda-device 1 \
  --feature-batch-size 16
```

MLVU/LVB도 같은 runner를 사용하지만 라이선스 동의가 필요한 대용량 원본 영상은 자동 다운로드하지 않는다. 승인된 dataset root를 mount한 뒤 실행한다.

```bash
bash scripts/run_a100_experiment.sh \
  --benchmark mlvu \
  --dataset-root /data/MLVU \
  --no-download-data
```

## 5. 주요 산출물

기본 위치는 `artifacts/videomme_stage0_20/`이다.

```text
analysis/summary.json             representation/salience/selection 결과와 CI
analysis/item_metrics.csv         논문 표·plot용 item-level 지표
keyframes/*.json                  method × origin keyframe annotation
mllm/videomme/*/origin*/          cell별 lmms-eval 결과와 console log
predictions.jsonl                 공식 parser 기반 7-field prediction grid
mllm_stability_summary.json       accuracy/agreement/robust accuracy와 paired CI
matched_cardinality/b04/          동일 경계 수 DWT/SWT 반사실적 전체 산출물
.stage0_state/*.done.json         검증 가능한 Stage-0 resume marker
```

세부 protocol, schema, full-run 전환 및 실패 복구 규칙은 [PHASE_STABLE_EXPERIMENTS.md](PHASE_STABLE_EXPERIMENTS.md)에 정리되어 있다.
