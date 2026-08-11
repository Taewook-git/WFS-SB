# Phase-Stable WFS-SB 실험 실행 가이드

이 문서는 현재 저장소에 구현되어 있는 `phase_stable` 명령만 사용해 다음 두 protocol을 실행하는 방법을 설명한다.

- Protocol B: 동일 비디오에 여러 random sampling origin을 적용한 DWT 대 SWT 비교
- Protocol A: 하나의 고정 ITM signal을 circular shift한 controlled operator 비교

모든 명령은 WFS-SB 저장소 루트에서 Windows PowerShell로 실행한다고 가정한다. Linux에서도 `python -m phase_stable ...` 인자 자체는 동일하지만 경로와 환경 변수 문법은 바꿔야 한다.

A100 Linux 서버에서 설치부터 전체 Stage-0/MLLM 평가까지 한 번에 실행하려면 [A100_QUICKSTART.md](A100_QUICKSTART.md)와 `bash scripts/run_a100_experiment.sh`를 사용한다.

## 1. 현재 제공되는 명령

다음 명령은 `python -m phase_stable --help`에 실제 등록되어 있다.

```text
make-manifests
make-benchmark-manifests
preprocess-benchmark
analyze-signals
selection-baselines
matched-boundaries
export-keyframes
controlled-shifts
evaluate-predictions
```

이 가이드의 주 workflow는 benchmark annotation을 직접 읽는 `make-benchmark-manifests`를 사용한다. `make-manifests`는 이미 `video_id`와 `duration_sec`를 가진 별도 video JSONL이 있을 때 쓰는 저수준 명령이다.

## 2. 환경과 의존성

권장 Python은 기존 WFS-SB 환경과 같은 3.10이다.

```powershell
Set-Location C:\path\to\WFS-SB

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-phase-stable.txt
```

`requirements.txt`는 BLIP/CLIP feature extractor와 PyTorch 계층이고, `requirements-phase-stable.txt`는 NumPy, SciPy, PyWavelets, scikit-learn, PyAV, Pillow, PyYAML 및 테스트 계층이다. GPU preprocessing을 할 경우 설치된 PyTorch가 로컬 CUDA와 맞는지 별도로 확인한다.

```powershell
python -c "import av, numpy, pywt, scipy, sklearn, torch, transformers, yaml; print('dependencies: OK'); print('cuda:', torch.cuda.is_available())"
python -m phase_stable --help
```

### lmms-eval 준비

저장소의 `lmms-eval-diff`는 실행 가능한 checkout이 아니라 patch와 수정본 snapshot이다. 아직 `lmms-eval` 디렉터리가 없다면 정확한 base commit에 patch를 적용한다.

```powershell
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
Set-Location .\lmms-eval
git checkout bb1ebe76e7a942386c25c4664f902e0e59e8a401
git apply ..\lmms-eval-diff\lmms_eval_wfs.patch
python -m pip install -e .
Set-Location ..
```

이 patch가 있어야 `--data_files`, `use_keyframe=True`, `keyframe_indices` 기반 decoding이 동작한다.

## 3. 데이터 위치

기본 benchmark별 위치는 다음과 같다.

| benchmark 인자 | annotation | video root | lmms task | `data_files` split |
|---|---|---|---|---|
| `videomme` | `datasets/videomme/videomme_json_file.json` | `datasets/videomme/data` | `videomme` | `test` |
| `mlvu` | `datasets/mlvu/mlvu_dev.json` | `datasets/mlvu/video` | `mlvu_dev` | `test` |
| `lvb` | `datasets/longvideobench/lvb_val.json` | `datasets/longvideobench/videos` | `longvideobench_val_v` | `validation` |

`longvideobench`는 phase-stable CLI에서 `lvb`의 alias다. Export filename prefix는 정규화된 `lvb`가 된다.

## 4. 설정 파일의 적용 범위

주 설정은 `configs/phase_stable_icassp.yaml`이다.

```powershell
python -c "from phase_stable.config import load_phase_stable_config; c=load_phase_stable_config('configs/phase_stable_icassp.yaml'); print(c)"
```

중요한 적용 규칙은 다음과 같다.

- `analyze-signals --config ...`는 YAML의 `experiment`와 `selection` section을 사용한다.
- `--config`를 쓸 때 transform/selection 관련 CLI 기본값은 YAML 값으로 대체된다.
- bootstrap의 `--baseline-method`, `--treatment-method`, `--n-bootstrap`, `--confidence`, `--seed`, `--bootstrap-metrics`는 YAML 밖의 CLI 인자다.
- YAML의 `sampling`과 `metadata` section은 현재 `make-benchmark-manifests`에 자동 전달되지 않는다.
- 따라서 sampling seed, origin 수, FPS는 manifest 명령에 반드시 명시한다.
- `preprocess-benchmark`의 decoder는 현재 PyAV로 고정되어 있다. 같은 거리의 두 PTS가 있으면 앞 frame을 선택한다.

현재 ICASSP config의 주 비교는 `dwt` 대 undecimated `swt`, db4, 1 FPS, 5 origins, frame budget 16이다. `edge_margin_sec`는 Stage-0용으로 0이다.

## 5. 20-video Stage-0: Protocol B 전체 순서

아래 예시는 VideoMME annotation 순서에서 처음 20개의 고유 video를 사용한다. `--video-indices`는 QA row가 아니라 `load_benchmark_videos`가 만든 고유 video list의 index다. 별도로 선정한 20개가 있다면 `0 ... 19` 대신 그 index를 넣는다.

### 5.1 실행 경로 설정

```powershell
$Repo = (Get-Location).Path
$DatasetRoot = Join-Path $Repo "datasets\videomme"
$Questions = Join-Path $DatasetRoot "videomme_json_file.json"
$Config = Join-Path $Repo "configs\phase_stable_icassp.yaml"
$Run = Join-Path $Repo "artifacts\videomme_stage0_20"
$Manifests = Join-Path $Run "sampling_manifests.jsonl"
$Catalog = Join-Path $Run "video_catalog.jsonl"
$PreprocessDir = Join-Path $Run "preprocess"
$Signals = Join-Path $Run "origin_signals.jsonl"
$AnalysisDir = Join-Path $Run "analysis"
$BaselineDir = Join-Path $Run "baselines"
$KeyframeDir = Join-Path $DatasetRoot "keyframe_dir\phase_stable_stage0_20"

New-Item -ItemType Directory -Force -Path $Run, $KeyframeDir | Out-Null
```

### 5.2 Sampling manifest 생성

```powershell
python -m phase_stable make-benchmark-manifests `
  --benchmark videomme `
  --questions-file $Questions `
  --dataset-root $DatasetRoot `
  --output $Manifests `
  --catalog-output $Catalog `
  --video-indices 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 `
  --seed 20260810 `
  --num-origins 5 `
  --sample-fps 1.0
```

VideoMME의 `duration`은 `short/medium/long` category라서 실제 초가 아니다. 이 경우 명령은 PyAV로 각 video duration을 probe한다. `--no-probe-missing-duration`을 붙이면 VideoMME에서는 실패한다.

생성 직후 video 수와 origin 수를 확인한다.

```powershell
python -c "from phase_stable.sampling import read_manifests_jsonl; m=read_manifests_jsonl(r'$Manifests'); print('videos=', len(m)); print('origins=', sorted({x.num_origins for x in m})); print('min candidates=', min(x.candidate_count for x in m))"
```

`min candidates`가 16보다 작으면 K=16 실험에 사용할 수 없다. baseline은 즉시 실패하고, WFS trace도 export 시 `--expected-budget 16` 검증을 통과하지 못한다.

### 5.3 Origin별 frame decoding과 BLIP-2 ITM preprocessing

```powershell
python -m phase_stable preprocess-benchmark `
  --benchmark videomme `
  --questions-file $Questions `
  --dataset-root $DatasetRoot `
  --manifests $Manifests `
  --output-dir $PreprocessDir `
  --signal-jsonl $Signals `
  --feature-model blip2 `
  --device cuda `
  --batch-size 32 `
  --frame-buffer-size 256
```

GPU memory가 부족하면 `--batch-size`를 낮춘다. `--frame-buffer-size`는 GPU가 아니라 decode 중 host RAM에 유지할 RGB target 수를 제한한다. A100 80GB 한 장의 BLIP-2 ViT-g에는 batch 32, frame buffer 256을 보수적 기본값으로 사용한다. 원본이 4K이거나 host RAM이 작으면 frame buffer부터 64로 낮춘다. CPU smoke run은 `--device cpu`로 가능하지만 전체 BLIP-2 preprocessing은 매우 느리다.

이 단계는 한 video의 모든 origin target을 한 번의 sequential PyAV decode로 찾는다. 원해상도 RGB 전체를 쌓지 않고 bounded buffer로 extractor에 전달하며, visual feature는 임시 NumPy memmap에 바로 기록한다. 같은 decode batch는 그 video의 모든 question이 공유한다. 각 query-origin에 대해 ITM score와 visual feature를 저장한다.

기본 checkpoint 대신 로컬 또는 고정 revision 경로를 쓰려면 다음 인자를 추가한다.

```text
--model-path C:\path\to\blip2-checkpoint
```

### 5.4 DWT 대 SWT signal/segmentation/selection 분석

```powershell
python -m phase_stable analyze-signals `
  $Signals `
  $AnalysisDir `
  --config $Config `
  --baseline-method dwt `
  --treatment-method swt `
  --n-bootstrap 1000 `
  --confidence 0.95 `
  --seed 20260810
```

Stage-0에서는 빠른 검증을 위해 bootstrap 1,000회를 사용한 예다. 논문용 최종 실행에서는 `--n-bootstrap 10000`을 사용한다. `--config`가 있으므로 frame budget, transform, padding, boundary tolerance와 WFS selection weight는 YAML에서 읽는다.

완료 조건:

```powershell
Get-Item `
  (Join-Path $AnalysisDir "traces.jsonl"), `
  (Join-Path $AnalysisDir "item_metrics.jsonl"), `
  (Join-Path $AnalysisDir "item_metrics.csv"), `
  (Join-Path $AnalysisDir "summary.json"), `
  (Join-Path $AnalysisDir "manifest\run_manifest.json"), `
  (Join-Path $AnalysisDir "manifest\environment.json")
```

### 5.5 Matched-cardinality boundary 진단

TI-DWT가 boundary 수를 줄여서 안정적으로 보이는 효과를 분리하려면 두 method에 calibration split에서 미리 정한 동일 top-B를 적용한다. Stage-0 명령 smoke에서는 고정 `B=4`를 다음처럼 실행할 수 있다.

```powershell
python -m phase_stable matched-boundaries `
  --traces (Join-Path $AnalysisDir "traces.jsonl") `
  --output (Join-Path $AnalysisDir "matched_boundary_metrics.jsonl") `
  --count 4 `
  --tolerance-sec 1.0 `
  --edge-margin-sec 0.0
```

논문 test split에서는 test origin 결과를 보고 B를 다시 고르면 안 된다. 별도 calibration split에서 video별 count를 결정해 다음 형태의 JSON object를 준비한다.

```json
{
  "001": 4,
  "videomme/002": 5
}
```

그다음 `--count` 대신 `--counts-json`을 사용한다.

```powershell
python -m phase_stable matched-boundaries `
  --traces (Join-Path $AnalysisDir "traces.jsonl") `
  --output (Join-Path $AnalysisDir "matched_boundary_metrics.jsonl") `
  --counts-json (Join-Path $Run "calibrated_boundary_counts.json") `
  --tolerance-sec 1.0 `
  --edge-margin-sec 0.0
```

count key는 `video_id` 또는 `dataset/video_id`다. 현재 CLI는 count를 calibration data에서 자동 추정하지 않는다. 출력은 query-method별 `matched_boundary_f1_mean/worst`, mean displacement, matched segment ARI/VI를 담은 JSONL이다.

동일 boundary count가 최종 keyframe 선택과 MLLM에 미치는 영향을 검사하려면
기존 signal/feature artifact에서 post-transform selection 전체를 다시 실행한다.

```powershell
python -m phase_stable matched-selection `
  $Signals (Join-Path $Run "matched_cardinality\b04\analysis") `
  --config .\configs\phase_stable_icassp.yaml `
  --count 4 `
  --n-bootstrap 1000 `
  --seed 20260810
```

이 경로는 DWT/SWT 모두 saliency 상위 `B`개를 동일한 deterministic NMS로
선택해 정확히 `B+1`개 segment를 만든다. 이후 segment importance, filtering,
K=16 allocation과 MMR은 원 pipeline을 그대로 사용한다. trace method는
`dwt_matched`, `swt_matched`이며 기존 adaptive 결과와 섞이지 않는다. A100에서
완료된 Stage-0 이후 export부터 Qwen 평가까지 한 번에 실행하려면 다음을 쓴다.

```bash
bash scripts/run_a100_experiment.sh \
  --skip-bootstrap --no-download-data \
  --matched-only --matched-count 4
```

산출물은 `artifacts/<run>/matched_cardinality/b04/` 아래에 격리된다. 현재
20-video 결과에 대한 이 재실험은 원인 진단용이며, 논문 confirmatory test에는
별도 calibration subset에서 고정한 count만 사용한다.

### 5.6 Uniform과 Top-K selection baseline

```powershell
python -m phase_stable selection-baselines `
  $Signals `
  $BaselineDir `
  --methods uniform topk `
  --frame-budget 16 `
  --selected-tolerance-sec 1.0
```

이 명령은 transform representation이나 boundary를 만들지 않고 동일 origin candidate grid에서 selection stability만 계산한다.

### 5.7 lmms-eval keyframe annotation export

DWT/SWT trace를 export한다.

```powershell
python -m phase_stable export-keyframes `
  --traces (Join-Path $AnalysisDir "traces.jsonl") `
  --benchmark videomme `
  --questions-file $Questions `
  --dataset-root $DatasetRoot `
  --output-dir $KeyframeDir `
  --methods dwt swt `
  --origin-ids 0 1 2 3 4 `
  --expected-budget 16 `
  --allow-partial
```

Uniform/Top-K도 같은 디렉터리에 export한다.

```powershell
python -m phase_stable export-keyframes `
  --traces (Join-Path $BaselineDir "baseline_traces.jsonl") `
  --benchmark videomme `
  --questions-file $Questions `
  --dataset-root $DatasetRoot `
  --output-dir $KeyframeDir `
  --methods uniform topk `
  --origin-ids 0 1 2 3 4 `
  --expected-budget 16 `
  --allow-partial
```

20-video subset trace를 전체 VideoMME annotation과 결합하므로 Stage-0에서만 `--allow-partial`이 필요하다. Full benchmark run에서는 이 flag를 빼서 모든 annotation row의 누락과 중복을 strict하게 검증한다.

생성 파일은 다음과 같다.

```text
videomme_dwt_origin00.json ... videomme_dwt_origin04.json
videomme_swt_origin00.json ... videomme_swt_origin04.json
videomme_uniform_origin00.json ... videomme_uniform_origin04.json
videomme_topk_origin00.json ... videomme_topk_origin04.json
```

각 파일은 원래 benchmark annotation row를 보존하면서 `selected_source_frame_indices`를 `keyframe_indices`로 주입한 JSON list다.

### 5.8 method × origin lmms-eval

Linux/A100에서는 검증·재개·prediction 병합을 포함한 runner를 사용한다.

```bash
bash scripts/run_mllm_grid.sh \
  --benchmark videomme \
  --keyframe-dir artifacts/videomme_stage0_20/keyframes \
  --output-root artifacts/videomme_stage0_20/mllm \
  --methods dwt,swt \
  --origins 0,1,2,3,4 \
  --predictions-output artifacts/videomme_stage0_20/predictions.jsonl
```

다음 loop는 4 methods × 5 origins를 각각 독립 output directory에서 실행한다. 우선 DWT/SWT만 실행하려면 `$Methods`를 `@("dwt", "swt")`로 줄인다.

```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
$env:QWEN_CKPT = "Qwen/Qwen2.5-VL-7B-Instruct"
$Methods = @("uniform", "topk", "dwt", "swt")
$Task = "videomme"
$Split = "test"
$RelativeKeyframeDir = "keyframe_dir/phase_stable_stage0_20"
$ModelArgs = "max_num_frames=16,use_keyframe=True,pretrained=$env:QWEN_CKPT,max_pixels=200704,attn_implementation=sdpa,interleave_visuals=False"

foreach ($Method in $Methods) {
  foreach ($Origin in 0..4) {
    $OriginToken = "{0:D2}" -f $Origin
    $FileName = "videomme_${Method}_origin${OriginToken}.json"
    $DataFiles = @{ $Split = "$RelativeKeyframeDir/$FileName" } | ConvertTo-Json -Compress
    $OutputPath = Join-Path $Run "mllm\$Method\origin$OriginToken"

    python -m lmms_eval `
      --model qwen2_5_vl `
      --tasks $Task `
      --model_args $ModelArgs `
      --batch_size 1 `
      --output_path $OutputPath `
      --log_samples `
      --data_files $DataFiles

    if ($LASTEXITCODE -ne 0) {
      throw "lmms-eval failed: method=$Method origin=$Origin"
    }
  }
}
```

`200704 = 256 × 28²`는 Qwen의 256 visual-token 설정이다. 16-frame
SDPA를 40 GiB A100 MIG에서 실행할 때 모든 method/origin cell에 이 값을
고정한다. 더 큰 spatial cap은 Transformers 4.49 vision SDPA의 dense
attention 메모리를 크게 늘리므로, 한 run 안에서 값을 혼용하지 않는다.

이 명령은 WFS-SB patch의 local task YAML이 저장소 루트의 `datasets/...`를 찾으므로 반드시 저장소 루트에서 실행한다. `sdpa` 대신 FlashAttention 2를 쓰려면 호환 wheel을 별도 설치한 뒤 `attn_implementation=flash_attention_2`로 바꾼다. 모든 cell에서 checkpoint, model args, task, prompt와 decoding 설정을 동일하게 유지한다.

다른 benchmark의 loop 변경점:

| benchmark | filename prefix | `$Task` | `$Split` | relative keyframe directory 예시 |
|---|---|---|---|---|
| VideoMME | `videomme` | `videomme` | `test` | `keyframe_dir/...` under `datasets/videomme` |
| MLVU | `mlvu` | `mlvu_dev` | `test` | `keyframe_dir/...` under `datasets/mlvu` |
| LVB | `lvb` | `longvideobench_val_v` | `validation` | `keyframe_dir/...` under `datasets/longvideobench` |

## 6. Prediction JSONL과 downstream 평가

`run_mllm_grid.sh`는 각 완료 marker의 keyframe JSON과 sample JSONL을 `doc_id`로 결합하고, benchmark의 공식 `process_results`가 저장한 parser 결과를 사용해 `predictions.jsonl`을 자동 생성한다. raw model response를 별도로 재해석하지 않는다. 기존 완료 grid를 수동 병합하려면 다음 명령을 사용한다.

```bash
python scripts/convert_lmms_logs.py \
  --grid-root artifacts/videomme_stage0_20/mllm/videomme \
  --benchmark videomme \
  --methods dwt,swt \
  --origins 0,1,2,3,4 \
  --output artifacts/videomme_stage0_20/predictions.jsonl
```

한 줄의 필수 필드는 다음 일곱 개다.

```json
{"dataset":"videomme","video_id":"001","question_id":"001-1","origin_id":0,"method":"dwt","prediction":"C","gold":"C"}
```

규칙:

- `dataset`, `video_id`, `question_id`는 trace와 같은 identifier를 사용한다.
- `origin_id`는 sampling manifest의 0-based integer다.
- `method`는 keyframe JSON을 만든 method 이름이다.
- `prediction`과 `gold`는 같은 normalization을 적용한 label이어야 한다.
- 각 method는 모든 item에 대해 같은 origin grid를 가져야 한다.
- 같은 method/item/origin row가 중복되면 실패한다.
- 한 item의 gold가 origin에 따라 바뀌면 실패한다.
- 기본 비교는 `dwt`와 `swt`이며 두 method의 item, origin, gold가 정확히 맞아야 한다.
- `raw_prediction`, `keyframe_json`, `lmms_result_path` 같은 추가 필드는 허용되며 evaluator는 무시한다.

준비한 JSONL을 평가한다.

```powershell
$Predictions = Join-Path $Run "predictions.jsonl"
$PredictionSummary = Join-Path $Run "prediction_summary.json"

python -m phase_stable evaluate-predictions `
  $Predictions `
  $PredictionSummary `
  --baseline-method dwt `
  --treatment-method swt `
  --n-bootstrap 1000 `
  --confidence 0.95 `
  --seed 20260810
```

최종 논문 run에서는 `--n-bootstrap 10000`을 사용한다. 출력에는 method별 다음 값이 포함된다.

- `mean_accuracy`
- `robust_accuracy`
- `worst_origin_accuracy`
- `accuracy_std`
- `answer_agreement`
- `pairwise_answer_disagreement`
- origin별 accuracy

`comparison.effect_order`는 다음 순서다.

```text
delta_mean_accuracy
delta_robust_accuracy
delta_pairwise_answer_disagreement
```

paired confidence interval의 cluster 단위는 `video_id`다.

## 7. Protocol A: controlled circular shifts

`controlled-shifts`는 RGB frame이나 real origin을 바꾸지 않고 하나의 고정 1-D signal에 circular shift를 적용한 뒤 output을 inverse-align한다. 입력은 NumPy `.npy` 한 개다. 기본 동작은 wavelet filter의 effective support, 실제 decomposition level과 최대 shift를 이용해 양쪽 edge crop을 자동 계산하고, representation·saliency·energy metric을 공통 interior에서만 계산한다.

먼저 Protocol B signal artifact에서 origin 0의 한 item을 선택해 `.npy`로 저장한다.

```powershell
$env:PHASE_SIGNALS = $Signals
$env:CONTROLLED_SIGNAL = Join-Path $Run "protocol_a_signal.npy"

@'
import os
import numpy as np
from phase_stable.artifacts import read_signal_records

records = read_signal_records(os.environ["PHASE_SIGNALS"])
record = next(row for row in records if row.origin_id == 0)
np.save(os.environ["CONTROLLED_SIGNAL"], np.asarray(record.relevance_scores, dtype=float))
print(record.dataset, record.video_id, record.question_id, len(record.relevance_scores))
'@ | python -
```

16개 shift에서 DWT/SWT consistency를 측정한다.

```powershell
python -m phase_stable controlled-shifts `
  $env:CONTROLLED_SIGNAL `
  (Join-Path $Run "protocol_a_metrics.json") `
  --methods dwt swt `
  --wavelet db4 `
  --drift-level 3 `
  --shared-padding `
  --shifts 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
```

db4 filter length를 `L`, 실제 level을 `J`, 최대 절대 shift를 `S`라고 하면 기본 crop은 effective support를 반영한 다음 값에서 계산된다.

```text
effective_support = 1 + (L - 1) * (2**J - 1)
edge_samples = ceil(effective_support / 2) + S
```

짧은 signal에서는 최소 두 interior sample이 남도록 제한된다. 계산된 값은 각 method metric의 `edge_samples`에 저장된다. preregistered crop을 강제로 쓸 때만 `--edge-samples N`으로 override하며, `2*N`이 signal 길이 이상이면 명령이 실패한다.

출력은 cropped valid interior에서 계산한 method별 representation/saliency cosine mean·worst, normalized L1 mean·worst, scale-energy L1 drift mean·worst를 포함한다.

현재 CLI는 한 `.npy` signal씩 평가하며 여러 video-question의 Protocol A 결과를 자동 집계하는 명령은 없다. 여러 item을 논문 통계로 사용할 경우 public Python API인 `phase_stable.analysis.controlled_shift_metrics`를 item별로 호출해 별도 row를 만들고 video-level로 집계해야 한다.

## 8. Full benchmark 실행으로 전환

Stage-0가 통과하면 새 run directory를 만들고 다음만 변경한다.

1. `make-benchmark-manifests`에서 `--video-indices ...`를 제거한다.
2. preprocessing과 analysis output을 Stage-0와 다른 directory에 쓴다.
3. `analyze-signals`와 `evaluate-predictions`에서 `--n-bootstrap 10000`을 사용한다.
4. full export에서는 `--allow-partial`을 제거한다.
5. DWT와 SWT의 full paired run을 먼저 끝낸 뒤 Uniform/Top-K 또는 추가 model/budget을 실행한다.

MLVU 예시의 첫 단계는 다음과 같다.

```powershell
python -m phase_stable make-benchmark-manifests `
  --benchmark mlvu `
  --questions-file datasets\mlvu\mlvu_dev.json `
  --dataset-root datasets\mlvu `
  --output artifacts\mlvu_full\sampling_manifests.jsonl `
  --catalog-output artifacts\mlvu_full\video_catalog.jsonl `
  --seed 20260810 `
  --num-origins 5 `
  --sample-fps 1.0
```

이후 순서는 Stage-0와 동일하게 `preprocess-benchmark → analyze-signals → matched-boundaries → selection-baselines → export-keyframes → lmms-eval → evaluate-predictions`다.

## 9. 산출물과 schema

권장 run tree는 다음과 같다.

```text
artifacts/<run_id>/
  sampling_manifests.jsonl
  video_catalog.jsonl
  origin_signals.jsonl
  preprocess/
    visual_features/
      o<origin>_<hash>.npy
    manifest/
      run_manifest.json
      environment.json
  analysis/
    traces.jsonl
    trace_arrays/
      o<origin>_<method>_<hash>.npz
    item_metrics.jsonl
    item_metrics.csv
    matched_boundary_metrics.jsonl
    summary.json
    manifest/
      run_manifest.json
      environment.json
  baselines/
    baseline_traces.jsonl
    baseline_item_metrics.jsonl
  mllm/
    <method>/origin<id>/...
  predictions.jsonl
  prediction_summary.json
  protocol_a_signal.npy
  protocol_a_metrics.json
```

### Sampling manifest JSONL

video당 한 row이며 주요 field는 다음과 같다.

```text
schema_version, video_id, master_seed, duration_sec,
sample_fps, period_sec, epsilon_sec, num_origins, candidate_count,
origins[{origin_id, origin_sec, target_timestamps_sec}]
```

모든 origin은 같은 `candidate_count`를 가진다.

### Video catalog JSONL

```text
dataset, video_id, video_path, duration_sec,
num_questions, question_ids
```

### Origin signal JSONL

query-origin당 한 row다.

```text
schema_version, dataset, video_id, question_id,
origin_id, origin_sec,
timestamps_sec, actual_pts_sec, source_frame_indices,
relevance_scores, pixel_hashes,
visual_features_path, metadata
```

`metadata`에는 query, gold/choice를 포함한 query metadata, video path, sample FPS, decode error, decoded frame index, source PTS, feature shape/dtype와 extractor revision이 들어간다.

### Analysis trace JSONL 및 NPZ

trace JSONL은 query-origin-method당 한 row다.

```text
dataset, video_id, question_id, origin_id, method,
timestamps_sec, actual_pts_sec, source_frame_indices,
array_path, visual_features_path,
peaks, peaks_sec, segments, valid_segments,
importance_scores, allocation,
selected_indices, selected_timestamps_sec,
selected_actual_pts_sec, selected_source_frame_indices,
used_fallback, transform, level, min_peak_distance
```

각 `array_path` NPZ에는 다음 dense array가 있다.

```text
relevance_scores
representation
coarse_detail
saliency
saliency_norm
scale_energy_proportions
```

`saliency_norm`은 `saliency / sum(saliency)`의 L1-mass 표현이며, all-zero detail이면 all-zero로 남겨 degenerate rate가 드러나게 한다. Operational peak detector는 원 WFS-SB와 동일한 adaptive threshold를 raw coarse detail에 적용한다.

### Item metrics와 summary

`item_metrics.jsonl/csv`는 dataset/video/question/method별로 모든 origin pair를 집계한 row다. representation, saliency, energy, boundary, segment, selected-frame consistency와 fallback rate가 포함된다.

`matched_boundary_metrics.jsonl`은 calibration-derived top-B를 강제했을 때의 boundary F1/distance와 segment ARI/VI를 query-method별로 기록한다.

`summary.json`은 config snapshot, descriptive aggregate, DWT-SWT paired video-cluster bootstrap, artifact path를 포함한다.

### Reproducibility manifest

`manifest/run_manifest.json`은 command, config, input absolute path/size/SHA-256를 기록한다. `environment.json`은 Python/platform, package version, CUDA/GPU, git commit과 dirty 여부를 기록한다.

### Exported keyframe JSON

benchmark의 공식 annotation JSON list와 같은 row를 유지하고 다음 field 하나를 덮어쓰거나 추가한다.

```json
{"keyframe_indices":[754,783,812,841]}
```

실제 값은 원본 video decoder frame index이며 chronological order의 정확한 K개여야 한다.

## 10. 실패, 검증, 재개 시 주의사항

- `phase_stable` 명령에는 현재 `--resume`이나 `--skip-existing`이 없다.
- manifest와 JSONL/NPZ write는 임시 파일을 거쳐 교체하지만, preprocessing이 중간 실패하면 이미 생성된 feature NPY와 미완성 `.tmp`가 남을 수 있다.
- preprocessing을 같은 directory에서 재실행하면 feature를 다시 계산한다. 부분 artifact를 자동 판별해 건너뛰지 않는다.
- analysis는 signal JSONL 전체를 메모리에 읽고 trace array를 순서대로 쓴다. 중간 실패 후 남은 `trace_arrays`만으로 완료로 판단하면 안 된다. `traces.jsonl`, metrics, summary와 manifest가 모두 있어야 완료다.
- 동일 output directory에 서로 다른 seed, FPS, K, config를 섞지 않는다. 새 `run_id` directory를 사용하는 것이 안전하다.
- 동일 output directory를 두 process가 동시에 쓰면 안 된다.
- `visual_features_path`와 `array_path`는 absolute path다. run directory를 옮기면 기존 signal/trace가 해당 파일을 찾지 못할 수 있다.
- `analyze-signals`는 item마다 origin이 최소 2개이고 candidate 수가 같아야 한다. duplicate origin도 거부한다.
- `selection-baselines`는 candidate 수가 K보다 작으면 실패한다.
- `export-keyframes` 기본 strict mode는 duplicate trace, 누락 annotation, 불완전 method×origin grid, budget 불일치, 중복·비정렬 source frame index를 거부한다.
- Stage-0 subset에만 `--allow-partial`을 사용한다. 이 flag도 malformed/duplicate trace를 허용하지는 않는다.
- lmms-eval은 keyframe 수가 `max_num_frames`와 다르면 assertion으로 실패한다. export에 `--expected-budget`을 항상 지정한다.
- lmms-eval은 method-origin마다 별도 output directory를 사용한다. 실패한 cell만 같은 명령으로 다시 실행할 수 있지만, 기존 결과 파일과 새 결과가 섞이지 않았는지 확인한다.
- prediction JSONL을 만들 때 raw free-form response가 아니라 benchmark parser와 같은 normalized choice label을 `prediction`에 넣는다.
- `evaluate-predictions`는 DWT/SWT의 item과 origin grid가 하나라도 다르면 paired 결과를 만들지 않는다. 먼저 row count와 key uniqueness를 검증한다.
- config의 `sampling` section을 바꾼 것만으로 기존 manifest는 바뀌지 않는다. 반드시 manifest command 인자와 output run directory도 함께 바꾼다.

## 11. 빠른 CLI smoke 확인

실제 GPU/video run 전에 모든 명령의 parser가 import되는지 확인한다.

```powershell
python -m phase_stable --help
python -m phase_stable make-benchmark-manifests --help
python -m phase_stable preprocess-benchmark --help
python -m phase_stable analyze-signals --help
python -m phase_stable selection-baselines --help
python -m phase_stable matched-boundaries --help
python -m phase_stable export-keyframes --help
python -m phase_stable evaluate-predictions --help
python -m phase_stable controlled-shifts --help
```

코드 단위 smoke test는 다음과 같다.

```powershell
python -m pytest tests -q
```
