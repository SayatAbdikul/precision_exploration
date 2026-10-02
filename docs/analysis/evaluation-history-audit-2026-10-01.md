# Evaluation-history audit (2026-10-01, revision 2 after the independent check)

Question: which evaluation images has this project ever run a model on, and are the ImageNet 9k remainder and the COCO 4k remainder untouched?

Tool: `tools/analysis/evaluation_history_audit.py` (read-only, schema `evaluation-history-audit-2`). Result: `results/summaries/evaluation-history-audit-v1.json`. Test: `tests/unit/test_evaluation_history_audit.py` (synthetic tree).

This revision applies every finding of the independent check (`handoffs/L3-evaluation-history-review.md`): the first script skipped files over 8 MB, binary files, sqlite databases, three trees and the files directly under the two roots, and its patterns missed full paths and source names without the synset suffix. The verdict did not change; four statements in the first document were wrong and are corrected below.

Scan of this revision: started 2026-10-01 11:35 UTC, 270 s, 298,658 files, 21.7 GB, 0 unreadable files. Other lanes were still writing while it ran.

## Method

1. Coverage: every file under `artifacts/` and `results/` is read completely as bytes (32 MB chunks with 1 KB overlap, no size limit). That includes `.npz`, `.npy`, `.pt`, sqlite files, `.out`/`.err`/`.py`/`.sh`, files directly under `artifacts/` and `results/` (pseudo-trees `(root files)`), and the trees skipped before (`agent_orchestration`, `ppa`, `rtl`). Left out: only the script's own output file `results/summaries/evaluation-history-audit-v1.json` (it lists the 4,000 COCO remainder hashes, so reading it would count as a mention). Sqlite files are in addition queried read-only for every table with a `sample_id` or `image_id` column (3 databases have such a table; `phase3.sqlite` has 8,195 rows, `phase2*.sqlite` 0 rows; the four `shared-scale-cache` databases hold scale tables keyed by response hash, no image identity).
2. Frozen lists: `data/manifests/*.tsv`: ImageNet calibration 2k, screen 1k, evaluation 10k (remainder 9k); COCO calibration 2k, screen 1k, val 5k (remainder 4k). Calibration and evaluation lists share 0 hashes. Every image is keyed by content sha256.
3. Identity patterns (widened): the sha256 as a path part or anywhere in the bytes; `NNNN/<sha20>.JPEG` including inside a full path with a leading slash; `ILSVRC2012_val_NNNNNNNN` with or without the `_nXXXXXXXX` suffix (mapped by val number, unique over the 50k set); the ImageNet train name; COCO `NNNNNNNNNNNN.jpg`; `"image_id": N`. Evidence kinds: `stem`, `field`, `sqlite`, `mention` (a bare sha256; weakest).
4. Role of a file: any path part containing `calibration` is calibration; tree `dataset_indexes` is index; the rest is evaluation. A crossing is an evaluation-role file naming a calibration-list image, or a calibration-role file naming an evaluation-list image.
5. Record class (new; decided per file, reported separately, `by_record_class` per tree and `touched.*` per union):
   - `non_run`: an archive offset index (tree `dataset_indexes`: records name, start, size of tar members) or a manifest copy (a path part contains `manifest`, or the file starts with a manifest header). Such files list images, they say nothing about a model.
   - `run_record`: the image identity is the file name (a per-image output file), or a sqlite per-image row, or the same file carries per-image outcome keys (`top1`, `logits`, `prediction(s)`, `detections`, `bbox`, `score`, `category_id`, `output`, `correct`, `loss`, `margin`, `metrics`).
   - `other_mention`: the identity occurs in any other file (plans, logs, summaries, test reports, binary files). It is not discounted: the strict verdict counts `run_record` plus `other_mention`, and anything not known to be `non_run` is treated as possible use.
   The rule is mechanical and keyword based. Its weak side is `other_mention`/`run_record` for summaries; its safe side is that only a definite index or manifest is ever discounted.
6. Model attribution is by model name in the path or by the configuration record, as before (`model_attribution` in the JSON).

## Result: images touched, by list

Evaluation-role evidence. Counts are identical with and without weak mentions.

| List | Run record or other mention | Of | Digest of sorted touched hashes |
|---|---|---|---|
| ImageNet screen 1k | 1,000 | 1,000 | `d8acc5d45f1e...` |
| ImageNet remainder 9k | 0 | 9,000 | e3b0c442... (empty) |
| ImageNet calibration 2k | 1 (not a run, see crossings) | 2,000 | `f693edf1b125...` |
| COCO screen 1k | 1,000 | 1,000 | `ffde1ffab4a3...` |
| COCO remainder 4k | 4,000 | 4,000 | `01fd777fbf39...` |
| COCO calibration 2k | 0 | 2,000 | empty |

Run records versus non-run mentions, same lists:

| List | `run_record` | `other_mention` | `non_run` (index or manifest copy) |
|---|---|---|---|
| ImageNet screen 1k | 1,000 | 1,000 (summaries, `per_image_predictions` jsonl, phase3 job files) | 1,000 |
| ImageNet remainder 9k | 0 | 0 | 9,000 (all in `artifacts/dataset_indexes`) |
| COCO screen 1k | 1,000 | 1 (phase3) | 0 |
| COCO remainder 4k | 4,000 (YOLOv8n FP32 files in `per_image_predictions`) | 0 | 0 |

So the 9,000 ImageNet remainder names that do occur are all archive offset indexes (`artifacts/dataset_indexes/imagenet-val-*.jsonl`, 87 files, which name 50,000 validation members: the 10k list plus 40,000 names outside every list). They are not model runs and carry no model output. The strict ImageNet remainder count is 0 with and without mentions.

Calibration-role evidence: ImageNet 2,000 of 2,000 and COCO 2,000 of 2,000 calibration images (digests `124699589f96...` and `92703df5839f...`), none outside the calibration lists.

Per tree, evaluation role (screen / remainder; ImageNet and COCO). Every remainder count is 0 except the COCO baseline row. Equal to the checker's independent recount.

| Tree | ImageNet screen | ImageNet remainder | COCO screen | COCO remainder | Models |
|---|---|---|---|---|---|
| experiment_b | 1,000 | 0 | 0 | 0 | resnet18, mobilenet_v2, mobilenet_v3_large; 3 prediction dirs without configuration record |
| experiment_b_ext | 1,000 | 0 | 1,000 | 0 | same plus yolov8n; 6 dirs without configuration record |
| experiment_b2 | 1,000 | 0 | 0 | 0 | the three classifiers (still being written) |
| breadth_study | 256 | 0 | 0 | 0 | resnet18 (jobs), unattributed |
| phase3 | 1,000 | 0 | 1,000 | 0 | three classifiers, yolov8n (COCO: `phase3/baselines/yolov8n-*.json`, 15.8 MB, 1,000 per-image records; the first version of this audit gave 8 because of its 8 MB limit) |
| exact_execution_v2 | 8 | 0 | 0 | 0 | resnet18 |
| fixed_image_runs | 8 | 0 | 0 | 0 | classifiers |
| workload_runs | 0 | 0 | 8 | 0 | yolov8n |
| scaled_bridge_v1 | 128 | 0 | 0 | 0 | resnet18 (inferred) |
| scaled_bridge_v2 | 128 | 0 | 0 | 0 | resnet18 |
| scaled_bridge_gap_v1 | 1,000 | 0 | 0 | 0 | resnet18 (inferred) |
| per_image_predictions | 1,000 | 0 | 1,000 | 4,000 | FP32 baselines only |
| results/databases | 1,000 | 0 | 1 | 0 | `phase3.sqlite` per_image_results: 1,000 ImageNet screen, 1 COCO screen image |
| results/summaries | 16 | 0 | 8 | 0 | fixed-image and diagnosis files (plus 984 screen names as weak mentions) |
| dataset_indexes (index role) | 1,000 | 9,000 | 0 | 0 | not a run: archive offset indexes |

Calibration trees: `calibration_observations` (2,000 per classifier), `experiment_b/calibration` (2,000), `experiment_b_ext/calibration` (COCO 2,000), `breadth_study/rounding_calibration_v1` (512). All hit only calibration-list images.

Crossings: calibration-role files naming evaluation-list images: 0. Evaluation-role files naming a calibration-list image: 1 ImageNet calibration image, in `artifacts/conformance/phase2/pytest-cpu.xml`, by a full path from another machine (`/home/graff145/...`) inside a file-not-found message. It is a failed test lookup, not a model run on that image; it is a calibration image, so the evaluation lists are not affected. (The two ImageNet calibration images named in `phase3` and `results/summaries/phase2-calibration-cache-verification.json` are calibration-role files by the path rule and are weak mentions only.)

Outside every list: the scan finds names that are in no frozen list. `artifacts/dataset_indexes` holds 40,000 ImageNet validation names outside the 10k list (the other members of the 50k archive; they are index entries, not runs). `artifacts/agent_orchestration` names one such image (`ILSVRC2012_val_00023515`, a line of the task text of a lane). Nothing outside the lists appears in any run record: no unknown `relative_path`, COCO `file_name` or `image_id`, and no unknown sqlite `sample_id`, outside these two trees. The earlier statement "outside every list: none" was wrong. The dropped label-77 duplicate path `0077/4a4ce194...` is not named anywhere; its content hash equals the kept twin `0072/4a4ce194...`, in the 9k remainder, which has no run record.

## Which baselines and models saw which lists

- ImageNet FP32 baselines (resnet18, mobilenet_v2, mobilenet_v3_large; `per_image_predictions/*imagenet1k_screen.jsonl`, `reproduction`, `verification`; `results/summaries/*_fp32_imagenet1k_screen.json`): screen 1k only.
- YOLOv8n FP32 (`per_image_predictions/yolov8n_coco2017_val5k*`, `phase2-yolov8n-*`, `reproduction/yolov8n_coco_predictions.json`, `yolo-validation/coco2017-val5k`): all 5,000 COCO val images (1,000 screen plus the whole 4k remainder), 10 files or groups in 4 distinct contents, dated 2026-09-07 to 09-11. `phase2-yolov8n-fp32-cocoeval.json` has `image_count` 5000. The checker's review of the code found that full val2017 goes through `public/workloads/models/yolo_eval.py` (Ultralytics FP32) and that `tools/phase3/baselines.py` reads the 5k predictions but keeps only the screen images.
- Quantised or candidate runs, all trees: ImageNet screen only; COCO screen only (`experiment_b_ext` 1,000, `phase3` baseline record 1,000, `workload_runs` 8). The only code path that accepts `imagenet_evaluation_10k` or `coco_evaluation_5k` is `public/inference/workload_job.py`; its records show `coco_screen_1k`, first eight images. The list names `imagenet_evaluation_10k`, `coco_evaluation_5k` occur only in freeze, verification and environment files, never in a run configuration (checker finding).
- Calibration lists: calibration only.

## What was checked beyond the script (by the independent checker, not by this script)

- `backups/research-checkpoint-20260925T182113Z`: 49,175 files under `artifacts/` and `results/` in its manifest, all still in the live tree; 49,170 byte-identical, the 5 that differ were extracted and contain no remainder identity. Clean.
- Git history (`git log --all -p`, 15 commits): remainder hashes only in the two manifests and the selection record. Clean.
- `cache/` holds only the annotations zip, a font list and Ultralytics settings. File access times show nothing has opened remainder image files since about 2026-09-26 23:30 except the 200-file hash spot check at 15:57 today.
- Direct search (`grep -a -F`) of every file for the remainder hash prefixes and names, and for COCO ids: same result as this script.
These are one-off checks; this script does not repeat them.

## Verdict

1. ImageNet 9k remainder: untouched as far as any record on this machine shows. 0 of 9,000 hashes occur in any run record or other mention in `artifacts/` and `results/`; the only occurrences are the 9,000 names in archive offset indexes (not runs). Within the limits below it can serve as the interim held-out set.
2. COCO 4k remainder for candidate configurations: untouched. Candidates used at most the screen 1k.
3. COCO 4k remainder for baselines: already used. YOLOv8n FP32 predictions cover all 5,000 images, so FP32 detector results on the 4k are known; any held-out claim must be stated for candidates only, and anything defined relative to the FP32 baseline on the 4k (images or thresholds chosen by baseline behaviour) is contaminated.
4. Calibration/evaluation crossing: no run. One failed-test path message names a calibration image (see above).

What to exclude or disclose:
- Disclose that the YOLOv8n FP32 baseline was scored on all 5,000 COCO images; do not tune against its 4k predictions.
- Treat the ImageNet screen 1k and COCO screen 1k as development data, out of the confirmation numbers.
- Content duplicates: remainder image `0072/4a4ce194...` has a byte-identical twin dropped from the 10k at label 77. Keep the hash-level check when the 50k set arrives.
- ImageNet FP32 baselines on the remainder do not exist yet; they will be new data.

## Limits (what a record-based audit cannot show)

- It sees records, not intent. A model run that left no file under `artifacts/` or `results/` is invisible to this script and to the checker; the backup (from 2026-09-25) and git history were checked and show none, but a run that wrote nothing before 2026-09-26 cannot be excluded.
- Second machine: the phase 3 integer screening was run on another machine (`phase3_other_device_commands.txt`; the `/home/graff145/...` path in the conformance report). Its commands use the screen lists, but its records are not on this machine. "Untouched" therefore means "on this machine, as far as the records here show"; ask the other machine's owner or copy its records here and rerun.
- COCO images named only by a bare number outside an `image_id` key or a sqlite `sample_id`/`image_id` column are not found. Binary arrays are searched as bytes for text identities only; an array of numeric image ids would not be recognised (the checker read all npz array headers: no image lists, only calibration activation samples).
- The record-class rule is keyword based (see Method 5). It discounts only archive indexes and manifest copies.
- Other lanes keep writing on the screen 1k. Rerun just before any confirmation run (about 5 minutes on 16 workers) and compare `touched_digest` for `imagenet/remainder_9k` and `coco/remainder_4k`, and the counts in `touched.evaluation_run_record_strict`. The ImageNet remainder digest in the evaluation unions must stay `e3b0c442...`; `index_mentions` (9,000) is expected.

```
.venv/bin/python tools/analysis/evaluation_history_audit.py --out results/summaries/evaluation-history-audit-v1.json
.venv/bin/python -m pytest -q tests/unit/test_evaluation_history_audit.py
```
