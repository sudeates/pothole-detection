# Pothole detection — YOLOv8 + ByteTrack

**English** · [Türkçe](README.tr.md)

This repo has three parts: a YOLOv8n model that detects road potholes as a
single class (`pothole`), a ByteTrack-based display layer that keeps the boxes
stable on video, and a series of experiments that explains why the model
plateaus at ~0.72 mAP50.

![raw vs tracking](assets/demo_compare.gif)

*Left: every frame is predicted on its own. Right: tracking, a persistence
filter and duplicate-box suppression. The clip is 544×360 and was not used in
training.*

## Summary

| | Result |
| --- | --- |
| Model | YOLOv8n, 640 px, MWPD (reviewed-v1 labels) |
| Validation (249 images, 545 boxes) | P 0.726 · R 0.660 · **mAP50 0.716** · mAP50–95 0.331 |
| Test (97 images, 292 boxes, evaluated once) | P 0.763 · R 0.647 · **mAP50 0.715** · mAP50–95 0.353 |
| Speed | 15.1 ms/image (RTX 3050 Laptop, FP32, model only) |
| Video flicker | 532 → **47** box disappearances/min (tracking + filter) |

I ran five experiments (model capacity, architecture, label revision, extra
data) and tested six external models on the same validation set. Every
result landed between 0.54 and 0.73. The bottleneck is the data, not the
model. The 2,149 training images come from only 575 unique source scenes, and
pothole boundaries are hard to label consistently. Details below.

## Experiments

All results are on the MWPD validation split. "Train-time val" is the
mAP50 of the best epoch during training. "`eval.py`" is the same weights
re-measured with the standard protocol ([EVAL_PROTOCOL.md](EVAL_PROTOCOL.md)).
The validation score moves by about ±0.03 from epoch to epoch.

| Experiment | What changed | Train-time val mAP50 | `eval.py` mAP50 |
| --- | --- | ---: | ---: |
| **YOLOv8n baseline** | — | **0.729** | **0.716** |
| YOLOv8s | Model capacity | 0.715 | 0.698 |
| YOLOv8n, v2 labels | Label revision | 0.727 | — |
| YOLOv8n-P2 | Extra stride-4 detection head | 0.710 (stopped at ep48) | — |
| YOLOv8n, MWPD + HRP4K | +2,803 unique scenes (scale-matched crops) | 0.721 | — |

These were tried earlier and gave no lasting gain: 960/1280 input resolution,
TTA, NMS tuning, and SAHI. MWPD images are already 640×640, so SAHI covers each
image with a single window and changes nothing.

### Comparison with external models

I measured the six pretrained checkpoints from
[lukekratz/raspberry-pi-5-hailo-8-pothole-detection-system](https://github.com/lukekratz/raspberry-pi-5-hailo-8-pothole-detection-system)
with the same `eval.py` settings. Before loading any file, I statically
scanned its pickle contents with `tools/scan_pickle.py`.

| Model | mAP50 |
| --- | ---: |
| This repo, YOLOv8n baseline | **0.716** |
| YOLOv9t | 0.623 |
| YOLOv10n | 0.589 |
| YOLO11n | 0.581 |
| YOLOv8s* | 0.570 |
| YOLOv9s | 0.562 |
| YOLOv8n* | 0.539 |

\*These checkpoints have two classes. I merged both into the single `pothole`
class, so treat these numbers as indicative only. No external model beat the
baseline, so the ceiling is not specific to this training recipe.

## Why does the model stop at ~0.72?

**1. Most misses are low-confidence detections.** I analysed the validation
errors with `inspect_errors.py` (conf 0.25, IoU 0.5). Of 545 ground-truth
boxes, the model missed 171. It also produced 210 false positives.

- In 73% of the misses, the model does put a box in the right place, but its
  confidence stays below 0.25 (median 0.025). It never sees 18% of the missed
  potholes at all. In the remaining 9%, the box is shifted.
- I hand-reviewed a sample of misses (10 images, ~40 boxes), and ~85% of them
  are obvious potholes. The problem is not ambiguous labels. The model has not
  learned these scenes well enough.
- About half of the false positives are not model errors. Of 17 reviewed
  false positives, 4 are real potholes that were never labelled, 4 are correct
  potholes with a box of the wrong size, and 2 are puddles.

**2. Misses happen at every size.** The problem is not limited to small
objects. Box size below is box side divided by image side:

| Box size | <5% | 5–10% | 10–20% | 20–35% | ≥35% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Recall | 0.36 | 0.55 | 0.71 | 0.73 | 0.86 |
| Boxes | 11 | 120 | 183 | 126 | 105 |

This is why small-object fixes such as a P2 head and SAHI did not help.

**3. The training set is smaller than it looks.** The train split has 2,149
images. After removing Roboflow's augmentation copies (~3.7×), only
**575 unique source scenes** remain. Validation has 249 images from 197
scenes. No source scene appears in both train and validation, so there is no
leakage.

**4. More data did not solve it.** I added crops from HRP4K, a set of 6,003
unique 4K images, cut so that their box sizes match the MWPD distribution
(`make_hrp4k_scaled.py`). The score went from 0.729 to 0.721. The MWPD model
scores only 0.259 mAP50 on the HRP4K crops. The two datasets look very
different (HRP4K potholes are shallower and look like stains), so the extra
scenes do not carry over to MWPD.

**5. Low mAP50–95 comes from unclear boundaries.** The median IoU of true
positives is 0.74, and only 5% reach 0.9 or higher. A pothole's edge is not
clearly visible in the image. At strict IoU thresholds, labelling
consistency limits the score ([ANNOTATION_POLICY.md](ANNOTATION_POLICY.md)).

## Video demo

![tracking demo](assets/demo_track.gif)

`demo_video.py` processes a video in two modes:

- **raw:** Each frame is predicted on its own, conf 0.25.
- **track:** ByteTrack ([trackers/pothole_bytetrack.yaml](trackers/pothole_bytetrack.yaml))
  plus three display rules:
  - *Persistence:* A track is drawn only after it has been seen in at least
    5 frames. This removes single-frame false positives.
  - *Hold:* If a confirmed track disappears for up to 8 frames, its last box
    stays on screen (orange). This reduces flicker.
  - *Smoothing and duplicate suppression:* Box coordinates are smoothed with
    an EMA (α 0.5). When boxes overlap by 0.6 or more, measured relative to
    the smaller box, only one of them is drawn.

Results on an unlabelled 3.4-minute clip (5,121 frames):

| Mode | Frames with a box | Boxes/frame | Flicker (disappearances/min) |
| --- | ---: | ---: | ---: |
| raw | 87.4% | 2.55 | 532 |
| track + filter | 85.9% | 2.63 | 75 |
| track + filter + duplicate suppression | 85.9% | 2.19 | **47** |

The clip has no labels, so these numbers describe display stability, not
accuracy. Flicker counts how often a drawn box disappears in the next frame.
In raw mode, boxes in consecutive frames are matched at IoU ≥0.3. When
suppression hands a box over to another track ID, that hand-off does not
count as flicker.

## Known limitations

- **Distant and small potholes:** In low-resolution video, potholes that
  shrink to a few pixels are almost never detected.
- **Very large potholes that fill the frame:** The training data has few of
  these. They are sometimes missed or split into several boxes.
- **Motion blur:** Detection drops on blurred frames.
- **False positives:** Car windows, rocks and puddles. Tracking removes the
  single-frame ones but not those that persist across frames.
- **Duplicate suppression:** It can also suppress a genuinely separate small
  pothole that sits inside a larger one.
- **Test set:** I did not separately check that test scenes are independent
  of train scenes. Do not read the test result as performance on new
  locations.

## Usage

This project runs on Windows with Python 3.11, the existing CUDA-enabled
`venv` and Ultralytics 8.4.x. Do not rebuild the virtual environment, and do
not delete the data or the existing `runs/` records. The main dataset is
`data/MWPD_reviewed_v1`, configured in `data.reviewed-v1.yaml`. The `test`
split is not used for model selection. `data/`, `runs/`, `reports/` and
`*.pt` files are not in the repo.

### Environment check

Open a PowerShell / VS Code terminal in the project root:

```powershell
& .\venv\Scripts\python.exe -c "import torch, ultralytics; print(torch.__version__, ultralytics.__version__, torch.cuda.is_available())"
```

### Training

Run this first. It checks the data paths and label counts, verifies the
SHA-256 data manifest if one exists, and loads the model, but does not start
training:

```powershell
& .\venv\Scripts\python.exe .\train.py --config .\experiments\reviewed_v1_baseline.yaml
```

Add `--start` to the same command to start training. Each run writes to a
new `runs/` folder named after the config and a timestamp. Other experiments
are in `experiments/`: `reviewed_v1_yolov8s.yaml`, `p2_reviewed_v1.yaml` and
`mwpd_hrp4k_scaled.yaml`. The P2 recipe and its memory test are described in
the [P2 experiment note](experiments/p2_reviewed_v1_plan.md) (in Turkish).

### Validation

```powershell
& .\venv\Scripts\python.exe .\eval.py --weights .\runs\reviewed-v1-baseline_20260915_223719_077806\weights\best.pt
```

To compare several models, repeat `--weights`. Defaults: data
`data.reviewed-v1.yaml`, split `val`, image size 640. P/R/mAP come from
Ultralytics `model.val()`. Size buckets and TP IoU diagnostics use
`conf=0.25` and matching `IoU=0.5`. Results are written to
`reports/eval_<run>_<time>/summary.json` and `summary.md`. The final model is
evaluated on the test split only with `evaluate_reviewed_test.py`.
[EVAL_PROTOCOL.md](EVAL_PROTOCOL.md) (in Turkish) explains why an earlier
SAHI-based score differs from the `model.val()` score.

### Error analysis

```powershell
& .\venv\Scripts\python.exe .\inspect_errors.py --weights .\runs\<run>\weights\best.pt
```

This sorts misses and false positives into categories. It writes annotated
images and a `review.csv` for manual review to
`reports/errors_<run>_<time>/`.

### Video demo

```powershell
& .\venv\Scripts\python.exe .\demo_video.py --weights .\runs\reviewed-v1-baseline_20260915_223719_077806\weights\best.pt --source <video.mp4> --mode track
```

Tune the filter with `--min-hits`, `--hold`, `--alpha` and
`--overlap-thresh`. Use `--max-frames` for a short test run. Output goes to
`reports/video_demo_track_<time>/`: `compare.mp4`, `track.mp4`, `summary.md`
and `per_frame.csv`.

### Scanning external checkpoints

```powershell
& .\venv\Scripts\python.exe .\tools\scan_pickle.py <file.pt> [--output report.json]
```

This checks the pickle references inside a `.pt` file against an allowlist
before you load it. A clean result does not guarantee the file is safe.

## Files

| File | Purpose |
| --- | --- |
| `train.py`, `experiments/*.yaml` | Config-based training and pre-flight checks |
| `eval.py`, [EVAL_PROTOCOL.md](EVAL_PROTOCOL.md) | Standard validation measurement |
| `evaluate_reviewed_test.py` | One-off test-split evaluation |
| `prepare_reviewed_dataset.py`, [ANNOTATION_POLICY.md](ANNOTATION_POLICY.md) | Label revision |
| `inspect_errors.py` | Miss and false-positive analysis |
| `make_hrp4k_scaled.py`, `tile_yolo_dataset.py`, [CROP_PIPELINE.md](CROP_PIPELINE.md) | HRP4K preparation |
| `demo_video.py`, `trackers/` | Video demo and tracking |
| `tools/scan_pickle.py` | Safety scan for external checkpoints |
| `assets/` | README GIFs |

The docs linked from this README (`EVAL_PROTOCOL.md`, `ANNOTATION_POLICY.md`,
`CROP_PIPELINE.md`) are in Turkish.
