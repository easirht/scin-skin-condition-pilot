# Skin condition classification on SCIN: a pre-registered pilot with controls

Research prototype. **Not a medical device and not for diagnosis.** Predictions are research
outputs; they must not be used to decide anything about a person's health.

## What this is

A small, protocol-first study of image classification on the SCIN dermatology dataset, with
explicit controls (metadata-only baselines, class-weight controls), group-aware splits,
calibration, selective prediction and a skin-tone composition analysis. The main outcome is a
set of honest, uncertain results, not a high accuracy.

## Scope (read this first)

- This is **not a facial-skin study**. The first plan was facial skin conditions. SCIN has only a
  coarse head-or-neck tag, and a manual review of 267 images found that most do not show the face
  (see "Face annotation" below), so the scope moved.
- **v1** (head/neck only, 114 cases, 4 classes) was a null result: no model could be distinguished
  from chance. **v2** (all body parts, 1,172 cases, 8 classes) was started after seeing the v1 result
  and is exploratory.
- Labels are dermatologist-estimated from photographs, not biopsy-confirmed. Cases where
  dermatologists tied for the top condition (30.2% of labelled cases) were excluded, so the classes
  describe a consensus subset.

## Data

SCIN (Skin Condition Image Network), Google, released under the SCIN Data Use License. No SCIN
images, face-review sheets or model weights are included in this repository.

## Attribution

This project uses the SCIN (Skin Condition Image Network) dataset, released by Google under the
SCIN Data Use License (https://github.com/google-research-datasets/scin/blob/main/LICENSE).
Dataset: https://github.com/google-research-datasets/scin
Ward A, et al. Creating an Empirical Dermatology Dataset Through Crowdsourcing With Web Search
Advertisements. JAMA Netw Open. 2024;7(11):e2446615. doi:10.1001/jamanetworkopen.2024.46615
Derived data in this repository (label tables, aggregate results) were modified from the original:
labels were restricted to cases with a dermatologist label and a unique top-1 condition, grouped
by duplicate-linked cases, and split into folds. No SCIN images are redistributed here.
This project is not endorsed by or affiliated with Google or the SCIN authors.

## Method in short

- Unit of analysis: the case. Cases that share a byte-identical image are merged into one group;
  folds are StratifiedGroupKFold (K=5, seed 42); metrics are pooled out-of-fold, with confidence
  intervals from a bootstrap that resamples groups (2,000 resamples, seed 42).
- v2 classes (>= 50 cases after tie exclusion): Eczema 368, Allergic Contact Dermatitis 224,
  Urticaria 175, Insect Bite 134, Folliculitis 92, Psoriasis 65, Tinea 60, Herpes Simplex 54.
  Eczema and Allergic Contact Dermatitis overlap clinically and are kept separate.
- Models: frozen ImageNet features (MobileNetV3-Small, ResNet-18, EfficientNet-B0) with logistic
  regression; fine-tuned MobileNetV3-Small. Controls: majority class and metadata-only logistic
  regression (body part, and a richer set with textures, symptoms, age group, sex).
- Primary metric: balanced accuracy. Chance for 8 classes is 0.125.

## Results (v2, 1,172 cases, balanced accuracy [95% CI], pooled out-of-fold)

| Model | Balanced accuracy | Accuracy |
|---|---|---|
| Majority class | 0.125 | 0.314 [0.288, 0.341] |
| Body parts only (metadata) | 0.163 [0.148, 0.179] | 0.323 |
| Richer metadata | 0.207 [0.184, 0.232] | 0.306 |
| Frozen MobileNetV3-S | 0.209 [0.185, 0.232] | 0.302 |
| Frozen ResNet-18 | 0.214 [0.192, 0.237] | 0.324 |
| Frozen EfficientNet-B0 | 0.218 [0.193, 0.244] | 0.299 |
| Fine-tuned MobileNetV3-S (class-weighted loss) | 0.301 [0.272, 0.333] | 0.294 [0.268, 0.319] |

Class-weight control (same models re-run with class_weight="balanced", because the fine-tuned
model used a class-weighted loss and the others did not): body parts 0.205 [0.176, 0.235],
richer metadata 0.253 [0.222, 0.285], frozen MobileNetV3-S 0.282 [0.252, 0.310], ResNet-18
0.273 [0.243, 0.305], EfficientNet-B0 0.282 [0.250, 0.314].

What the data support:
- The image models are above chance and above body-part-only metadata.
- They are **not distinguishable from contributor-reported metadata** at the stricter
  weight-matched threshold (the richer-metadata upper bound is 0.285; no model's interval lower
  bound exceeds it).
- Fine-tuning is **not shown to beat frozen features** at equal class weighting (paired
  differences +0.019 to +0.028, all intervals include 0).
- Overall accuracy does not exceed the majority class (0.314).
- v1 (114 head/neck cases, 4 classes): no model distinguishable from chance (all balanced-accuracy
  intervals include 0.25).

## Calibration and abstention

Unweighted frozen models were overconfident (mean confidence 0.525 / 0.444 / 0.639 against
accuracy 0.302 / 0.324 / 0.299). Class-weighted and fine-tuned models were closer to the
class-frequency reference. Keeping only the 20% most confident cases raises accuracy (fine-tuned:
0.372) but most of those predictions are still wrong. Confidence-based abstention is somewhat
informative and does not make these models reliable.

## Skin tone

Dermatologist-estimated Fitzpatrick FST5-6 is 73 of 1,105 cases (6.6%), FST6 alone 7. The data
cannot support any statement about darker skin tones. Group results are in the policy log and are
descriptive; no fairness or bias claim is made.

## Face annotation

267 candidate head/neck images were judged by one annotator (author), blind to the condition:
face 9, partial face 83, no face 125, unclear 50. Intra-rater agreement on 12 re-shown images was
poor (exact 2/12, Cohen kappa -0.11). A Haar-cascade face detector had sensitivity 0.11 against
this annotation and was not used.

## Limitations

- Single annotator, small samples, labels estimated from photographs, ties excluded.
- v2 was designed after the v1 null result; the claim-rule thresholds came from the controls, but
  several later analyses (class-weight control, subgroup, calibration) are post-hoc.
- 8 classes of mostly inflammatory or infectious conditions; no healthy class.
- The protocol log (docs/label_policy_v2.md) was created after some v2 runs and says so at the top;
  file times indicate order but do not prove it.

## Reproducing

Scripts are numbered in src/ (00 download metadata, 04 and 24 images, 25 duplicate groups, 27
folds, 28 metadata baselines, 29 and 30 frozen features, 31 fine-tuning, 33 and 36 paired
bootstraps, 37 results table, 38 calibration, 40 subgroups, 41 errors). Images are downloaded from
the public SCIN bucket and are never committed (see .gitignore). Seed 42 everywhere.
