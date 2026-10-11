# Model card: SCIN skin-condition classifiers (research prototype)

**Not a medical device. Not for diagnosis, triage or treatment decisions.** No model weights are
released; this card describes the experiments.

## Models

Frozen ImageNet features (MobileNetV3-Small, ResNet-18, EfficientNet-B0; torchvision IMAGENET1K_V1)
with multinomial logistic regression, and a fine-tuned MobileNetV3-Small (AdamW, learning rate 1e-4,
weight decay 1e-2, batch 16, up to 12 epochs, class-weighted cross-entropy, epoch chosen by loss on
an inner validation split of the training cases). Metadata-only logistic regressions and a majority
class predictor serve as controls. Predictions are made per case (mean over the case's images).

## Intended use

Studying evaluation practice on a small dermatology dataset: group-aware splits, controls,
calibration, ranking metrics, subgroup composition. Not for any clinical, consumer or commercial
decision about a person's skin.

## Out-of-scope use

Diagnosis, screening, triage, advice on treatment, or any statement about a real person's condition.
Use on skin tones, body sites, devices or conditions outside the training data. Use as evidence
about facial skin (the data are not a facial-skin set).

## Training and evaluation data

SCIN (Google), SCIN Data Use License, cases with a dermatologist label and a unique top-1 condition
(30.2% of labelled cases excluded for ties), classes with at least 50 such cases: Eczema 368,
Allergic Contact Dermatitis 224, Urticaria 175, Insect Bite 134, Folliculitis 92, Psoriasis 65,
Tinea 60, Herpes Simplex 54 (1,172 cases, 2,560 image rows). Labels are dermatologist-estimated
from photographs, not biopsy-confirmed. Images are crowdsourced phone photographs from US internet
users. 5-fold StratifiedGroupKFold, groups = cases sharing a byte-identical image; all numbers are
pooled out-of-fold with group-bootstrap 95% intervals.

## Results (balanced accuracy; chance 0.125)

Fine-tuned MobileNetV3-S (class-weighted loss) 0.301 [0.272, 0.333]; class-weighted frozen models
0.273 to 0.282; unweighted frozen models 0.209 to 0.218; richer metadata only 0.207 (0.253
class-weighted); body-part metadata only 0.163 (0.205 class-weighted). Accuracy 0.294 for the
fine-tuned model, below the majority class (0.314). Macro one-vs-rest ROC-AUC 0.707 [0.687, 0.728]
(fine-tuned); every class above 0.5 (0.64 to 0.79). Per-class recall 0.17 (Allergic Contact
Dermatitis) to 0.52 (Folliculitis). Details and caveats are in README.md and docs/label_policy_v2.md.

## Calibration

Unweighted frozen models are overconfident (mean confidence 0.44 to 0.64 against accuracy 0.30 to
0.32). The fine-tuned model is closer to the class-frequency reference (ECE 0.056, Brier 0.814, NLL
1.824; reference Brier 0.816, NLL 1.866). Among the 20% most confident cases, fine-tuned accuracy is
0.372. Predicted probabilities are not reliable enough to act on.

## Known limitations and risks

- Small sample; wide intervals; 8 inflammatory or infectious conditions only; no healthy class;
  single-annotator face review with poor measured reliability (kappa -0.11, n=12).
- Skin tone: dermatologist-estimated Fitzpatrick FST5-6 is 73 of 1,105 cases (6.6%), FST6 alone 7.
  Nothing can be concluded about darker skin tones. No fairness or bias claim is made.
- Eczema and Allergic Contact Dermatitis overlap clinically and are kept as separate classes.
- Image models were not shown to add information beyond contributor-reported metadata at the
  argmax (balanced accuracy); by ranking (AUC) three of four exceed it. Image models may encode
  body location.
- Fine-tuning was not shown to beat weight-matched frozen features.
- The class set and the v2 scope were chosen after a null result on v1 (head/neck only, 114 cases);
  several later analyses are post-hoc.
- Errors in a skin classifier can lead to false reassurance or false alarm; none of these models
  should be shown to people as a prediction about their skin.

## Privacy and license

No SCIN image, face-review sheet or model weight is redistributed. The SCIN license prohibits
re-identifying contributors; no attempt was made. Weights are not published pending a check of the
terms of the pretrained weights (torchvision states they may have their own terms; none was found
for IMAGENET1K_V1) and of whether weights count as adapted material under the SCIN license.

## Attribution

This project uses the SCIN (Skin Condition Image Network) dataset, released by Google under the
SCIN Data Use License (https://github.com/google-research-datasets/scin/blob/main/LICENSE).
Ward A, et al. JAMA Netw Open. 2024;7(11):e2446615. doi:10.1001/jamanetworkopen.2024.46615.
Not endorsed by or affiliated with Google or the SCIN authors.
