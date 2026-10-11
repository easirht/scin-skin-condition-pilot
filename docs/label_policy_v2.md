# Protocol v2 and log

DISCLOSURE: this file was created on 2026-10-10 AFTER the v2 metadata, frozen-feature and
fine-tuning runs had been executed. Entries 1-8 are retrospective records of what was specified
in the working conversation and found; they were not saved to a file when first stated. Their
order can be checked against file modification times (outputs/reports, src). Entry 9 onward
is written before the run it describes.

1. Scope. v1 (114 head/neck cases, 4 classes) was a null result: no model distinguishable from
   chance. v2 widens to all body parts to increase sample size. Post-null-result scope change,
   exploratory. v2 is NOT a facial-skin study.
2. Eligibility and classes. Dermatologist label present; top-1 ties excluded (925 of 3,061
   labelled cases, 30.2%). Classes: top-1 condition with >= 50 cases after tie exclusion. The
   threshold was chosen after seeing class-count scenarios and confirmed after seeing class names;
   it was not chosen from any model result. 8 classes, 1,172 cases: Eczema 368, Allergic Contact
   Dermatitis 224, Urticaria 175, Insect Bite 134, Folliculitis 92, Psoriasis 65, Tinea 60,
   Herpes Simplex 54. Overlapping conditions (Eczema, Allergic Contact Dermatitis) are not merged.
3. Data. 2,560 image rows, 2,553 unique files after excluding one file that returns HTTP 404
   (a Tinea case, which keeps its other images). 7 files are shared across cases; 3 groups of 2
   cases; 1,169 duplicate-linked groups; 0 groups with different labels. Folds:
   manifests/folds_v2.csv, StratifiedGroupKFold, K=5, seed 42, sizes 235/235/234/234/234, created
   before any v2 model. Per-fold metrics are not reported; all metrics are pooled out-of-fold with
   group-bootstrap 95% CIs (2,000 resamples, seed 42).
4. Rejected dataset. A Kaggle "Skin diseases image dataset" (~27k images) was considered and
   rejected: no source/provenance, no identifiable license, no case identifiers.
5. Controls (balanced accuracy [95% CI], pooled out-of-fold, n=1,172). Majority class 0.125
   (accuracy 0.314 [0.288, 0.341]). Body parts only + logistic regression 0.163 [0.148, 0.179].
   Richer metadata (body parts, textures, symptoms, age group, sex) 0.207 [0.184, 0.232]. These
   were run before any image model.
6. Claim rule. An image model is called better than chance only if its balanced-accuracy CI lower
   bound exceeds 0.179 (upper bound of the body-part baseline). 0.232 (richer metadata upper
   bound) is an additional stricter reference. Both thresholds come from the control results above.
7. Frozen features (MobileNetV3-S, ResNet-18, EfficientNet-B0, torchvision IMAGENET1K_V1,
   logistic regression, C by inner group-CV, run once; one run was interrupted and re-run
   unchanged with identical ResNet-18 figures). Balanced accuracy 0.209 [0.185, 0.232], 0.214
   [0.192, 0.237], 0.218 [0.193, 0.244]; accuracy 0.302, 0.324, 0.299. All PASS the claim rule
   (lower bounds 0.185/0.192/0.193), none exceeds 0.232, none distinguishable from the others
   or from richer metadata. Accuracy does not exceed the majority class.
8. Fine-tuned MobileNetV3-S (AdamW lr 1e-4, wd 1e-2, batch 16, 12 epochs, class-weighted loss,
   epoch by inner validation loss; best epochs 3/5/6/4/6; inner validation losses 1.960/1.879/
   1.875/1.921/1.831, below the 8-class chance cross-entropy ln 8 = 2.079). Accuracy 0.294
   [0.268, 0.319]; balanced accuracy 0.301 [0.272, 0.333]; macro-F1 0.265 [0.238, 0.291]. PASS
   (lower bound 0.272) and above 0.232. Paired bootstrap, balanced-accuracy difference to
   majority +0.176, body-part +0.138, richer metadata +0.094, frozen MobileNetV3-S +0.093,
   ResNet-18 +0.087, EfficientNet-B0 +0.083 (all CIs exclude 0). CAVEAT: the fine-tuned model used
   a class-weighted loss and the compared baselines did not, so these differences are confounded
   with class weighting. Per-class recall: Folliculitis 0.522, Urticaria 0.440, Herpes Simplex
   0.315, Eczema 0.277, Insect Bite 0.269, Tinea 0.233, Psoriasis 0.185, Allergic Contact
   Dermatitis 0.170.
9. (Written before the run.) Class-weight controls, exploratory, post-hoc: the three frozen-feature
   logistic-regression models and both metadata baselines are re-run with
   class_weight="balanced", everything else unchanged (src/34_..._balanced.py,
   src/35_..._balanced.py), because the fine-tuned model was class-weighted and they were not.
   Both versions are reported. The claim-rule thresholds (0.179, 0.232) are not changed.

10. Class-weight controls, results (balanced accuracy [95% CI], pooled out-of-fold, n=1,172,
    class_weight="balanced"; majority class identical to the unweighted run). Body parts only
    0.205 [0.176, 0.235]; richer metadata 0.253 [0.222, 0.285]; frozen MobileNetV3-S 0.282
    [0.252, 0.310]; ResNet-18 0.273 [0.243, 0.305]; EfficientNet-B0 0.282 [0.250, 0.314].
    Accuracy fell for the metadata models (0.169, 0.204) and for the frozen models (0.298,
    0.259, 0.274). Class weighting raised balanced accuracy by about 0.04-0.07 for every
    unweighted baseline, so the earlier gap to the class-weighted fine-tuned model (0.301
    [0.272, 0.333]) was largely a weighting effect. The locked claim-rule thresholds (0.179,
    0.232) are unchanged and their PASS results stand.
11. (Written before the run.) src/36_v2_matched_paired.py, exploratory and post-hoc: paired group
    bootstrap (2,000 resamples, seed 42) of balanced accuracy between the class-weighted models
    (fine-tuned MobileNetV3-S vs class-weighted frozen and metadata models; each class-weighted
    frozen model vs class-weighted metadata). A weight-matched claim rule is also reported:
    thresholds are the 97.5th percentile of the class-weighted body-part and richer-metadata
    balanced accuracy, computed from the same bootstrap. It is secondary and does not replace
    the locked rule.

11b. Results of entry 11 (src/36_v2_matched_paired.py, run 2026-10-10 after entry 11 was saved).
    Class-weighted balanced accuracy [95% group-bootstrap CI]: body parts 0.205 [0.176, 0.235];
    richer metadata 0.253 [0.222, 0.285]; frozen MobileNetV3-S 0.282 [0.252, 0.310]; ResNet-18
    0.273 [0.243, 0.305]; EfficientNet-B0 0.282 [0.250, 0.314]; fine-tuned MobileNetV3-S 0.301
    [0.272, 0.333]. Weight-matched thresholds (97.5th percentile): body-part 0.235,
    richer-metadata 0.285. Lower bounds: frozen 0.252/0.243/0.250 and fine-tuned 0.272 all exceed
    0.235; none exceeds 0.285. Paired differences (same resampled groups): fine-tuned minus body
    parts +0.096 [+0.059, +0.135]; minus richer metadata +0.048 [+0.007, +0.090]; minus frozen
    MobileNetV3-S +0.019 [-0.017, +0.057], ResNet-18 +0.028 [-0.009, +0.064], EfficientNet-B0
    +0.019 [-0.020, +0.058] (all three include 0). Frozen minus body parts +0.068 to +0.077 (all
    exclude 0); frozen minus richer metadata +0.020 to +0.029 (all include 0). Eleven paired
    comparisons, no multiplicity correction; the fine-tuned-vs-richer-metadata lower bound
    (+0.007) is marginal. Conclusion: image models exceed chance and body-part-only metadata, are
    not distinguishable from contributor-reported metadata at the stricter threshold, and
    fine-tuning is not shown to beat frozen features at equal class weighting.

13. (Written before the run.) Calibration and selective prediction, src/38_v2_calibration.py, on
    saved out-of-fold case-level probabilities; no training. Models: fine-tuned MobileNetV3-S and
    the frozen MobileNetV3-S, ResNet-18, EfficientNet-B0, each unweighted and class-weighted
    (fine-tuned is class-weighted only). Metrics: top-label ECE (10 equal-width bins), multiclass
    Brier score, negative log-likelihood, reliability diagram; 95% CIs from a group bootstrap
    (2,000 resamples, seed 42). Selective prediction: cases ranked by top-label probability;
    the top 100/80/60/40/20% are retained; accuracy on the retained set is reported with the
    largest-class share of that set as a reference, with group-bootstrap CIs. No threshold is
    tuned. Class-weighted models are not expected to be calibrated to class frequencies, so ECE
    is a descriptive property, not a claim.

12. Order check from file times (2026-10-10, local time). folds_v2.csv 2:00 PM; majority and
    metadata-only predictions 2:03-2:04 PM; src/30_v2_frozen_logreg.py (contains the claim-rule
    thresholds 0.179 and 0.232) last saved 5:19 PM; frozen predictions 5:20-5:22 PM;
    src/31_v2_finetune.py last saved 5:28 PM; fine-tuned predictions 6:24 PM. This file was created
    at 10/10/2026 18:40:14, before the first class-weighted prediction file (6:40:28 PM), so entry 9 was written
    before that run. Entries 10 and 11 were in the file when it was last written at 6:44:45 PM;
    v2_matched_paired.txt was last written at 10/10/2026 22:33:27, after entry 11. Before this entry the file's last
    write was 10/10/2026 22:40:49 (entry 13); calibration_selective_v2.csv was written at 10/10/2026 22:41:50, so entry 13 was in the file before
    that run finished. Modification times change when a file is edited: they indicate order, they
    do not prove it.

14. Results of entry 13 (src/38_v2_calibration.py; point estimates, CIs in
    outputs/reports/v2_calibration.txt, full table in calibration_selective_v2.csv). References from
    class frequencies: Brier 0.816, NLL 1.866; uniform: Brier 0.875, NLL 2.079. ECE / Brier / NLL:
    fine-tuned (weighted) 0.056/0.814/1.824; frozen MobileNetV3-S 0.223/0.872/2.132, ResNet-18
    0.119/0.833/1.938, EfficientNet-B0 0.340/1.007/4.195; class-weighted frozen MobileNetV3-S
    0.058/0.819/1.847, ResNet-18 0.061/0.841/1.928, EfficientNet-B0 0.037/0.819/1.852. The unweighted
    frozen models are worse than the class-frequency reference on NLL (EfficientNet-B0 is worse than
    uniform on both metrics); the weighted and fine-tuned models are close to it. Class-weighted
    models are not expected to match class frequencies, so the comparison is rough. Selective
    prediction, accuracy at 100/40/20% kept (largest-class share of kept set at 40/20% in brackets):
    fine-tuned 0.294/0.328(0.292)/0.372(0.226); frozen MobileNetV3-S 0.302/0.371(0.341)/0.427(0.372);
    ResNet-18 0.324/0.343(0.316)/0.385(0.346); EfficientNet-B0 0.299/0.358(0.318)/0.363(0.316);
    weighted MobileNetV3-S 0.298/0.316(0.269)/0.380(0.244); weighted ResNet-18
    0.259/0.256(0.279)/0.291(0.256); weighted EfficientNet-B0 0.274/0.313(0.294)/0.350(0.274).
    At 20% kept every model is more accurate than at 100%; the margin over the kept-set largest-class
    share is +0.146 (fine-tuned) and +0.136 (weighted MobileNetV3-S), and +0.035 to +0.076 for the
    others. Even at 20% coverage most fine-tuned predictions are wrong (accuracy 0.372, about 234
    cases). 35 model-by-fraction results, no multiplicity correction. Confidence-based abstention is
    somewhat informative but does not make these models reliable.

14b. Correction and CIs for entry 14 (from outputs/reports/v2_calibration.txt). Class-frequency
    reference computed from the 1,172 case labels: Brier 0.816, NLL 1.866 (in-sample). Entry 14
    said the unweighted frozen models were worse than this reference on NLL; that holds only for
    MobileNetV3-S (2.132 [2.031, 2.230]) and EfficientNet-B0 (4.195 [3.903, 4.488]). ResNet-18
    unweighted has NLL 1.938 [1.859, 2.013] and Brier 0.833 [0.811, 0.854], both CIs include the
    reference. Fine-tuned: Brier 0.814 [0.801, 0.827], NLL 1.824 [1.786, 1.863]. Class-weighted
    frozen MobileNetV3-S NLL 1.847 [1.804, 1.890] and EfficientNet-B0 1.852 [1.826, 1.877] include
    the reference; class-weighted ResNet-18 (Brier 0.841 [0.832, 0.850], NLL 1.928 [1.900, 1.956])
    is worse than it. Mean confidence vs accuracy: unweighted frozen 0.525/0.302, 0.444/0.324,
    0.639/0.299 (overconfident); weighted frozen and fine-tuned 0.242-0.332 vs 0.259-0.298.
15. Skin-tone analysis plan. The rule was stated in the working conversation before any
    skin-tone count was run and is recorded in this file only now: exploratory and descriptive; a
    tone group is analysed only if it has >= 30 cases and >= 2 classes with >= 5 cases each,
    otherwise it is listed as too small; no significance tests, no fairness or bias claim, no
    multiplicity correction because no inferential claim is made. Marginal counts (dermatologist
    FST label 1: FST1 116, FST2 412, FST3 341, FST4 163, FST5 66, FST6 7, missing 67; Monk US: 1:
    138, 2: 389, 3: 303, 4: 175, 5: 77, 6: 49, 7: 23, 8: 9, 9: 3, 10: 1, missing 5) were seen
    BEFORE the groupings below were chosen; no class-by-group count and no group-level
    performance had been seen. Groupings: dermatologist FST (label 1 only; labels 2 and 3 are
    not used): FST1-2, FST3, FST4, FST5-6 (FST6 alone has 7 cases). Monk (US): 1-2, 3, 4, 5-10.
    Self-reported Fitzpatrick is not used (544 of 1,172 missing). Models: fine-tuned
    MobileNetV3-S and class-weighted richer-metadata logistic regression (reference). Metric:
    balanced accuracy over the classes with >= 5 cases in that group, plus accuracy, group
    bootstrap CI (2,000 resamples, seed 42). Marginal observation: dermatologist estimates call
    6.6% of cases FST5-6 versus 18.1% among those who self-reported an FST; the denominators
    differ and case-level agreement has not been examined.

16. Skin-tone subgroup protocol, written BEFORE any group-level performance was computed
    (src/40_v2_subgroups.py). Class-by-group counts were seen first (all eight groups pass the
    entry-15 rule). Existing out-of-fold predictions are used (no retraining), so subgroup results
    describe a model trained on all tones. Per group: balanced accuracy = mean recall over the
    classes with >= 5 cases in that group, over the same cases for every model; predictions may
    be any of the 8 classes. Models: fine-tuned MobileNetV3-S, class-weighted richer-metadata
    logistic regression, and the majority-class predictor (for k included classes it scores 1/k).
    Reported per group: n, included classes, accuracy of the fine-tuned model, balanced accuracy
    with a duplicate-linked-group bootstrap CI (2,000 resamples, seed 42; a class absent from a
    resample is skipped for that resample), and the paired difference fine-tuned minus
    class-weighted metadata. Groups are NOT compared to each other with tests or differences,
    because class mixes and chance references differ (e.g. FST5-6 has 5 included classes, chance
    reference 0.2, versus 8 classes elsewhere); each group is read only against its own references.
    No fairness or bias claim is made. FST5-6 (73 cases; Herpes Simplex 5, Tinea 5) is expected to
    have very wide intervals.

17. Order check, entries 13 and 16 (local time, 2026-10-10). Entry 13 was in the file by 22:40:49; v2_calibration.txt, the log of that run, was created at 22:41:37 (48 seconds later) and calibration_selective_v2.csv at 22:41:50. The file's last write before this entry was at 10/11/2026 10:30:19 (entry 16); v2_subgroups.txt was created at 10/11/2026 10:30:55, so entry 16 was in the file before the run started. Modification times indicate order; they do not prove it.

18. Skin-tone subgroup results (src/40_v2_subgroups.py, existing out-of-fold predictions,
    outputs/reports/v2_subgroups.txt and subgroups_v2.csv). Balanced accuracy of the fine-tuned
    MobileNetV3-S [95% group-bootstrap CI]; class-weighted richer-metadata baseline; paired
    difference. Dermatologist FST (label 1): FST1-2 n=528: 0.286 [0.242, 0.333]; 0.251;
    +0.036 [-0.026, +0.098]. FST3 n=341: 0.332 [0.271, 0.398]; 0.272; +0.059 [-0.028, +0.150].
    FST4 n=163: 0.297 [0.213, 0.384]; 0.226; +0.071 [-0.014, +0.159]. FST5-6 n=73 (5 included
    classes; Psoriasis 3, Insect Bite 3, Folliculitis 2 excluded; majority reference 0.200): 0.218
    [0.125, 0.347]; 0.185; +0.033 [-0.116, +0.166]. Monk (US): 1-2 n=527: 0.310 [0.260, 0.357];
    0.277; +0.032 [-0.041, +0.097]. 3 n=303: 0.257 [0.196, 0.324]; 0.274; -0.017 [-0.108, +0.073].
    4 n=175: 0.367 [0.277, 0.460]; 0.217; +0.150 [+0.067, +0.240]. 5-10 n=162: 0.242 [0.176,
    0.314]; 0.221; +0.021 [-0.081, +0.122]. Majority reference 0.125 where 8 classes are included.
    Interpretation, fixed here: (1) the data cannot support any statement about darker tones
    (FST5-6 has 73 cases and a CI spanning both its own chance reference and the other groups'
    point estimates); (2) one of eight paired intervals (Monk 4) excludes 0; it was not a
    hypothesis, shows no monotonic pattern across tone (Monk 3 -0.017, Monk 5-10 +0.021), the
    FST and Monk groupings overlap in cases so the comparisons are not independent, and there is
    no multiplicity correction, so it is not reported as a finding; (3) seven of eight point
    differences are positive, consistent with the pooled +0.048, but this is not independent
    evidence; (4) no fairness or bias claim is made, and groups are not compared to each other.
    Sample composition is the main result: dermatologist-estimated FST5-6 is 73 of 1,105 cases
    (6.6%) and FST6 alone is 7.

19. (Written before the run.) Error analysis, src/41_v2_errors.py, on the saved out-of-fold
    predictions of the fine-tuned MobileNetV3-S; no training. Reported: (a) row-normalised
    confusion matrix (recall by true class) with group-bootstrap CIs for the diagonal; (b) the
    largest off-diagonal cells; (c) for the pair Eczema / Allergic Contact Dermatitis: the share of
    each class predicted as the other, and the balanced accuracy when the two are merged into one
    class (descriptive only: the classes are never merged in any analysis, the merge is used only
    to measure how much of the error is this one confusion); (d) distribution of the top-label
    probability for correct versus incorrect predictions. No clinical interpretation is attached
    to any cell: the author is not a dermatologist, and labels are dermatologist-estimated from
    photographs, not biopsy-confirmed.

20. Results of entry 19 (src/41_v2_errors.py, outputs/reports/v2_errors.txt; fine-tuned
    MobileNetV3-S, out-of-fold, n=1,172). Recall [95% group-bootstrap CI]: Folliculitis 0.522
    [0.417, 0.622], Urticaria 0.440 [0.372, 0.518], Herpes Simplex 0.315 [0.192, 0.444], Eczema
    0.277 [0.236, 0.324], Insect Bite 0.269 [0.198, 0.345], Tinea 0.233 [0.132, 0.345], Psoriasis
    0.185 [0.092, 0.286], Allergic Contact Dermatitis 0.170 [0.124, 0.221]. Largest off-diagonal
    cells (count, true to predicted): Eczema to Urticaria 69 (0.19 of Eczema), Eczema to
    Folliculitis 59 (0.16), Allergic Contact Dermatitis to Urticaria 43 (0.19), Urticaria to
    Eczema 38 (0.22), Allergic Contact Dermatitis to Eczema 38 (0.17), Eczema to Herpes Simplex 37
    (0.10), Eczema to Allergic Contact Dermatitis 35 (0.10), Allergic Contact Dermatitis to Insect
    Bite 33 (0.15). Eczema predicted as Allergic Contact Dermatitis 35/368 = 0.10; the reverse
    38/224 = 0.17. Descriptive merge of the two classes (7 classes, never used in any analysis):
    balanced accuracy 0.332 [0.299, 0.366] versus 0.301 unmerged; the intervals overlap, so this
    one confusion is not the main source of error. Top-label probability, correct versus incorrect
    predictions: median 0.289 vs 0.277, quartiles 0.243/0.380 vs 0.228/0.344 (heavily overlapping).
    No clinical interpretation is attached to any cell.

21. SCIN Data Use License read in full on 2026-10-11 from github.com/google-research-datasets/scin
    (LICENSE, 147 lines); this was not done before the data were downloaded, and earlier entries
    that mention the license describe it only by name. Terms relevant here (author's reading, not
    legal advice): worldwide, royalty-free, non-exclusive, irrevocable right to reproduce and Share
    the Licensed Material and Adapted Material; no non-commercial or no-derivatives clause seen;
    attribution conditions (creator identification, license notice and link, warranty-disclaimer
    notice, dataset link, indication of modification); re-identification or re-linking of
    contributors is prohibited and ends all rights; privacy and publicity rights are not licensed;
    no downstream restrictions; no endorsement; as-is. Not addressed in the license: trained model
    weights. Decisions: no SCIN image is placed in any public repository, demo, figure or
    README; face-review sheets and annotation files stay private; public outputs are aggregate
    numbers and charts; any shared manifest carries the attribution; model weights are not
    published until (a) the author has settled whether weights count as Adapted Material and
    (b) the license of the torchvision IMAGENET1K_V1 weights has been checked (not yet done).
    Planned demo: audit and uncertainty explorer, no diagnosis output, no images, no weights.

22. (Written before the run.) Ranking and curve evaluation, src/42_v2_curves.py, on saved
    out-of-fold case-level probabilities; no training. Models: fine-tuned MobileNetV3-S and the
    frozen MobileNetV3-S, ResNet-18, EfficientNet-B0, each unweighted and class-weighted (fine-tuned
    is class-weighted only). Metrics: one-vs-rest ROC-AUC per class and macro-average (unweighted
    mean over the 8 classes; chance 0.5); average precision (AP) per class against that class's
    prevalence, and macro-AP against the mean prevalence 0.125 (chance reference, approximate);
    95% CIs from the duplicate-linked-group bootstrap (2,000 resamples, seed 42; a class absent from
    a resample is skipped for that resample). Paired macro-AUC differences, fine-tuned minus each
    of the six frozen models, same resampled groups; six comparisons, no multiplicity correction.
    Figures (no images): ROC and precision-recall curves for the fine-tuned model, row-normalised
    confusion-matrix heatmap for the fine-tuned model, macro-AUC with CIs for all models.
    Metadata-only baselines are not included because only their predicted classes were saved, not
    probabilities. Interpretation rule: a model is said to rank a class above chance only if the
    CI of its AUC excludes 0.5; AP is read against prevalence. No threshold is chosen.

23. Results of entry 22 (src/42_v2_curves.py, outputs/reports/v2_curves.txt and curves_v2.csv). Order check: entry 22 present in file = True; file last written (before this entry) at 10/11/2026 10:49:51; v2_curves.txt created at 10/11/2026 10:50:24; so entry 22 was in the file before the run started. Macro one-vs-rest ROC-AUC [95% group-bootstrap CI] / macro AP: fine-tuned MobileNetV3-S (weighted) 0.707 [0.687, 0.728] / 0.255; frozen MobileNetV3-S 0.669 [0.647, 0.689] / 0.221; ResNet-18 0.659 [0.637, 0.681] / 0.212; EfficientNet-B0 0.634 [0.612, 0.655] / 0.207; class-weighted frozen MobileNetV3-S 0.693 [0.674, 0.713] / 0.246; ResNet-18 0.672 [0.651, 0.693] / 0.214; EfficientNet-B0 0.704 [0.686, 0.723] / 0.239. Fine-tuned per-class AUC: Folliculitis 0.787, Herpes Simplex 0.761, Urticaria 0.749, Insect Bite 0.705, Psoriasis 0.705, Tinea 0.663, Allergic Contact Dermatitis 0.644, Eczema 0.643; every AUC CI (all 56 model-by-class cells) excludes 0.5; fine-tuned AP exceeds class prevalence in all 8 classes with CI lower bounds above prevalence. Paired macro-AUC, fine-tuned minus: frozen MobileNetV3-S +0.038 [+0.016, +0.063]; ResNet-18 +0.048 [+0.025, +0.072]; EfficientNet-B0 +0.073 [+0.050, +0.098]; class-weighted MobileNetV3-S +0.013 [-0.008, +0.035]; ResNet-18 +0.035 [+0.012, +0.055]; EfficientNet-B0 +0.003 [-0.018, +0.024]. Six comparisons, no multiplicity correction; only the weight-matched ResNet-18 comparison excludes 0 and it has not been checked against a corrected interval. Class weighting raised macro AUC of the frozen models by +0.024, +0.013 and +0.070, so AUC is not independent of class weighting (contrary to an earlier expectation in the working conversation). Ranking is above chance for every class but modest (AUC 0.63-0.79); fine-tuning is not shown to beat weight-matched frozen features. Metadata-only models were not included because their probabilities had not been saved.

24. (Written before the run.) Metadata-only ranking comparison, src/43_v2_meta_auc.py. The four
    metadata logistic-regression models (body parts and richer metadata, each unweighted and
    class-weighted) are refit with the same features, folds, inner group-CV and grid as scripts 28
    and 35, now saving predict_proba; the script asserts that the argmax of the new probabilities
    equals the saved predictions of those scripts. Reported: macro ROC-AUC and macro AP with
    group-bootstrap CIs (same resamples as entry 22), per-class AUC point estimates. Paired
    macro-AUC differences are computed only within the class-weighted setting: fine-tuned minus
    class-weighted body parts and richer metadata, and each class-weighted frozen model minus
    class-weighted richer metadata (five comparisons, no multiplicity correction). Unweighted
    metadata results are shown for information, without paired tests. No training of any image
    model. The fine-tuned macro AUC is recomputed as a consistency check (expected 0.707).

25. Results of entry 24 (src/43_v2_meta_auc.py, outputs/reports/v2_meta_auc.txt). Order check: entry 24 present = True; file last written (before this entry) at 10/11/2026 10:56:35; v2_meta_auc.txt created at 10/11/2026 10:57:13; so entry 24 was in the file before the run started. The script completed, so its assertion that the refit metadata probabilities reproduce the saved predicted classes passed. Macro ROC-AUC [95% group-bootstrap CI] / macro AP: body parts unweighted 0.588 [0.567, 0.610] / 0.171; richer metadata unweighted 0.635 [0.612, 0.657] / 0.200; body parts weighted 0.599 [0.577, 0.620] / 0.179; richer metadata weighted 0.649 [0.627, 0.670] / 0.210; fine-tuned 0.707 / 0.255; frozen weighted MobileNetV3-S 0.693 / 0.246, ResNet-18 0.672 / 0.214, EfficientNet-B0 0.704 / 0.239. Paired macro-AUC differences (class-weighted setting, same resampled groups): fine-tuned minus body parts +0.108 [+0.082, +0.134]; minus richer metadata +0.058 [+0.031, +0.086]; frozen MobileNetV3-S minus richer metadata +0.044 [+0.015, +0.072]; ResNet-18 +0.023 [-0.006, +0.053]; EfficientNet-B0 +0.055 [+0.029, +0.081]. Five comparisons, no multiplicity correction (a rough normal-approximation 99% interval, not computed by script, would keep fine-tuned and EfficientNet-B0 above 0 and MobileNetV3-S barely so). Metric disagreement, recorded as found: at the argmax, balanced accuracy did not separate any image model from richer metadata at the weight-matched threshold (0.285); by ranking (macro AUC) three of four image models exceed it. The locked primary metric is balanced accuracy; AUC was added afterwards (entry 22) as a secondary metric, so the primary conclusion is unchanged and the AUC result is reported alongside it. Caveats: the metadata control is a single logistic regression on 41 contributor-reported features; a stronger metadata model was not tried; image models may encode body location, which this comparison cannot separate. Per-class AUC, metadata: Herpes Simplex 0.791 (richer, weighted) exceeds the fine-tuned model's 0.761; Psoriasis body-parts 0.476-0.490 is below chance.

26. Pretrained-weights license check (2026-10-11, from the torchvision README): the pre-trained
    models in torchvision may carry their own licenses or terms derived from their training dataset;
    the user is responsible for checking permission for their use case; only the SWAG weights are
    stated to be CC-BY-NC 4.0; the library code is BSD-3-Clause. The weights used here
    (IMAGENET1K_V1 for MobileNetV3-Small, ResNet-18, EfficientNet-B0) are not SWAG; no explicit
    license for them was found, and the terms of ImageNet-1k itself were not checked. Decision
    unchanged: no model weights (pretrained or fine-tuned) are published.
