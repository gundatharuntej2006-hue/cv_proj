BMS INSTITUTE OF TECHNOLOGY AND MANAGEMENT

Department of Artificial Intelligence and Machine Learning
Mini Project BAI506, Academic Year 2026-27

Explainable Tuberculosis Screening from Chest X-Rays with
Lesion Localisation for Triage in Resource-Limited Settings

Technical Methodology

The proposed system is an explainable and uncertainty-aware computer vision pipeline for
tuberculosis (TB) triage and lesion localisation from chest X-ray images, intended for
resource-limited settings. The study adopts a target operating point of at least 90% sensitivity
and at least 70% specificity, consistent with the World Health Organization's minimum target
product profile for a TB triage test.
The system is a decision-support and triage tool, not a replacement for clinical diagnosis. The
work is divided into a core system, which must be delivered, and extensions, which are
attempted only after the core is complete (Section 18).
1. Data, Splits and Leakage Control
The system accepts chest X-ray images in DICOM or PNG format.
1.1 Primary Dataset: TBX11K
TBX11K contains 11,200 chest X-rays released at 512 × 512 pixels, with radiologist-drawn
lesion bounding boxes on TB images. Its official splits are:

Explainable Tuberculosis Screening from Chest X-Rays with Lesion Localisation

Dept of AI & ML, BMSIT&M 2026-27 Page 2
Category Train Validation Test Total
Healthy 3,000 800 1,200 5,000
Sick but non-TB 3,000 800 1,200 5,000
Active TB 473 157 294 924
Latent TB 104 36 72 212
Active and latent TB 23 7 24 54
Uncertain TB 0 0 10 10
Total TB 600 200 400 1,200
Total 6,600 1,800 2,800 11,200
The labels of the official test split are withheld for an online challenge.
1.2 How the Data Are Used
Use Source Images TB images
Training Official training split, minus the development

split

5,940 540
Development Stratified 10% of the official training split 660 60
Calibration Stratified half of the official validation split about 900 about 100
Internal test The other half of the official validation split about 900 about 100
Challenge test Official test split, submitted once at the end 2,800 400 (labels
withheld)
• Development is used for all model and preprocessing selection: CLAHE on or off, crop
margin, segmentation sanity ranges, output head, detector input size, detector confidence
threshold, training length and the OOD gate threshold. After every selection is fixed, the
development split is used once more to fit temperature scaling (Section 8).
• Calibration is used only to set the two triage thresholds.
• Internal test is used once, after every threshold, setting and model choice is locked.
• Challenge test is submitted once, and only the metrics returned by the challenge are
reported.
• All stratified splits preserve the proportions of healthy, sick but non-TB, active TB, latent
TB, and active and latent TB images. Exact counts are recorded after drawing the splits
with a fixed random seed, and the split file is committed to the repository.

Explainable Tuberculosis Screening from Chest X-Rays with Lesion Localisation

Dept of AI & ML, BMSIT&M 2026-27 Page 3
1.3 Leakage Control
• The TBX11K authors state that each X-ray corresponds to a unique individual.
• Near-duplicate images are additionally checked across all splits: image pairs whose
embedding cosine similarity exceeds a fixed threshold are inspected, and duplicates are
removed from the later split.
• No-test-tuning rule: no threshold, hyperparameter, preprocessing choice or model choice
is ever selected using internal test, challenge test or external data.

1.4 External Datasets
• Shenzhen: 662 X-rays, 336 TB and 326 normal.
• Montgomery: 138 X-rays, 58 TB and 80 normal.
Both are obtained by request from the United States National Library of Medicine. They are
never used for training, calibration or model selection.
2. Definition of the Positive Class
The primary positive class is the TBX11K image-level TB label, which includes active TB,
latent TB, and combined active and latent TB. Latent TB denotes radiographic evidence of
inactive disease in people who are neither sick nor contagious.
This definition is used because it matches the TBX11K benchmark, and because the Shenzhen
and Montgomery TB labels cannot be separated into active and latent cases.
As a secondary analysis, sensitivity is also reported for active TB alone, with latent-only
images excluded from the positive class.
3. Image Preprocessing
• DICOM decoding where required.
• Conversion to grayscale.
• Source resolution: TBX11K images are released at 512 × 512 pixels. External images are
resampled to 512 × 512 first, so that every image enters the pipeline at the same level of
detail.
• Model input size: each model then resizes the 512 × 512 image to its own required input
size and applies its own intensity normalisation, following its model card. The detector
input size is chosen as described in Section 7.

Explainable Tuberculosis Screening from Chest X-Rays with Lesion Localisation

Dept of AI & ML, BMSIT&M 2026-27 Page 4
• Contrast Limited Adaptive Histogram Equalisation (CLAHE) is tested on and off once. The
decision criterion is fixed in advance: TB sensitivity at 70% specificity on the development
split, with ROC-AUC used to break ties.

4. Quality and Out-of-Distribution (OOD) Gate
Before classification, each full image passes through a screening stage.
• Basic image-quality check: detection of blank, severely under-exposed, severely
over-exposed or heavily cropped images.
• Embedding-based OOD score: the image is embedded with RAD-DINO, and the
embedding is L2-normalised. The OOD score is the mean cosine distance to its k = 5
nearest neighbours in a reference bank made of one unaugmented embedding per training
image. The threshold is set so that 95% of development-split X-rays pass. The
single-nearest-neighbour version, k = 1, is reported as an ablation.
• Frontal/lateral view check (extension): a small head on the same embeddings. TBX11K
contains no lateral views, so lateral examples come from a public chest X-ray dataset such
as CheXpert.
Rejected images are not classified and are marked for manual review.
The embedding-distance score is evaluated as an OOD signal for non-chest images and
distribution-shifted chest radiographs, rather than assumed to detect both. It is tested on a
held-out collection of lateral chest views, radiographs of other body parts and non-medical
images, and its pass rate on Shenzhen and Montgomery is reported.
5. Lung Segmentation, Sanity Check and Region of Interest
A pretrained lung segmentation model from TorchXRayVision identifies the lung fields, so
no segmentation model is trained from scratch.
Each mask passes a segmentation sanity check:
• Total lung area within an expected range
• Two separate lung regions present
• Lung bounding box within an expected size range and not severely truncated
The ranges are fixed on the development split. If a mask fails the check, the full image is used
instead of the crop, and the case is logged. The failure rate is reported, so that segmentation
errors are not mistaken for classifier errors.

Explainable Tuberculosis Screening from Chest X-Rays with Lesion Localisation

Dept of AI & ML, BMSIT&M 2026-27 Page 5
A passing mask is used to crop the image to the lung bounding box with a margin chosen on
the development split, rather than blacking out everything outside the lungs. This keeps
findings near the hila, apices and pleura, and avoids an artificial black border that a model
could learn as a shortcut.
Radiologist lesion boxes are shifted into the coordinates of the crop. Any box that falls outside
the crop is counted and reported.
6. Classification with a Chest X-Ray Foundation Model
A pretrained RAD-DINO chest X-ray foundation model is the feature extractor.
• Core: RAD-DINO is kept frozen, its embeddings are extracted offline once, and
classification heads are trained on them. This configuration is designed to fit within a 6 GB
laptop GPU and is verified in the first week.
• Extension: the last transformer blocks are fine-tuned on a Kaggle T4 GPU.
Two output heads are trained on the same embeddings and compared:
• Three-class head: healthy, sick but non-TB, TB
• Binary head: TB, not TB
The head used for triage is chosen on the development split by TB sensitivity at 70%
specificity. If the binary head is chosen, the three-class output is still shown as auxiliary
information.
Class imbalance is handled with a weighted sampler or a logit-adjusted loss. Focal loss is not
used, because it distorts predicted probabilities and works against calibration.
An ImageNet-pretrained EfficientNet-B3, fine-tuned on the same data, is the baseline. A
TorchXRayVision classifier is an extension comparison.
7. Lesion Detection
A D-FINE detector localises TB lesions on the lung crop.
• Input size: TBX11K images are released at 512 × 512 pixels, so a larger input size cannot
add image detail; it only enlarges small lesions relative to the detector's feature maps. Input
sizes of 512 and 640 are compared on the development split by AP at IoU 0.50, and the
better one is kept.
• Classes: the detector is trained on a single TB lesion class. Active, latent, and combined
active and latent TB categories are retained for secondary stratified analysis.
• Confidence threshold: chosen on the development split and locked before testing.

Explainable Tuberculosis Screening from Chest X-Rays with Lesion Localisation

Dept of AI & ML, BMSIT&M 2026-27 Page 6
• Compute: training runs on a Kaggle T4 GPU with a small batch size; inference runs on the
laptop.
The detector outputs lesion bounding boxes, a confidence score for each box, and the number
of detected lesions.
Extensions: RT-DETRv2 as a comparison detector, and pixel-level lesion segmentation.
8. Probability Calibration
TB probabilities from the selected head are calibrated with temperature scaling. The
temperature is fitted on the development split after every model and preprocessing choice has
been fixed, and is then locked.
It is deliberately not fitted on the calibration split. The triage thresholds in Section 9 are set on
calibrated probabilities, and conformal risk control requires the score to be fixed before the
calibration data are used. Fitting both on the same images would break that requirement.
Calibration is evaluated on the internal test split with:
• A reliability diagram
• Expected Calibration Error (ECE), before and after temperature scaling

9. Triage Decision: Conformal Risk Control with a
Precision-Constrained Upper Threshold
The calibrated TB probability is converted into one of three triage outcomes using two
thresholds set on the calibration split.
• NOT TB: probability below the lower threshold.
• REFER: probability between the two thresholds; the case is sent to a radiologist.
• TB: probability above the upper threshold; the case is prioritised.

9.1 Lower Threshold: Conformal Risk Control
The controlled risk is the false-negative risk, that is, the probability that a TB image falls into
NOT TB. Let s1
, ..., s
n
be the locked calibrated TB probabilities of the n TB images in the
calibration split, and let α = 0.10. The lower threshold t is the largest value satisfying:

(1 + number of calibration TB images with si

< t) / (n + 1) ≤ α

With about 100 TB images in the calibration split, at most 9 of them may lie below the
threshold. For a new TB image that is exchangeable with the calibration images, this bounds

Explainable Tuberculosis Screening from Chest X-Rays with Lesion Localisation

Dept of AI & ML, BMSIT&M 2026-27 Page 7
the expected probability of it being placed in NOT TB at α. This is finite-sample risk control
under the assumptions of the procedure, not a guarantee for every hospital. Whether the target
is maintained on external data is measured. The procedure is implemented directly in NumPy.
9.2 Upper Threshold: Precision-Constrained
The upper threshold is not conformal. On the calibration split, it is the lowest value at which the
precision of cases above it reaches a target agreed with the guide, for example 80%. If the upper
threshold would fall below the lower threshold, it is set equal to the lower threshold. If no value
reaches the precision target, the TB category is left empty and every screen-positive case is
labelled REFER.
For the sensitivity and specificity targets, both TB and REFER count as screen-positive,
because both send the case to a clinician.
9.3 Detector Link Rule
A case placed in NOT TB is moved to REFER when the detector identifies a lesion above its
development-selected confidence threshold. A detected lesion is treated only as a reason for
clinical review; it does not establish TB. Whether the rule improves screening sensitivity, and at
what cost in REFER rate, is measured.
10. Explainability
Each prediction comes with two kinds of visual evidence:
• Classifier heatmap: Grad-CAM applied to the last RAD-DINO block, using the reshape
step required for vision transformers.
• Detector evidence: lesion boxes with confidence scores.
The heatmap is scored as localisation agreement with annotated TB regions, on internal test
TB images that have boxes:
• Pointing accuracy: the share of images whose heatmap peak lies inside an annotated box.
• Energy in box: the share of total heatmap intensity that lies inside annotated boxes.
High agreement means the heatmap overlaps the available annotations under these metrics. It is
not treated as proof that an explanation is clinically correct, because TBX11K boxes are not
complete pixel-level outlines of every abnormality.
11. Risk-Ranked Worklist
Each case displays:

Explainable Tuberculosis Screening from Chest X-Rays with Lesion Localisation

Dept of AI & ML, BMSIT&M 2026-27 Page 8
• Calibrated TB probability
• Triage category: TB, REFER or NOT TB
• Whether the detector link rule changed the category
• Number of detected lesions and their confidence
• Gate status
• Heatmap and lesion boxes
The worklist is sorted by calibrated TB probability. The triage category is shown alongside, and
no claim is made that the category order itself represents clinical risk.
12. Evaluation
All results are reported on the internal test split unless stated otherwise.
Targets: at least 90% sensitivity and at least 70% specificity for TB, with TB and REFER
counted as screen-positive.
Uncertainty reporting is mandatory. Every result states the number of TB and non-TB
images it is computed on. Proportions such as sensitivity and specificity carry 95% Wilson
confidence intervals. ROC-AUC, ECE and average precision carry 95% bootstrap confidence
intervals.
12.1 Classification
• ROC-AUC for TB against not TB
• Three-class confusion matrix
• Secondary: sensitivity for active TB alone

12.2 Calibration
• ECE before and after temperature scaling
• Reliability diagram

12.3 Triage
The following are separate measures and are reported separately, each with and without the
detector link rule.

Explainable Tuberculosis Screening from Chest X-Rays with Lesion Localisation

Dept of AI & ML, BMSIT&M 2026-27 Page 9
Group Measure Compared against

Screen-positive (TB or REFER) Sensitivity 90% target
Screen-positive (TB or REFER) Specificity 70% target
Screen-positive (TB or REFER) Positive predictive value (PPV) Reported
Screen-positive (TB or REFER) Negative predictive value (NPV) Reported
TB category Precision Agreed precision target
TB category Share of all cases flagged TB Reported
REFER category Share of all cases sent to REFER Reported

12.4 Lesion Detection
• AP at IoU 0.50 for the single TB lesion class
• Image-level lesion sensitivity: the share of TB images with at least one predicted box
overlapping an annotated box at IoU 0.50 or above
• Free-response ROC (FROC): lesion sensitivity at 1, 2 and 4 false positives per image
• Secondary: results stratified by active, latent, and combined active and latent TB

12.5 Explainability
• Pointing accuracy
• Energy in box

12.6 Quality and OOD Gate
• ROC-AUC of the OOD score for separating chest X-rays from the held-out OOD collection
• Rejection rate on that collection
• Pass rate on Shenzhen and Montgomery

12.7 Segmentation
• Sanity-check failure rate

12.8 Challenge Test
• The metrics returned by the TBX11K online challenge, from a single submission

Explainable Tuberculosis Screening from Chest X-Rays with Lesion Localisation

Dept of AI & ML, BMSIT&M 2026-27 Page 10
13. External Validation
The locked system is evaluated on Shenzhen and Montgomery.
External datasets are used to assess binary TB-versus-not-TB generalisation, not three-class
classification, because they label only TB and normal. They contain no lesion boxes, so
detection and explainability are measured on TBX11K only.
For each external dataset the following are reported, each with its sample size and 95%
confidence interval:
• ROC-AUC, sensitivity and specificity
• Whether the conformal false-negative target is maintained
• Gate pass rate
The size and direction of any change from internal results are not assumed in advance; they are
measured and reported as an empirical finding. Because Montgomery contains only 58 TB
images, its confidence intervals will be wide, and conclusions from it are stated accordingly.
Extension, recalibration on external data: a small labelled portion of Shenzhen is used to
refit temperature scaling and the triage thresholds, and the rest of Shenzhen is used for testing.
This shows how many local images are needed to restore the target in a new hospital. It is
reported separately from the main external result.
14. Comparative Experiments
Each experiment answers one question. Choices are made on the development split; reported
results come from the internal test split and external data.

Explainable Tuberculosis Screening from Chest X-Rays with Lesion Localisation

Dept of AI & ML, BMSIT&M 2026-27 Page 11
Question Compared Deciding metric
Does a chest X-ray backbone
help?

RAD-DINO vs ImageNet
EfficientNet-B3 (vs
TorchXRayVision, extension)

ROC-AUC, internal and
external

Which output head? Three-class vs binary TB sensitivity at 70%
specificity

Does lung cropping help? Full image vs lung crop with margin Drop in AUC from internal to
external; pointing accuracy
Does CLAHE help? CLAHE on vs off TB sensitivity at 70%
specificity
Does calibration help? Before vs after temperature scaling ECE
Does conformal triage meet its
target?

ROC-selected threshold vs conformal
lower threshold

Achieved sensitivity; REFER
rate

Does the detector link rule help? With vs without the rule Sensitivity; REFER rate
Does k-nearest-neighbour OOD
scoring help?

k = 5 vs k = 1 OOD ROC-AUC
Which detector input size? 512 vs 640 AP at IoU 0.50
Which detector? (extension) D-FINE vs RT-DETRv2 AP at IoU 0.50; FROC

Explainable Tuberculosis Screening from Chest X-Rays with Lesion Localisation

Dept of AI & ML, BMSIT&M 2026-27 Page 12
15. Complete System Pipeline

Chest X-ray input

DICOM or PNG; decode, grayscale, resize to 512 × 512

Quality and OOD gate
image quality; RAD-DINO k-NN OOD score

Accepted
continues to analysis

Rejected
manual review

Lung segmentation
pretrained model with sanity check

Sanity check passed
lung crop with margin

Sanity check failed
full image, logged

Image for analysis
crop or full-image fallback

RAD-DINO + selected head
TB probability

D-FINE detector
single TB lesion class

Temperature scaling
fitted on development split

Lesion boxes
locked confidence threshold

Triage thresholds
conformal lower, precision upper

Detector link rule
lesion moves NOT TB to REFER

TB / REFER / NOT TB
screen-positive = TB or REFER

Explainability

Grad-CAM, lesion boxes, localisation agreement

Worklist, dashboard and PDF report
sorted by calibrated TB probability

Figure 1: Complete system pipeline. A rejected image stops at manual review, and a failed segmentation falls
back to the full image. The detector runs alongside the classifier and joins at the triage decision only through
the link rule, where a detected lesion is a reason for review and does not establish TB.

Explainable Tuberculosis Screening from Chest X-Rays with Lesion Localisation

Dept of AI & ML, BMSIT&M 2026-27 Page 13
16. Implementation Order
1. Dataset access, split file and leakage checks
2. Preprocessing
3. Lung segmentation, sanity check and crop
4. RAD-DINO embeddings
5. Three-class and binary classification heads
6. EfficientNet-B3 baseline
7. OOD gate, with its threshold locked on the development split
8. Temperature scaling
9. Conformal and precision-constrained triage thresholds
10. D-FINE detector
11. Detector link rule
12. Internal test evaluation, run once after everything above is locked
13. External validation
14. Explainability scoring
15. Dashboard and PDF report
16. Extensions
This order produces a complete, defensible core experiment even if later components or
extensions are not finished.
17. Research Comparison
The project compares architectures on its own data rather than relying on benchmark claims
from unrelated datasets.
• Classification: RAD-DINO (primary), ImageNet EfficientNet-B3 (baseline),
TorchXRayVision (extension)
• Lesion detection: D-FINE (primary), RT-DETRv2 (extension)

18. Scope
18.1 Core System (Must Be Delivered)
• Data splits, leakage checks and the no-test-tuning rule

Explainable Tuberculosis Screening from Chest X-Rays with Lesion Localisation

Dept of AI & ML, BMSIT&M 2026-27 Page 14
• Lung segmentation, sanity check and crop with margin
• Frozen RAD-DINO with three-class and binary heads, and the EfficientNet-B3 baseline
• D-FINE lesion detector with the link rule
• Temperature scaling and conformal triage
• Grad-CAM with localisation agreement scoring
• Embedding-based OOD gate
• Internal evaluation, one challenge submission, and external validation, all with confidence
intervals
• Streamlit dashboard and PDF report

18.2 Extensions (Only After the Core Is Complete)
• Fine-tuning of RAD-DINO
• TorchXRayVision classifier comparison
• RT-DETRv2 detector comparison
• Frontal/lateral view head
• Pixel-level lesion segmentation
• Recalibration of triage thresholds on a small portion of external data

19. Software Requirements
• Python 3.11
• PyTorch, torchvision, timm
• TorchXRayVision (pretrained lung segmentation; extension classifier)
• RAD-DINO model checkpoint
• Official D-FINE implementation; RT-DETRv2 implementation for the extension
• pytorch-grad-cam
• scikit-learn (nearest-neighbour search, metrics), SciPy and statsmodels (confidence
intervals)
• OpenCV, NumPy, pandas, Albumentations
• pydicom
• Streamlit, Matplotlib, ReportLab

Explainable Tuberculosis Screening from Chest X-Rays with Lesion Localisation

Dept of AI & ML, BMSIT&M 2026-27 Page 15
• Git and Git LFS
Model licences are checked before use. TorchXRayVision is released under Apache-2.0. The
RAD-DINO weights carry a Microsoft research licence, which permits academic use.
20. Hardware Requirements
• Laptop GPU: NVIDIA RTX 4050 with 6 GB of memory. The frozen-backbone
configuration is designed to fit within this limit through offline embedding extraction, and
this is verified in the first week.
• Cloud GPU: Kaggle or Google Colab NVIDIA T4 for detector training, the
EfficientNet-B3 baseline and any backbone fine-tuning. Detector training is expected to be
the main GPU memory bottleneck.
• Deployment: minimum 8 GB of system RAM; a GPU is recommended for faster inference.
• Storage: about 30 GB for datasets, embeddings, checkpoints and outputs, with 50 GB
recommended as a margin.

21. Final Objective
The final system is a TB triage pipeline that:
• Screens out unsuitable inputs using an evaluated quality and OOD gate
• Focuses analysis on the lung region, with a fallback when segmentation fails
• Classifies chest X-rays using a chest X-ray foundation model, compared against an
ImageNet baseline
• Localises suspected TB lesions
• Produces calibrated TB probabilities
• Makes TB / REFER / NOT TB triage decisions with finite-sample control of the
false-negative risk under the assumptions of the conformal procedure, and measures
whether that control holds on external data
• Scores its visual explanations for localisation agreement with annotated TB regions
• Presents a probability-sorted worklist in a clinician-oriented dashboard and report
• Reports every result with its sample size and confidence interval
• Targets at least 90% sensitivity and at least 70% specificity for TB

Explainable Tuberculosis Screening from Chest X-Rays with Lesion Localisation

Dept of AI & ML, BMSIT&M 2026-27 Page 16
The system is intended as a decision-support and triage tool, not as a replacement