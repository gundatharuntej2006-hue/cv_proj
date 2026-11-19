BMS INSTITUTE OF TECHNOLOGY AND MANAGEMENT
(An Autonomous Institute affiliated to VTU, Belagavi, Approved by AICTE New Delhi)

Yelahanka, Bengaluru 560064

Department of Artificial Intelligence and Machine Learning

Mini Project Proposal - BAI506
Academic Year 2026-27

Section and
Batch No:
AIML – A
A-10

Guide Name:
Dr. Niranjanamurthy M

Submission Date:
16/09/2026

Mini Project Title :

Explainable Tuberculosis Screening from Chest X-Rays with Lesion Localisation for

Triage in Resource-Limited Settings
Sl No USN Name
1 1BY24AI048 GUNDA THARUN TEJ
2 1TD24AI036 ATUL DHULL
3 1TD24AI094 KUNAL
4 1TD24AI150 RITIKA GIRISH KULKARNI
Project Category Research

Signature of Guide

Explainable Tuberculosis Screening from Chest X-Rays with Lesion Localisation

Dept of AI & ML, BMSIT&M 2026-27 1
Initial Project Proposal
Tuberculosis (TB) is one of the world's deadliest infectious diseases, and India carries the
largest share of the burden: 25% of global cases according to the World Health Organization
(WHO) Global Tuberculosis Report 2025, which reports data for the year 2024. Chest X-ray
(CXR) is a sensitive, low-cost first test, but India has only about one radiologist per lakh
people, most of them in large cities, so X-rays from district hospitals and screening camps can
wait days for a reading. Since 2021, WHO guidelines have recommended computer-aided
detection (CAD) as an alternative to human reading of digital chest X-rays for TB screening
and triage in people aged 15 years and older. Most academic CAD models are trained on small
datasets such as Shenzhen (662 images) and Montgomery (138 images), output only a TB or
normal label, and cannot tell TB apart from other lung diseases. This project proposes an
explainable CAD system that both classifies and localises TB. Using the Tuberculosis X-ray
(TBX11K) dataset of 11,200 radiographs with radiologist-drawn bounding boxes, a
Convolutional Neural Network (CNN) classifier separates healthy, sick-but-non-TB and TB
images, while an object detector marks active and latent TB regions. A U-Net lung
segmentation step restricts analysis to the lung fields, and Gradient-weighted Class Activation
Mapping (Grad-CAM) heatmaps are shown alongside the detected boxes. The decision
threshold is set to meet the WHO minimum targets for a TB triage test, which are at least 90%
sensitivity and at least 70% specificity, and generalisation is checked on the unseen Shenzhen
and Montgomery sets. The result is a web-based triage tool that ranks X-rays so likely TB
cases reach a radiologist first; it supports clinical decisions and does not replace them.
1. Introduction
Pulmonary TB shows up on a chest X-ray as consolidation, cavities, nodules or fibrosis, often
in the upper zones of the lungs. Deep CNNs have reached radiologist-level performance on
some chest X-ray tasks, which makes automated TB screening realistic. WHO has
recommended computer-aided detection software as an alternative to human reading of digital
chest X-rays for TB screening and triage in people aged 15 years and older since 2021, so the
idea is now policy and not only research.
The hard part is not spotting abnormal. It is telling TB apart from pneumonia and other lung
diseases that look similar, and showing the evidence so a clinician can check it. Independent
multi-country testing in 2026 found that only three of seven commercial CAD algorithms
reached the WHO minimum triage accuracy, and an earlier external validation of twelve
products on a single South African survey found the area under the receiver operating

Explainable Tuberculosis Screening from Chest X-Rays with Lesion Localisation

Dept of AI & ML, BMSIT&M 2026-27 2
characteristic curve (AUC) spread from about 0.90 down to below 0.80, with every product
doing worst in older people, people living with the human immunodeficiency virus (HIV) and
people with previous TB. Accuracy therefore has to be demonstrated at a stated operating point
on data the model has not seen, not claimed from a single headline number.
1.1 Motivation
India has more TB patients than any other country, and screening camps increasingly use
portable X-ray units. That moves the bottleneck from taking the image to reading it. A 2024
review in The Lancet Regional Health - Southeast Asia reports that only about 20,500
radiologists are registered with the Indian Society of Radiology and Imaging, roughly one per
100,000 people compared with about one per 10,000 in the United States, and that most of
them work in larger cities and towns. A tool that sorts a day's X-rays by risk, and marks the
suspicious region on each, directly shortens the time to diagnosis for the patients who need it
most.
2. Literature Survey
The published work falls into three groups. First are the small public benchmarks that most
student projects still use. Second are the research benchmarks that added radiologist-drawn
lesion boxes and turned TB screening into a joint classification and detection problem. Third
are the large clinical and commercial systems, together with the independent evaluations
published between 2024 and 2026 that measure them against the WHO triage target. The table
below takes representative work from each group, and each row states what the work leaves
open for this project.
Ref Work Approach Key result or note
[6] Jaeger et al., 2014.
Shenzhen set of 662
chest X-rays (326
normal, 336 with TB)
and Montgomery
County set of 138 chest
X-rays (80 normal, 58
with TB); the
Montgomery set
includes manual lung
masks.

Two de-identified frontal chest X-
ray collections released by the

United States National Library of
Medicine, with image-level TB
labels and clinical readings.

Standard early benchmarks and
still the usual external test sets.
Image-level labels only, very
small, and access is by request
rather than open download.

[7] Liu et al., TBX11K,
CVPR 2020.

11,200 X-rays (5,000 healthy, 5,000
sick but non-TB, 1,200 TB) with
boxes drawn by one radiologist with
5 to 10 years of TB experience and
checked by a second with more than
10 years; boxes are typed as active
or latent TB. Detectors are reformed

First TB detection benchmark and
the dataset this project uses. The
2,800-image test set is held back as
an online challenge, so its labels
are not available offline.

Explainable Tuberculosis Screening from Chest X-Rays with Lesion Localisation

Dept of AI & ML, BMSIT&M 2026-27 3
Ref Work Approach Key result or note

to do classification and detection at
once.
[8] Liu et al., SymFormer,
IEEE Transactions on
Pattern Analysis and
Machine Intelligence,
2024.

Symmetric Search Attention and
Symmetric Positional Encoding
exploit the left-right symmetry of
the chest, doing classification and
TB area detection jointly.

Reported as state of the art on
TBX11K and still the published
best as far as we could check. It is
a heavy Transformer model, so it is
a comparison point rather than a
target for a 6 GB laptop GPU.

[9] Kazemzadeh et al.,
2021.

Deep learning system trained and
tested on chest X-rays from 10
countries including India, compared
against radiologist panels.

AUC 0.90 (95% confidence
interval 0.87 to 0.92). Sensitivity
88% against 75% mean sensitivity
for India-based radiologists, but
specificity 79% against 84% for
those radiologists. Model and data
are not public, and the paper
remains a preprint.

[10] Subramanian et al.,
2025.

Multi-pathology chest X-ray system

trained on more than 5 million X-
rays from Indian healthcare systems

and deployed in 17 healthcare
systems in India.

Reports up to 98% precision and

over 95% recall for multi-
pathology classification. Shows

clinical feasibility in India, but it is
proprietary, it is not TB-specific,
and the paper is an unreviewed
preprint.

[11] Worodria et al., Annals
of the American
Thoracic Society, 2026.

Independent head-to-head
comparison of 7 CAD algorithms on
adults with at least two weeks of
cough in India, Madagascar, the
Philippines, South Africa, Tanzania,
Uganda and Vietnam.

At 90% sensitivity, only CAD4TB
(73.5% specificity), qXR (70.3%)
and INSIGHT CXR (70.1%) met
the WHO minimum triage target.

Setting-specific or population-
specific thresholds improved

results, which is direct evidence
that the operating point must be
tuned, not inherited.

[12] Qin et al., The Lancet
Digital Health, 2024.

External validation of 12
commercial CAD products on a
case-control sample from a South
African TB prevalence survey,
scored against microbiological
reference standards.

Accuracy varied widely: Lunit
0.902 and Nexus 0.897 AUC at the
top, three products below 0.80. All
products except one performed
worse for people aged 55 and over
than for people aged 15 to 34, and
worse for people with previous TB
and people living with HIV.

[13] Bosman et al., Clinical
Infectious Diseases,
2024.

CAD4TB version 7 and C-reactive
protein evaluated as TB triage tests
at health facilities in Lesotho and
South Africa.

AUC 0.87 (95% confidence
interval 0.84 to 0.91), sensitivity
89.8% and specificity 68.2%, so it
almost met but did not meet the
WHO minimum triage criteria. A
useful reminder that 90% and 70%
are hard numbers to hit on real
data.

Explainable Tuberculosis Screening from Chest X-Rays with Lesion Localisation

Dept of AI & ML, BMSIT&M 2026-27 4
2.1 Limitations of Existing System
• Small image-level datasets let models learn shortcuts such as text markers, borders and
hospital artefacts that do not carry over to a new hospital. Work presented at EUVIP 2024
showed that a TB model can score almost perfectly on its test set while looking at the
wrong parts of the image.
• Binary TB-versus-normal training hides the hard case. A model that never saw other lung
diseases flags every abnormal X-ray as TB, which is useless in a screening camp where
most abnormal films are not TB.
• A single probability with no localisation gives the radiologist nothing to verify, and
nothing to disagree with.
• The strongest systems are proprietary, and independent evaluations show that accuracy
varies a lot between products and between patient groups, so a published headline AUC
does not transfer to a new setting.
• Class imbalance. TBX11K has 1,200 TB images against 10,000 non-TB images, and inside
the TB class there are only 212 latent TB images, 54 with both active and latent findings
and 10 uncertain, so latent TB detection will be weak no matter what we do.
• The TBX11K test labels are withheld for the online challenge, so day-to-day development
cannot use the official test set and needs an internal validation split carved out of the
training data.
2.2 Outcome of the Literature Survey
A three-class classifier that includes sick but not TB, lesion boxes and Grad-CAM heatmaps as
visible evidence, a lung mask that blocks shortcuts outside the lung fields, a decision threshold
tuned to the WHO triage targets rather than to accuracy, and a test on hospitals the model never
saw. Everything is built with open-source tools and public de-identified data, and the
deliverable is a batch triage dashboard that a district hospital could actually sit in front of.
3. Problem Statement
To develop an explainable computer-aided detection system that identifies tuberculosis on
chest X-rays, distinguishes it from other lung abnormalities, localises the affected regions, and
operates at a sensitivity and specificity suitable for triage in resource-limited screening settings.

Explainable Tuberculosis Screening from Chest X-Rays with Lesion Localisation

Dept of AI & ML, BMSIT&M 2026-27 5
3.1 Objectives
1. Train a three-class classifier (healthy / sick-but-non-TB / TB) and a TB-region detector on
TBX11K, reaching at least 90% sensitivity at at least 70% specificity for TB on a held-out
internal split, and report AUC and detection average precision on that split, on one
submission to the TBX11K online challenge test set, and on the unseen Shenzhen and
Montgomery sets.
2. Deliver a web-based triage tool that shows TB probability, lesion boxes and a heatmap for
each X-ray and ranks a batch by risk, processing each image in under 2 s on a laptop GPU.
4. Proposed Methodology
The system is a straight pipeline with one fork. An X-ray comes in, is checked and cleaned,
and is then passed at the same time to a classifier and to a detector. Their outputs are merged,
calibrated and shown together on a dashboard. Splitting the work this way means the classifier
decides the triage score while the detector supplies the evidence, and either branch can be
improved without breaking the other.

Input
DICOM or PNG, single or batch

↓
Quality gate
reject lateral and non-X-ray views

↓
Pre-process
CLAHE, U-Net lung mask, crop
↓ ↓

Three-class classifier
EfficientNet-B3 or ConvNeXt-Tiny

TB lesion detector
boxes: active_tb, latent_tb

↓ ↓
Explain and calibrate
Grad-CAM, temperature scaling

↓
Triage worklist
Streamlit, risk-ranked, PDF export

Figure 1. System pipeline. The classifier and the detector run as parallel branches on the same pre-
processed, lung-masked image: the classifier produces the triage score, the detector produces the visible

evidence, and the two are merged before calibration and display.
Stage What happens Tools
Input and quality
gate

Accept frontal posteroanterior chest X-rays, single or
batch, as DICOM or PNG. A small check rejects

pydicom, OpenCV, Pillow

Explainable Tuberculosis Screening from Chest X-Rays with Lesion Localisation

Dept of AI & ML, BMSIT&M 2026-27 6
Stage What happens Tools

lateral views and images that are not chest X-rays
before anything else runs. Convert to 8-bit grayscale
and resize to 512 x 512.

Data preparation
and splitting

Build fixed, seeded splits. Because the TBX11K test
labels are withheld, carve an internal validation split
from the 6,600-image training set and keep the official
1,800-image validation set as the internal test set.
Shenzhen and Montgomery are never trained on.

scikit-learn, pandas,
versioned splits file

Pre-process Contrast-limited adaptive histogram equalisation
(CLAHE) to even out exposure, then a U-Net trained
on the Montgomery lung masks isolates the lungs. The
mask is used to crop and to blank everything outside
the lung fields, which removes text markers and
borders the model could otherwise use as shortcuts.

OpenCV, PyTorch, smp
(segmentation models)

Classify An ImageNet-pretrained EfficientNet-B3 or
ConvNeXt-Tiny predicts three classes: healthy,
sick_non_tb, tb. Focal loss or class weights handle the
1,200 against 10,000 imbalance, with light
augmentation (random crop, small rotation,
brightness).

PyTorch, timm,
Albumentations

Localise An object detector trained on the TBX11K boxes
marks suspected TB regions and labels each box
active_tb or latent_tb. Latent TB results will be
reported separately and cautiously because only 212
images carry that label.

Ultralytics YOLO11
(AGPL-3.0); torchvision
Faster R-CNN or RT-DETR
as a permissive fallback

Explain and
calibrate

Grad-CAM produces a heatmap from the classifier,
temperature scaling on the validation split turns the
raw score into an honest probability, and the operating
threshold is read off the validation ROC curve at the
point that gives sensitivity of at least 0.90 while
keeping specificity of at least 0.70.

pytorch-grad-cam, scikit-
learn, NumPy

Triage worklist A dashboard takes a folder of X-rays, sorts them by
TB risk, overlays boxes and the heatmap on each
image, and exports a per-patient PDF stamped for
clinician review, decision support only.

Streamlit, ReportLab or
FPDF, Matplotlib

4.1 Experimental Plan and Validation
Five experiments are planned. Each names what it is trained on, what it is tested on and the
single number that decides whether it passed. The rule we follow throughout is that nothing is
reported on data that was used to pick a threshold or a checkpoint, and cross-dataset results are
always reported separately from in-dataset results.
Item What is measured Method Reference or metric
E1. U-Net lung
segmentation

Quality of the lung
mask

U-Net trained on the 138
Montgomery images with
their manual left and right
lung masks, with five-fold
cross-validation

Tested on the held-out
Montgomery folds. Dice
coefficient, target at least
0.94. Masks for TBX11K
are then produced by this

Explainable Tuberculosis Screening from Chest X-Rays with Lesion Localisation

Dept of AI & ML, BMSIT&M 2026-27 7
Item What is measured Method Reference or metric
model and spot-checked by
eye on 50 images.

E2. Three-class
classifier

TB detection at the
triage operating point

EfficientNet-B3 or
ConvNeXt-Tiny on the
TBX11K training split,
lung-masked, with focal loss
or class weights

Tested on the internal
validation split. AUC for TB
versus not-TB, plus
sensitivity and specificity at
the chosen threshold.
Target: sensitivity at least
90% with specificity at least
70%.

E3. Shortcut ablation Does lung masking
stop the model
learning from the
background?

Same classifier trained
twice: once on full images,
once on lung-masked
images

Both tested on the unseen
Shenzhen and Montgomery
sets. Metric: the drop in
AUC from the internal split
to the external sets. A
smaller drop for the masked
model supports the design
choice.

E4. TB lesion
detector

Are the boxes in the
right place?

YOLO11 (or the torchvision
fallback) trained on the
TBX11K training-split
boxes for active_tb and
latent_tb

Tested on the internal
validation split. Average
precision at intersection over
union 0.50 (AP50), reported
separately for active TB and
latent TB, and one scored
submission to the TBX11K
online challenge.

E5. Calibration and
external
generalisation

Are the probabilities
honest, and do they
hold up elsewhere?

Temperature scaling fitted
on the internal validation
split only

Tested on Shenzhen and
Montgomery. Expected
calibration error before and
after scaling, plus sensitivity
and specificity at the frozen
threshold, reported per
dataset.

E6. Dashboard
latency

Is it fast enough to
use?

Full pipeline on the laptop
RTX 4050 6 GB GPU and
on CPU

Median and 95th percentile
wall-clock time per image
over a 100-image batch.
Target: under 2 s per image
on GPU, under 10 s on
CPU.

4.2 Evaluation
Evaluation is deliberately separated into three levels. In-dataset results come from an internal
validation split cut out of the TBX11K training set, because the labels for the official 2,800-
image test set are withheld for the online challenge and cannot be used for day-to-day work; we
will make at most a small number of submissions to that challenge so it stays an honest test.
Cross-dataset results come from Shenzhen and Montgomery, which the model never trains on
and which come from different hospitals and different X-ray machines, so they are the real test
of whether the model learned lung findings or hospital artefacts. Clinical relevance is judged

Explainable Tuberculosis Screening from Chest X-Rays with Lesion Localisation

Dept of AI & ML, BMSIT&M 2026-27 8
only at the operating point: we freeze one threshold on the validation ROC curve and then
report sensitivity, specificity, positive and negative predictive value at that frozen threshold on
every test set, alongside AUC. The pass mark is the WHO minimum target product profile for a
TB triage test, at least 90% sensitivity and at least 70% specificity, and we will state plainly
when we fall short rather than moving the threshold. For the detector we report average
precision at intersection over union 0.50, separately for active TB and latent TB, and we will
report the latent TB number with a warning attached because only 212 training images carry
that label. Calibration is checked with a reliability diagram and expected calibration error
before and after temperature scaling. Finally, we will show Grad-CAM heatmaps for a sample
of correct and incorrect predictions and ask our guide whether the highlighted regions match
what a reader would look at, because a confident model looking at the wrong place is the
failure mode this project is built to expose.
5. System Requirement Specifications
5.1 Functional Requirements
• Upload a single X-ray or a batch of X-rays in DICOM or PNG format.
• Automatically reject images that are not frontal chest X-rays and tell the user why.
• Output the three class probabilities, the lesion bounding boxes with their active or latent
label, and a Grad-CAM heatmap for each image.
• Rank a batch of X-rays by TB risk and show them as a worklist, highest risk first.
• Export a per-patient PDF report containing the image, the overlay, the probability and a
decision-support disclaimer.
• Let the user see and change the operating threshold, with the WHO triage default clearly
marked.
• Log every prediction with the model version and the split version used, so any result in the
report can be traced back.
5.2 Non-Functional Requirements
• Safety: the operating point favours sensitivity of at least 90%, because a missed TB case
costs more than a false alarm.
• Performance: under 2 s per image on a laptop GPU and under 10 s per image on CPU.
• Privacy: DICOM headers are anonymised on ingest and no patient identifiers are written to
disk or logs.

Explainable Tuberculosis Screening from Chest X-Rays with Lesion Localisation

Dept of AI & ML, BMSIT&M 2026-27 9
• Ethics: public de-identified data only; every screen and every export is labelled decision
support, not diagnosis.
• Reproducibility: fixed random seeds, versioned model weights, and a splits file committed
to the repository so any result can be re-run.
• Usability: the dashboard must be usable by a technician with no machine learning
background, on a 1366 x 768 laptop screen.
• Robustness: an unreadable or corrupt file must produce a clear message and must not stop
the rest of the batch.
• Scope honesty: the interface and the report state that the system is a triage aid validated
only on public research datasets, and that it has not undergone prospective clinical
validation.
5.3 Hardware and Software Requirements

Hardware Software

Laptop with an NVIDIA GPU of at least 6 GB
VRAM (an RTX 4050 is the development machine)
for training and for the demo.

Python 3.11, PyTorch, timm, Albumentations,
segmentation-models-pytorch.

Free Kaggle (Tesla T4, 16 GB) or Google Colab
sessions for the longer training runs and for the
detector.

Ultralytics (YOLO11) for detection, with
torchvision detection models available as a
permissively licensed fallback.

Any laptop with 8 GB RAM to run the dashboard in
CPU mode.

pydicom, OpenCV, NumPy, pandas, scikit-learn for
input handling, metrics and calibration.

About 30 GB of free disk space for TBX11K,
Shenzhen, Montgomery, checkpoints and cached
masks.

pytorch-grad-cam for explanations; Streamlit,
Matplotlib and ReportLab for the dashboard and the
PDF export.
Git and Git LFS for code, weights and the versioned
splits file.

References
[1] World Health Organization, "Global tuberculosis report 2025," World Health Organization, Geneva,

Switzerland, Nov. 2025. [Online]. Available: https://www.who.int/teams/global-programme-on-
tuberculosis-and-lung-health/tb-reports/global-tuberculosis-report-2025

[2] World Health Organization, "WHO consolidated guidelines on tuberculosis. Module 2: Screening -
Systematic screening for tuberculosis disease," World Health Organization, Geneva, Switzerland,
2021, ISBN 978-92-4-002267-6. [Online]. Available:
https://www.ncbi.nlm.nih.gov/books/NBK569338/
[3] World Health Organization, "High priority target product profiles for new tuberculosis diagnostics:
Report of a consensus meeting, 28-29 April 2014, Geneva, Switzerland," Rep.

Explainable Tuberculosis Screening from Chest X-Rays with Lesion Localisation

Dept of AI & ML, BMSIT&M 2026-27 10
WHO/HTM/TB/2014.18, World Health Organization, Geneva, Switzerland, 2014. [Online].
Available: https://www.who.int/publications/i/item/WHO-HTM-TB-2014.18
[4] World Health Organization, "Target product profiles for tuberculosis screening tests," World Health
Organization, Geneva, Switzerland, Aug. 2025. [Online]. Available:

https://www.who.int/news/item/13-08-2025-who-releases-new-target-product-profiles-for-
tuberculosis-screening-tests

[5] M. Kohli, A. Korobitsyn, N. Ismail et al., "WHO target product profile for TB detection at
peripheral settings: 2024 update," PLOS Global Public Health, vol. 5, no. 6, art. no. e0004612,
2025. [Online]. Available: https://pmc.ncbi.nlm.nih.gov/articles/PMC12157113/
[6] S. Jaeger, S. Candemir, S. Antani, Y.-X. J. Wang, P.-X. Lu and G. Thoma, "Two public chest X-ray
datasets for computer-aided screening of pulmonary diseases," Quantitative Imaging in Medicine
and Surgery, vol. 4, no. 6, pp. 475-477, 2014. [Online]. Available:
https://qims.amegroups.org/article/view/5132/6030
[7] Y. Liu, Y.-H. Wu, Y. Ban, H. Wang and M.-M. Cheng, "Rethinking computer-aided tuberculosis
diagnosis," in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), 2020,
pp. 2646-2655. [Online]. Available:

https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_Rethinking_Computer-
Aided_Tuberculosis_Diagnosis_CVPR_2020_paper.html

[8] Y. Liu, Y.-H. Wu, S.-C. Zhang, L. Liu, M. Wu and M.-M. Cheng, "Revisiting computer-aided
tuberculosis diagnosis," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46,
no. 4, pp. 2316-2332, Apr. 2024. [Online]. Available: https://arxiv.org/abs/2307.02848
[9] S. Kazemzadeh, J. Yu, S. Jamshy et al., "Deep learning for detecting pulmonary tuberculosis via
chest radiography: An international study across 10 countries," arXiv preprint arXiv:2105.07540,
2021. [Online]. Available: https://arxiv.org/abs/2105.07540
[10] B. Subramanian, S. Jaikumar, P. Shastry et al., "Autonomous AI for multi-pathology detection in
chest X-rays: A multi-site study in the Indian healthcare system," arXiv preprint
arXiv:2504.00022, 2025. [Online]. Available: https://arxiv.org/abs/2504.00022
[11] W. Worodria, R. Castro, S. V. Kik et al., "A multi-country head-to-head accuracy comparison of
automated chest X-ray algorithms for tuberculosis," Annals of the American Thoracic Society,
vol. 23, no. 5, pp. 728-736, 2026. [Online]. Available:
https://academic.oup.com/annalsats/article/23/5/728/8450244
[12] Z. Z. Qin, M. van der Walt, S. Moyo et al., "Computer-aided detection of tuberculosis from chest
radiographs in a tuberculosis prevalence survey in South Africa: External validation and modelled
impacts of commercially available artificial intelligence software," The Lancet Digital Health,
vol. 6, no. 9, pp. e605-e613, 2024. [Online]. Available:
https://pmc.ncbi.nlm.nih.gov/articles/PMC11339183/
[13] S. Bosman et al., "Evaluation of C-reactive protein and computer-aided analysis of chest X-rays as
tuberculosis triage tests at health facilities in Lesotho and South Africa," Clinical Infectious
Diseases, vol. 79, no. 5, pp. 1293-1302, 2024. [Online]. Available:
https://pmc.ncbi.nlm.nih.gov/articles/PMC11581699/
[14] A. Chandramohan, V. Krothapalli, A. Augustin et al., "Teleradiology and technology innovations
in radiology: Status in India and its role in increasing access to primary health care," The Lancet
Regional Health - Southeast Asia, vol. 23, art. no. 100195, 2024. [Online]. Available:
https://pmc.ncbi.nlm.nih.gov/articles/PMC10884973/

Explainable Tuberculosis Screening from Chest X-Rays with Lesion Localisation

Dept of AI & ML, BMSIT&M 2026-27 11
[15] O. Ronneberger, P. Fischer and T. Brox, "U-Net: Convolutional networks for biomedical image
segmentation," in Proc. Int. Conf. Medical Image Computing and Computer-Assisted Intervention
(MICCAI), 2015, pp. 234-241. [Online]. Available: https://arxiv.org/abs/1505.04597
[16] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh and D. Batra, "Grad-CAM: Visual
explanations from deep networks via gradient-based localization," in Proc. IEEE Int. Conf.
Computer Vision (ICCV), 2017, pp. 618-626. [Online]. Available:
https://arxiv.org/abs/1610.02391

Explainable Tuberculosis Screening from Chest X-Rays with Lesion Localisation

Dept of AI & ML, BMSIT&M 2026-27 12
Appendix A. Work Plan, Responsibilities and Risks
This appendix is supplementary to the prescribed proposal sections. It records the schedule the
team will follow, how the work is divided among the four members, and the main risks with
their mitigations.
A.1 Twelve-Week Plan
Weeks Milestone
Weeks 1-2 Read the WHO screening guideline and the TBX11K papers as a team. Download
TBX11K (Creative Commons Attribution 4.0) and request Shenzhen and Montgomery
from the United States National Library of Medicine repository, since those two are
request-based and can take time. Set up the repository, the environment and the fixed
splits file. Agree the licence question with the guide.

Weeks 3-4 Data pipeline complete: DICOM and PNG loading, anonymisation, the frontal-view
quality gate, CLAHE, and the U-Net lung segmenter trained on Montgomery masks
(experiment E1 passed). Masks generated and spot-checked for all of TBX11K.
Weeks 5-6 Baseline three-class classifier trained on the internal split, with and without lung masking.
First AUC and confusion matrix reported. Class imbalance handled with focal loss or
class weights and the choice justified with numbers (experiment E2 first pass).
Weeks 7-8 Lesion detector trained on the TBX11K boxes; AP50 reported for active TB and latent
TB separately (experiment E4). Grad-CAM integrated on the classifier and heatmaps
reviewed with the guide. Mid-project demo: one image in, class plus boxes plus heatmap
out.

Weeks 9-10 Calibration with temperature scaling, threshold frozen on the validation ROC at
sensitivity of at least 0.90. Cross-dataset evaluation on Shenzhen and Montgomery, plus
the shortcut ablation (experiments E3 and E5). One submission to the TBX11K online
challenge.

Week 11 Streamlit triage dashboard: batch upload, risk-ranked worklist, overlay viewer, PDF
export with the disclaimer. Latency measured on GPU and CPU (experiment E6).
Week 12 Freeze the code, write the report, prepare the demo script and the slide deck, and record a

short backup video of the working demo in case the venue has no GPU.

Buffer (rolling) Half a day each week is kept free for reruns, for Kaggle session limits and for whichever
experiment overruns. If the schedule slips, the detector branch is simplified before the
classifier or the evaluation is cut.

A.2 Team Responsibilities
Member Owns
Member 1 Data pipeline: DICOM and PNG handling, anonymisation, the frontal-view quality

gate, CLAHE, the U-Net lung segmenter and the versioned splits file.
Member 2 Three-class classifier: backbone choice, training loop, class-imbalance handling,

augmentation, and the classification metrics.

Member 3 Lesion detector and explanations: the TBX11K box pipeline, detector training and
AP50 reporting, the permissive-licence fallback detector, and Grad-CAM.
Member 4 Calibration, cross-dataset evaluation and the product: temperature scaling,
threshold selection, Shenzhen and Montgomery testing, the Streamlit dashboard,
the PDF report and the demo.

Explainable Tuberculosis Screening from Chest X-Rays with Lesion Localisation

Dept of AI & ML, BMSIT&M 2026-27 13
A.3 Risks and Mitigations

Risk Mitigation

Class imbalance and the tiny latent TB class.
TBX11K has 1,200 TB images against 10,000
non-TB, and only 212 images labelled latent
TB, 54 labelled both active and latent and 10
uncertain. A detector trained on 212 examples
will not generalise.

Use focal loss or class weights and report balanced
metrics, never plain accuracy. Treat latent TB as a stretch
goal: report its average precision separately and say
openly in the report that the number is not reliable. If it
stays near zero, fall back to a single TB-region class and
keep active versus latent as a secondary label shown only
when confident.

The TBX11K test labels are withheld for the
online challenge, so the team cannot iterate
against the official test set and could
accidentally tune on data it then reports.

Cut a fixed internal validation split out of the 6,600-image
training set on day one, commit the splits file, and use the
official 1,800-image validation set as the internal test set.
Limit submissions to the Codabench challenge to a small
number and record every submission in the report.

Ultralytics YOLO11 is released under AGPL-
3.0. If the college or the guide wants the code

released without copyleft obligations, or if the
tool is ever hosted for others to use, AGPL-3.0
requires the complete corresponding source of
the service to be offered to users.

Keep the detector behind a thin interface so it can be
swapped. Have a permissively licensed fallback ready
from week 1: torchvision's Faster R-CNN or RetinaNet
(BSD-3-Clause) or the original RT-DETR release
(Apache-2.0). Raise the licence question with the guide in
week 1 rather than week 11, and state the chosen licence
in the report.

Shortcut learning and cross-dataset collapse.
The model may score highly on TBX11K and
then fall apart on Shenzhen and Montgomery
because it learned scanner artefacts, text
markers or borders.

Lung masking is in the pipeline for exactly this reason,
and experiment E3 measures whether it helps by
comparing the AUC drop with and without masking.
Grad-CAM maps are reviewed with the guide on real
cases. Any external result is reported as-is, even if it is
bad.
Compute and data access. The 6 GB laptop
GPU may not fit the detector at 512 x 512, free
Kaggle and Colab sessions time out, and
Shenzhen and Montgomery are obtained by
request from the United States National
Library of Medicine rather than by direct
download, so they may arrive late.

Request Shenzhen and Montgomery in week 1. Train with
mixed precision, gradient accumulation and a smaller
batch, and checkpoint every epoch to Google Drive so a
killed Kaggle session loses at most one epoch. If the
external sets do not arrive, substitute another public TB
chest X-ray set and say clearly in the report which
external test was actually used.

Over-reading the explanations. A Grad-CAM
heatmap shows where the model looked, not
proof that the region is causally responsible for
the prediction, and a bounding box drawn for a
benchmark is not a definitive pathology
boundary.

Label every heatmap as attention evidence rather than as a
finding, review a sample of correct and incorrect cases
with the guide, and print the same caveat on the exported
report. Report detection average precision as a
localisation score, not as clinical agreement.

Over-claiming clinical readiness. A student
prototype trained on public datasets cannot
establish clinical safety, and public data may
not represent the range of machines, exposures
and patients seen in Indian district hospitals.

State in the report and on every screen that the tool is
decision support and not a diagnosis, that prospective
clinical validation is outside the scope of this project, and
report external test results exactly as they come out,
including bad ones.