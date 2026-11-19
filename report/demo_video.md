# Role 3 backup demo recording script

Status: script prepared; video not recorded. Record after the real detector and pipeline smoke checks pass.
Use development images only. Obtain the team's approval for the examples before sharing a recording.

| Shot | Action | What to explain |
|---|---|---|
| 1 — scope | Show the research-system banner and the model/version information | This is decision support for review, not a diagnosis. Scores from the regional classifier are logits, not calibrated probabilities. |
| 2 — batch | Upload the agreed development batch | Show ranking and quality rejection; a rejected case must carry no model score. |
| 3 — evidence | Select a case with detector boxes | Cyan detector boxes and the classifier's region cells are different outputs. Boxes do not establish TB. Show the detector checkpoint/threshold provenance. |
| 4 — referral | Show a development case where the detector requests additional review | The detector can request review; it cannot downgrade a positive flag. In v2 this is an additional flag, not a new calibrated class. |
| 5 — explanation | Show the approved explanation view | Describe region attribution and Grad-CAM separately. Do not present the P2 95.9% region-cell result as D-FINE accuracy. |
| 6 — report | Export/open a real C14 case PDF | Show the case identifier, output, boxes, thresholds, versions, disclaimer and scope statement. If using the v2 service, explain that its logit output needs its own approved report contract rather than filling C14 probability fields. |
| 7 — log | Show the corresponding structured prediction record | One case record, salted case ID, version information, no source filename or patient identifiers. |
| 8 — performance | Show the actual E6 result table | State machine, number of images, warm-up, median and p95. Do not substitute P2-only latency for the complete detector/explanation pipeline. |
| 9 — limits | Close on the limitations slide | Provisional/definitive split status, exposed cohorts, exploratory localisation, and missing prospective validation. |

Before recording: confirm the overlays and PDF are legible at 1366×768; hide local paths, credentials and
notifications; use no sealed evaluation images. Keep a local backup copy and verify the video opens.

If demonstrating synthetic inputs while training is pending, label the entire recording **synthetic plumbing
demo**. It does not complete the real-model demo acceptance criterion.
