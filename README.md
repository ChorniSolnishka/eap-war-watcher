# EaP WarWatcher — MVP

> **Status:** MVP of my personal pet‑project.  
> **Game:** _Empires & Puzzles_ (community tool, unaffiliated).  
> **Concept:** A player uploads war screenshots → the backend analyzes them (OpenCV‑first pipeline with a lightweight OCR tail) → returns a polished **Excel** report of the war.

This repository demonstrates the approach and the live MVP slice. It is intentionally minimal in UI/ops and **heavy on computer‑vision primitives** to make the pipeline robust on noisy, real‑world screenshots.

> ⚠️ **Important:** the most valuable assets (trained weights, finely tuned thresholds/coefficients, and a few helper datasets) are **intentionally NOT included**. I plan to **monetize** the service later. **The code here is not licensed for public use** (see _License & Usage_).

![Видео без названия — сделано в Clipchamp (17)](https://github.com/user-attachments/assets/ddbf6c6d-a7f1-4d4d-9708-83a8bbffb8aa)
<img width="1919" height="1073" alt="image" src="https://github.com/user-attachments/assets/c9fe5ef3-32f8-4b99-8c02-d4abadedfe05" />
<img width="1919" height="1132" alt="image" src="https://github.com/user-attachments/assets/850abe6d-1e20-4d40-99e7-9958781a5c12" />



## What it does (short)

- Creates a **War** by naming **my alliance** and the **enemy alliance**.  
- Accepts **one or many** screenshots from the _Empires & Puzzles_ war screen.  
- Runs an **OpenCV‑first “imatch” pipeline** to segment rows, extract numeric regions, and track players across screenshots.  
- Uses OCR **only for digits**, after aggressive image normalization.  
- Generates a **formatted Excel report** and returns it via HTTP.


## Why **OpenCV‑first** (and not OCR‑first)

OCR alone is brittle on game UIs: compression artifacts, subtle gradients, drop‑shadows, and scaling vary a lot between devices and messengers. The MVP therefore treats OCR as the **final narrow step** and relies on strong **image matching** and **geometric normalization** earlier in the pipeline to deliver stable crops and to validate numbers over time.


## The `imatch` module (core CV/matching stack)

`imatch` is the set of OpenCV‑centric building blocks and heuristics that make the pipeline stable. It combines fast pre‑filters with more precise (but more expensive) matchers and small geometry refinements. At a high level:

### 1) Pre‑processing & normalization
- **Color to Gray** and **dynamic range normalization** (min–max + optional local contrast adjustments).  
- **Denoising** and **mild sharpening** to stabilize edges without overshooting.  
- **Adaptive binarization** for numeric ROIs (when helpful), keeping a grayscale copy for correlation.  
- **Intra‑app resizing policy** so the same designs align in a common pixel space.

### 2) Row segmentation (screen → player rows)
- **Vertical & horizontal projection profiles** over **Sobel magnitude** to locate horizontal separators and row bands.  
- **Contour filtering** with geometric sanity checks (aspect ratios, minimum widths/heights, padding).  
- **Anchor‑based slicing**: once a template anchor is found (e.g., the score column gutter), rows are sliced with fixed offsets/tolerances.

### 3) Fast similarity filters (cheap checks)
- **Perceptual hash**: custom **dHash64** on stabilized crops. Distances below a small Hamming threshold are treated as “same or near‑same”.  
- **Global histogram checks** (intensity/edge histograms) to early‑reject obvious non‑matches.

### 4) Precise matching (expensive, used after pre‑filters)
- **Normalized Cross‑Correlation (NCC)** on luminance‑normalized crops.  
- **Edge‑domain NCC** on **Sobel magnitude** (robust to small color shifts).  
- Optional **ECC‑based micro‑alignment** (`cv2.findTransformECC`) for sub‑pixel translation/scaling before scoring.  
- Final score is a **blend**: `score = w1*NCC_gray + w2*NCC_edges − w3*penalties`, where penalties encode crop drift, aspect deviation, and low edge energy.

### 5) ROI refinement & NMS
- **Iterative ROI nudging**: small ±Δx/±Δy/±Δs grid to maximize the score under tight bounds.  
- **Non‑Maximum Suppression** on overlapping candidates so each row/number yields a single accepted ROI.  
- **Temporal smoothing** across screenshots for the same player row (stability over time is rewarded).

### 6) Number field extraction (pre‑OCR)
- Per‑column **layout priors** (expected x offsets/widths in the row).  
- **Local contrast stretch** and **binarization** tuned for digits.  
- **Shape sanity checks** (width/height ranges, hole counts for 0/8, etc.) to reject nonsense before OCR.

### 7) Validation across time
- **Monotonicity/consistency rules** for scores across sequential screenshots (e.g., score should not decrease).  
- **Delta limits** to flag improbable jumps.  
- Confidence aggregation from matching + OCR to decide keep/retry/fallback.

> The exact **weights**, **thresholds**, and **blend coefficients** that make this work well are **private** in this MVP. They were tuned on an internal corpus and are excluded from the repo.


## OCR (kept intentionally lightweight)

- **Digits‑only** reading via **Tesseract** (whitelist).  
- Optional experiment: a tiny CNN (0–99) exported to **ONNX** for speed.  
- OCR is invoked **after** imatch delivers a clean, normalized crop; its confidence is fused with imatch scores and temporal checks.


## Report generation (Excel)

- Built with **xlsxwriter**: formatting, color coding, optional thumbnails.  
- The generated `.xlsx` is streamed by the API and also saved alongside artifacts for convenience.


## Roadmap

Planned evolutions of this MVP:
- **Accounts & authentication**, team/organization workspaces.  
- **Global statistics across wars** (aggregations, leaderboards, time‑series).  
- **Monetization** (tiered accuracy/volume, premium exports & dashboards).  
- Deeper **modeling** for icon/text separation and robust low‑res handling.  
- Cloud deployment templates for easy spin‑up.

If you’re interested in early access or collaboration, please open an issue.


## License & Usage

This repository is shared as a **portfolio/MVP showcase**. Unless you have a **written agreement** from me:

- **All rights reserved.**  
- **Commercial use, redistribution, and derivative works are not permitted.**  
- Internal evaluation is fine, but **do not** deploy it publicly.

> Critical assets (trained weights, tuned coefficients, helper datasets) are **intentionally omitted** to preserve commercial potential.


## Acknowledgements

- _Empires & Puzzles_ is a trademark of its respective owners. This project is unaffiliated and provided for community/educational purposes.  
- Thanks to the authors of **OpenCV**, **FastAPI**, **SQLAlchemy**, **Tesseract**, **xlsxwriter**, and the broader open‑source community.
