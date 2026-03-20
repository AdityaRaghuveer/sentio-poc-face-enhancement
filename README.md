# Low resolution CCTV Face Enhancement Pipeline
**Project 4: Low-Resolution CCTV Face Enhancement** **Candidate:** Aditya Raghuveer (CS22B019)

This repository contains a specialized Computer Vision pipeline designed to transform low-resolution (12px–80px) CCTV facial crops into sharp, $240 \times 240$ forensic-grade profile images using classical image processing techniques.

##  Project Overview
The objective is to improve the clarity of facial features extracted from blurry, noisy CCTV footage. To ensure forensic integrity and high-speed CPU performance, this project avoids Deep Learning and instead utilizes a strictly ordered **4-Stage Classical Pipeline**.

### 📁 Repository Structure
* `extract_faces.py`: Script to detect and isolate unique faces from source video.
* `solution.py`: The core 4-stage enhancement engine.
* `evaluation_metrics.json`: Automated quality validation (SSIM & Laplacian Sharpness).
* `enhancement_report.html`: Side-by-side A/B comparison of results.
* `raw_faces/`: extracted low-resolution crops.
* `enhanced_faces/`: Final processed $240 \times 240$ outputs.
* `reference_identities/`: given ID photos for recognition testing.

---

##  The 4-Stage Forensic Pipeline

1.  **Stage 1: Denoising** Uses `cv2.fastNlMeansDenoisingColored` ($h=8$) to eliminate sensor noise and compression artifacts without destroying edge information.
    
2.  **Stage 2: Contrast Enhancement (CLAHE)** Processing is performed in the **LAB color space**. Contrast Limited Adaptive Histogram Equalization ($clipLimit=3.5$) is applied to the L-channel to recover facial details hidden in shadows or harsh lighting.

3.  **Stage 3: Multi-step Upscaling** Small faces ($<64px$) undergo a sequence of Lanczos4 interpolation $\rightarrow$ Unsharp Masking $\rightarrow$ final resizing to the target $240 \times 240$ resolution.

4.  **Stage 4: Zone-Specific Sharpening** Leverages `MediaPipe Face Mesh` to create a spatial mask. The eye and nose regions receive aggressive sharpening ($strength=2.0$), while the rest of the face receives lighter sharpening to maintain a natural, non-processed appearance.

---

##  Execution Guide

Follow these steps in order to reproduce the results:

###  Environment Setup and execution
Install the optimized dependencies (compatible with CPU-only environments):
```bash
pip install -r requirements.txt (after this we need to make sure we load Video_1/Class_8_cctv_video_1.mov as it is big file which cant be pushed to github)

python extract_faces.py (creates raw_faces/ which consists of images of persons extracted from cctv footage)

python solutions.py (creates enhanced_faces/ based on 4 stgae piepline code and also creates evaluation_metrics.json , enhancement_report.html )
