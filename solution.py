"""
face_enhancement.py
Sentio Mind · Project 4 · Low-Resolution CCTV Face Enhancement

Copy this file to solution.py and fill in every TODO block.
Do not rename any function.
Run: python solution.py
Output goes into enhanced_faces/ (created automatically).
"""

import cv2
import json
import base64
import time
import numpy as np
from pathlib import Path
import mediapipe as mp
from skimage.metrics import structural_similarity as ssim

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
RAW_FACES_DIR    = Path("raw_faces")
ENHANCED_DIR     = Path("enhanced_faces")
REPORT_HTML_OUT  = Path("enhancement_report.html")
METRICS_JSON_OUT = Path("evaluation_metrics.json")

TARGET_SIZE      = (240, 240)
ENHANCED_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# STAGE 1 — DENOISE
# ---------------------------------------------------------------------------

def stage1_denoise(img: np.ndarray) -> np.ndarray:
    """
    cv2.fastNlMeansDenoisingColored with h=8, hColor=8, templateWindowSize=7, searchWindowSize=21
    """
    return cv2.fastNlMeansDenoisingColored(img, None, 8, 8, 7, 21)


# ---------------------------------------------------------------------------
# STAGE 2 — CLAHE
# ---------------------------------------------------------------------------

def stage2_clahe(img: np.ndarray) -> np.ndarray:
    """
    Convert to LAB. Apply CLAHE (clipLimit=3.5, tileGridSize=(4,4)) to L channel. Merge + convert back.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4, 4))
    l_final = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_final, a, b)), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# STAGE 3 — MULTI-STEP UPSCALE
# ---------------------------------------------------------------------------

def unsharp_mask(img: np.ndarray, sigma: float, strength: float) -> np.ndarray:
    """
    blurred = GaussianBlur(img, sigma)
    result  = img + strength * (img - blurred)
    Clip to 0–255.
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    # Using addWeighted for precise linear blending
    result = cv2.addWeighted(img, 1.0 + strength, blurred, -float(strength), 0)
    return np.clip(result, 0, 255).astype(np.uint8)


def stage3_upscale(img: np.ndarray) -> np.ndarray:
    """
    If short side < 64px: 2× LANCZOS4 → unsharp(1.0, 1.6) → 2× LANCZOS4 → resize to TARGET_SIZE.
    Otherwise: direct resize to TARGET_SIZE LANCZOS4.
    """
    h, w = img.shape[:2]
    if min(h, w) < 64:
        # Step 1: First 2x Upscale
        img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
        # Step 2: Intermediate Sharpening
        img = unsharp_mask(img, 1.0, 1.6)
        # Step 3: Final Resize to Target
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
    else:
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
    return img


# ---------------------------------------------------------------------------
# STAGE 4 — ZONE SHARPENING
# ---------------------------------------------------------------------------

def stage4_zone_sharpen(img: np.ndarray) -> np.ndarray:
    """
    MediaPipe Face Mesh → locate eye + nose region → create mask.
    Apply unsharp(0.8, 2.0) to eye+nose zone.
    Apply unsharp(1.2, 1.3) to the rest.
    Blend using the mask.
    Fallback if no face found: unsharp(1.0, 1.5) uniformly.
    """
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        h, w = img.shape[:2]
        
        if results.multi_face_landmarks:
            mask = np.zeros((h, w), dtype=np.float32)
            # Landmarks: Left Eye, Right Eye, Nose Tip, Nose Bridge
            for idx in [33, 133, 362, 263, 1, 4]:
                lm = results.multi_face_landmarks[0].landmark[idx]
                cv2.circle(mask, (int(lm.x * w), int(lm.y * h)), 35, 1.0, -1)
            
            # Smooth the transition mask
            mask = cv2.GaussianBlur(mask, (41, 41), 20)
            mask_3c = cv2.merge([mask, mask, mask])
            
            # Create two sharpened versions
            strong_sharp = unsharp_mask(img, 0.8, 2.0)
            base_sharp   = unsharp_mask(img, 1.2, 1.3)
            
            # Linear blend: (Strong * Mask) + (Base * InverseMask)
            return (strong_sharp * mask_3c + base_sharp * (1.0 - mask_3c)).astype(np.uint8)
            
    return unsharp_mask(img, 1.0, 1.5)


# ---------------------------------------------------------------------------
# FULL PIPELINE — do not change this function
# ---------------------------------------------------------------------------

def enhance_face(img: np.ndarray) -> np.ndarray:
    """Run all 4 stages in order. Do not modify."""
    img = stage1_denoise(img)
    img = stage2_clahe(img)
    img = stage3_upscale(img)
    img = stage4_zone_sharpen(img)
    return img


# ---------------------------------------------------------------------------
# EVALUATION HELPERS
# ---------------------------------------------------------------------------

def sharpness(img: np.ndarray) -> float:
    """Laplacian variance. Higher = sharper. Convert to grayscale first."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    """
    Structural Similarity Index between two images.
    Both resized to TARGET_SIZE before comparison. Convert to grayscale.
    """
    gray_a = cv2.cvtColor(cv2.resize(a, TARGET_SIZE), cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(cv2.resize(b, TARGET_SIZE), cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray_a, gray_b, full=True)
    return float(score)


# ---------------------------------------------------------------------------
# HTML A/B REPORT
# ---------------------------------------------------------------------------

def generate_ab_report(results: list, output_path: Path):
    """
    Self-contained HTML report showing Original vs Enhanced.
    """
    avg_gain = np.mean([r["sharpness_after"] - r["sharpness_before"] for r in results]) if results else 0
    
    html = f"""
    <html>
    <head><style>
        body {{ font-family: 'Segoe UI', sans-serif; background: #1a1a1a; color: #eee; padding: 40px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th {{ background: #333; padding: 15px; border: 1px solid #444; }}
        td {{ padding: 15px; border: 1px solid #444; text-align: center; background: #222; }}
        .header {{ border-bottom: 2px solid #00d4ff; padding-bottom: 10px; }}
        img {{ border: 1px solid #555; border-radius: 4px; }}
    </style></head>
    <body>
        <h1 class="header">Sentio Mind: Forensic Enhancement Report</h1>
        <p><b>Total Processed:</b> {len(results)} | <b>Average Sharpness Increase:</b> +{avg_gain:.2f}</p>
        <table>
            <tr>
                <th>CCTV Source</th>
                <th>Forensic Enhanced Result</th>
                <th>Sharpness Metrics</th>
                <th>SSIM Score</th>
            </tr>
    """
    for r in results:
        html += f"""
            <tr>
                <td><img src="data:image/jpeg;base64,{r['raw_b64']}" width="120"></td>
                <td><img src="data:image/jpeg;base64,{r['enhanced_b64']}" width="240"></td>
                <td>{r['sharpness_before']:.1f} &rarr; <b>{r['sharpness_after']:.1f}</b></td>
                <td>{r['ssim_improvement']:.4f}</td>
            </tr>
        """
    html += "</table></body></html>"
    output_path.write_text(html)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()

    face_paths = sorted(RAW_FACES_DIR.glob("*.jpg")) + sorted(RAW_FACES_DIR.glob("*.png"))
    print(f"Enhancing {len(face_paths)} faces using 4-Stage Pipeline...")

    results = []

    for fp in face_paths:
        raw = cv2.imread(str(fp))
        if raw is None: continue

        # Run Pipeline
        enhanced = enhance_face(raw.copy())
        
        # Save output
        cv2.imwrite(str(ENHANCED_DIR / fp.name), enhanced, [cv2.IMWRITE_JPEG_QUALITY, 98])

        # Calculate Metrics
        s_b = sharpness(raw)
        s_a = sharpness(enhanced)
        ssim_val = ssim_score(raw, enhanced)

        # Encode for report
        _, rb = cv2.imencode(".jpg", cv2.resize(raw, TARGET_SIZE), [cv2.IMWRITE_JPEG_QUALITY, 85])
        _, eb = cv2.imencode(".jpg", enhanced, [cv2.IMWRITE_JPEG_QUALITY, 85])

        results.append({
            "filename": fp.name,
            "sharpness_before": round(s_b, 2),
            "sharpness_after": round(s_a, 2),
            "ssim_improvement": round(ssim_val, 4),
            "raw_b64": base64.b64encode(rb).decode(),
            "enhanced_b64": base64.b64encode(eb).decode(),
        })
        print(f"  [PROCESSED] {fp.name} | Gain: +{s_a - s_b:.1f}")

    # Metrics Export
    metrics = {
        "total_faces": len(results),
        "avg_sharpness_gain": round(float(np.mean([r["sharpness_after"]-r["sharpness_before"] for r in results])), 2) if results else 0,
        "avg_ssim": round(float(np.mean([r["ssim_improvement"] for r in results])), 4) if results else 0,
        "runtime_sec": round(time.time() - t_start, 2)
    }
    with open(METRICS_JSON_OUT, "w") as f:
        json.dump(metrics, f, indent=2)

    generate_ab_report(results, REPORT_HTML_OUT)
    
    print("\n" + "="*50)
    print(f"ENHANCEMENT COMPLETE")
    print(f"Time Taken: {metrics['runtime_sec']}s")
    print(f"Report: {REPORT_HTML_OUT}")
    print("="*50)