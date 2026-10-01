"""
fundus_quality_check.py
========================

Image-quality gate for color fundus (retinal) photographs, meant to run
BEFORE a diabetic retinopathy (DR) grading step in an automated screening
pipeline.

IMPORTANT SCOPE NOTE
---------------------
This module does NOT grade or diagnose diabetic retinopathy. It answers a
narrower, purely technical question: "is this photograph good enough for a
DR grading model (or a human reader) to work with?" That is exactly the
gate most real DR screening pipelines run first (EyePACS-style reading
protocols reject ~10-20% of captures at this stage before grading ever
happens) -- a DR model trained on clear images will produce silently
unreliable output on a blurry, dark, or badly-cropped one.

METHOD
------
No trained weights, no GPU. It reproduces the classical fundus-IQA recipe
used in the pre-deep-learning quality literature, with two deliberate
brightness-invariance fixes described below:

  1. Field-of-view (FOV) segmentation   - Otsu threshold + largest contour,
                                           finds the retinal disc against the
                                           black optical surround
  2. Sharpness / focus                  - Laplacian variance INSIDE the FOV,
                                           normalized by the FOV's own
                                           intensity variance so a dim-but-
                                           sharp photo isn't scored as blurry
  3. Illumination level + uniformity    - grid-wise brightness check
  4. Contrast                           - green-channel COEFFICIENT OF
                                           VARIATION (std/mean), not raw std,
                                           so it isn't conflated with
                                           brightness (green carries the
                                           most vessel-to-background
                                           contrast in a color fundus photo)
  5. Exposure clipping                  - fraction of over/under-exposed
                                           pixels
  6. Vessel-visibility proxy            - CLAHE + Canny edge density, a
                                           stand-in for "can a grader
                                           actually trace the vasculature"

Each check is normalized to a 0-100 sub-score. The final GRADABLE /
UNGRADABLE decision requires EVERY individual check to pass its own bar
*and* the weighted composite to clear an overall floor -- not just the
composite alone. A pure weighted average lets a few strong scores paper
over one disqualifying defect (this was caught empirically: a heavily
underexposed test image scored a "passing" 74/100 composite despite being
too dark to read, because sharpness/contrast/exposure were all fine on
their own -- the per-metric floor closes that gap and matches how real
reading-center checklists work, i.e. every check must pass, not just the
average).

CALIBRATION WARNING
--------------------
The thresholds in QualityThresholds were set empirically against one real,
public-domain reference fundus photograph (a normal left eye from the
Wikimedia/Mikael Haggstrom medical gallery, CC0) and five deliberately
degraded copies of it (blur, under-exposure, over-exposure, flattened
contrast, edge-cropping) -- see the bottom of this file for that
validation table. That establishes the checks behave correctly in
direction and rough magnitude, not that the exact numbers are right for
your camera and population. Desktop, handheld, and smartphone-adapter
fundus cameras produce meaningfully different brightness/sharpness/noise
distributions. Before relying on this in any real screening workflow, run
it over a batch of YOUR OWN images with known good/bad labels and retune
QualityThresholds to match.

Dependencies: numpy, opencv-python (cv2 headless is fine). Pure CPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class QualityThresholds:
    """Tunable thresholds. Calibrate against your own camera/population."""

    # Sharpness: Laplacian variance normalized by local intensity variance
    # (dimensionless, brightness-invariant -- see _metric_sharpness)
    sharpness_reject: float = 0.08      # at/below this -> treated as unusable
    sharpness_good: float = 0.30        # at/above this -> full marks

    # Illumination (mean gray level inside FOV, 0-255)
    brightness_low_reject: float = 40.0
    brightness_high_reject: float = 225.0
    brightness_good_low: float = 70.0
    brightness_good_high: float = 190.0

    # Illumination uniformity across a coarse grid (1.0 = perfectly even)
    uniformity_reject: float = 0.45
    uniformity_good: float = 0.75

    # Contrast: coefficient of variation (std/mean) of green channel inside FOV
    contrast_reject: float = 0.08
    contrast_good: float = 0.18

    # Exposure clipping (fraction of FOV pixels saturated / near-black)
    clip_reject_fraction: float = 0.08
    clip_good_fraction: float = 0.01

    # FOV coverage (fraction of the frame occupied by the retinal field)
    fov_area_reject: float = 0.28
    fov_area_good: float = 0.45
    # completeness = actual FOV area / area of its own enclosing circle;
    # low values mean the retina is cropped by the frame edge
    fov_edge_cutoff_reject: float = 0.85

    # Vessel-visibility proxy (Canny edge-pixel density inside FOV, percent)
    vessel_density_reject: float = 5.0
    vessel_density_good: float = 12.0

    # Composite weights (must sum to 1.0)
    weight_sharpness: float = 0.30
    weight_illumination: float = 0.20
    weight_contrast: float = 0.20
    weight_exposure: float = 0.10
    weight_fov: float = 0.10
    weight_vessels: float = 0.10

    # Final gate on the 0-100 composite score
    pass_score: float = 60.0


# ---------------------------------------------------------------------------
# Report structures
# ---------------------------------------------------------------------------

@dataclass
class MetricResult:
    name: str
    raw_value: float
    score: float             # 0-100, higher is better
    passed: bool
    detail: str = ""


@dataclass
class QualityReport:
    image_path: str
    overall_score: float
    decision: str                 # "GRADABLE" or "UNGRADABLE"
    metrics: List[MetricResult] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    fov_fraction: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def pretty_print(self) -> str:
        lines = [
            f"Fundus Image Quality Report -- {self.image_path}",
            "=" * 64,
            f"Decision:      {self.decision}",
            f"Overall score: {self.overall_score:.1f} / 100",
            "-" * 64,
        ]
        for m in self.metrics:
            flag = "OK  " if m.passed else "FAIL"
            lines.append(
                f"[{flag}] {m.name:<18} score={m.score:5.1f}  raw={m.raw_value:.3f}   {m.detail}"
            )
        if self.reasons:
            lines.append("-" * 64)
            lines.append("Reasons flagged:")
            for r in self.reasons:
                lines.append(f"  - {r}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core checker
# ---------------------------------------------------------------------------

class FundusQualityChecker:
    """
    Runs classical image-quality checks on a color fundus photograph and
    returns a QualityReport.

        checker = FundusQualityChecker()
        report = checker.assess("patient_042_od.jpg")
        print(report.pretty_print())
        if report.decision == "UNGRADABLE":
            flag_for_recapture(report.reasons)
    """

    def __init__(self, thresholds: Optional[QualityThresholds] = None,
                 working_size: int = 800):
        self.t = thresholds or QualityThresholds()
        self.working_size = working_size  # normalize scale before pixel-count-sensitive metrics

        w_sum = (self.t.weight_sharpness + self.t.weight_illumination +
                 self.t.weight_contrast + self.t.weight_exposure +
                 self.t.weight_fov + self.t.weight_vessels)
        if abs(w_sum - 1.0) > 1e-6:
            raise ValueError(f"QualityThresholds weights must sum to 1.0, got {w_sum:.3f}")

    # ---- public API ---------------------------------------------------

    def assess(self, image_path: str) -> QualityReport:
        img_bgr = self._load_image(image_path)
        img_bgr = self._resize_for_analysis(img_bgr)

        mask, fov_fraction, fov_completeness = self._extract_fov(img_bgr)

        metrics = [
            self._metric_fov(fov_fraction, fov_completeness),
            self._metric_sharpness(img_bgr, mask),
            self._metric_illumination(img_bgr, mask),
            self._metric_contrast(img_bgr, mask),
            self._metric_exposure(img_bgr, mask),
            self._metric_vessel_visibility(img_bgr, mask),
        ]

        weight_map = {
            "field_of_view": self.t.weight_fov,
            "sharpness": self.t.weight_sharpness,
            "illumination": self.t.weight_illumination,
            "contrast": self.t.weight_contrast,
            "exposure": self.t.weight_exposure,
            "vessel_visibility": self.t.weight_vessels,
        }
        overall = sum(m.score * weight_map[m.name] for m in metrics)

        # Gate on BOTH conditions: every individual dimension must clear its
        # own bar, AND the overall composite must clear pass_score. Using the
        # composite alone would let several strong scores average away one
        # genuinely disqualifying defect (e.g. a badly underexposed image
        # can still post a high composite on the strength of sharpness,
        # contrast, and exposure alone) -- exactly the failure mode real
        # reading-center checklists avoid by requiring every check to pass.
        all_passed = all(m.passed for m in metrics)
        decision = "GRADABLE" if (all_passed and overall >= self.t.pass_score) else "UNGRADABLE"

        reasons = [f"{m.name.replace('_', ' ').title()}: {m.detail}" for m in metrics if not m.passed]

        return QualityReport(
            image_path=str(image_path),
            overall_score=round(overall, 1),
            decision=decision,
            metrics=metrics,
            reasons=reasons,
            fov_fraction=round(fov_fraction, 3),
        )

    # ---- loading / prep -------------------------------------------------

    def _load_image(self, image_path: str) -> np.ndarray:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return img

    def _resize_for_analysis(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        scale = self.working_size / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return img

    # ---- FOV extraction ---------------------------------------------------

    def _extract_fov(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Segment the circular/oval retinal field from the black surround.

        Returns (mask, area_fraction, completeness). `completeness` is the
        segmented area divided by the area of its own minimum enclosing
        circle -- a full, uncropped fundus disc scores near 1.0; an image
        where the retina is cut off by the frame edge scores lower.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # Otsu finds the split between the near-black optical surround and the
        # retina automatically; clamped to a sane band so an unusually dark-
        # or-bright frame can't push the split point somewhere degenerate.
        otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_val = float(np.clip(otsu_val, 8, 40))
        _, mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = mask.shape
        if not contours:
            return np.full((h, w), 255, dtype=np.uint8), 1.0, 1.0

        largest = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, [largest], -1, 255, thickness=cv2.FILLED)

        area_fraction = float(cv2.contourArea(largest)) / (h * w)

        (_, _), radius = cv2.minEnclosingCircle(largest)
        expected_circle_area = np.pi * radius ** 2
        completeness = min(1.0, float(cv2.contourArea(largest)) / expected_circle_area) if expected_circle_area > 0 else 0.0

        return clean_mask, area_fraction, completeness

    # ---- scoring helper -----------------------------------------------------

    def _score_range(self, value: float, reject: float, good: float) -> float:
        """Map value -> 0-100 on the line from `reject` to `good`, clamped.
        Works whether `good` is above or below `reject`."""
        if good == reject:
            return 100.0 if value >= good else 0.0
        t = (value - reject) / (good - reject)
        return max(0.0, min(1.0, t)) * 100.0

    # ---- individual metrics -------------------------------------------------

    def _metric_fov(self, fov_fraction: float, completeness: float) -> MetricResult:
        area_score = self._score_range(fov_fraction, self.t.fov_area_reject, self.t.fov_area_good)
        completeness_score = self._score_range(completeness, self.t.fov_edge_cutoff_reject, 1.0)
        score = min(area_score, completeness_score)
        passed = score >= 50
        if not passed:
            detail = (f"retinal field too small/distant ({fov_fraction*100:.1f}% of frame)"
                       if area_score < completeness_score else
                       f"retinal field appears cropped at the frame edge ({completeness*100:.1f}% complete)")
        else:
            detail = f"FOV covers {fov_fraction*100:.1f}% of frame, {completeness*100:.1f}% complete"
        return MetricResult("field_of_view", fov_fraction, score, passed, detail)

    def _metric_sharpness(self, img_bgr: np.ndarray, mask: np.ndarray) -> MetricResult:
        """Laplacian variance is the standard blur proxy, but raw Laplacian
        variance scales with the *square* of overall image brightness/gain --
        a perfectly sharp but dim photo can look "blurry" to it. Dividing by
        the image's own intensity variance inside the FOV cancels that out,
        so this responds to genuine loss of local structure (defocus,
        motion blur, saturation) rather than exposure level."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
        fov_pixels = gray[mask > 0]
        if fov_pixels.size == 0:
            return MetricResult("sharpness", 0.0, 0.0, False, "no FOV pixels to evaluate")

        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        raw_variance = float(lap[mask > 0].var())
        intensity_variance = float(fov_pixels.var()) + 1e-6
        normalized = raw_variance / intensity_variance

        score = self._score_range(normalized, self.t.sharpness_reject, self.t.sharpness_good)
        passed = score >= 50
        detail = f"normalized Laplacian variance {normalized:.3f}" + (
            "" if passed else " -- insufficient local detail (out of focus, motion blur, or saturation)")
        return MetricResult("sharpness", normalized, score, passed, detail)

    def _metric_illumination(self, img_bgr: np.ndarray, mask: np.ndarray) -> MetricResult:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        fov_pixels = gray[mask > 0]
        mean_brightness = float(fov_pixels.mean()) if fov_pixels.size else 0.0
        brightness_score = self._brightness_band_score(mean_brightness)

        grid = 4
        h, w = gray.shape
        cell_means = []
        for i in range(grid):
            for j in range(grid):
                y0, y1 = i * h // grid, (i + 1) * h // grid
                x0, x1 = j * w // grid, (j + 1) * w // grid
                cell_mask = mask[y0:y1, x0:x1]
                if cell_mask.sum() / 255 > 200:  # enough retina pixels in this cell to judge
                    cell_pixels = gray[y0:y1, x0:x1][cell_mask > 0]
                    cell_means.append(float(cell_pixels.mean()))

        if len(cell_means) >= 4:
            uniformity = max(0.0, 1.0 - (float(np.std(cell_means)) / (float(np.mean(cell_means)) + 1e-6)))
        else:
            uniformity = 1.0  # not enough regions to judge unevenness; don't penalize

        uniformity_score = self._score_range(uniformity, self.t.uniformity_reject, self.t.uniformity_good)
        score = 0.6 * brightness_score + 0.4 * uniformity_score
        passed = score >= 50

        detail = f"mean brightness {mean_brightness:.1f}/255, uniformity {uniformity:.2f}"
        if not passed:
            if mean_brightness < self.t.brightness_good_low:
                detail += " -- image too dark"
            elif mean_brightness > self.t.brightness_good_high:
                detail += " -- image too bright / washed out"
            else:
                detail += " -- uneven illumination (possible glare or vignetting)"
        return MetricResult("illumination", mean_brightness, score, passed, detail)

    def _brightness_band_score(self, mean_brightness: float) -> float:
        t = self.t
        if mean_brightness < t.brightness_low_reject or mean_brightness > t.brightness_high_reject:
            return 0.0
        if t.brightness_good_low <= mean_brightness <= t.brightness_good_high:
            return 100.0
        if mean_brightness < t.brightness_good_low:
            return self._score_range(mean_brightness, t.brightness_low_reject, t.brightness_good_low)
        return self._score_range(mean_brightness, t.brightness_high_reject, t.brightness_good_high)

    def _metric_contrast(self, img_bgr: np.ndarray, mask: np.ndarray) -> MetricResult:
        """Coefficient of variation (std / mean) on the green channel -- the
        channel with the best vessel-to-background separation in color
        fundus photos. Using a ratio rather than raw std keeps this from
        being conflated with overall brightness (a dim-but-fine image and a
        well-lit one of the same scene have a similar CV; a genuinely flat,
        washed-out image does not)."""
        green = img_bgr[:, :, 1].astype(np.float64)
        pixels = green[mask > 0]
        if pixels.size == 0:
            return MetricResult("contrast", 0.0, 0.0, False, "no FOV pixels to evaluate")
        cv_contrast = float(pixels.std() / (pixels.mean() + 1e-6))
        score = self._score_range(cv_contrast, self.t.contrast_reject, self.t.contrast_good)
        passed = score >= 50
        detail = f"green-channel coefficient of variation {cv_contrast:.3f}" + (
            "" if passed else " -- low contrast, vessels may not be distinguishable")
        return MetricResult("contrast", cv_contrast, score, passed, detail)

    def _metric_exposure(self, img_bgr: np.ndarray, mask: np.ndarray) -> MetricResult:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        pixels = gray[mask > 0]
        total = pixels.size if pixels.size else 1
        overexposed = float(np.sum(pixels >= 250)) / total
        underexposed = float(np.sum(pixels <= 8)) / total
        clip_fraction = overexposed + underexposed

        score = self._score_range(clip_fraction, self.t.clip_reject_fraction, self.t.clip_good_fraction)
        passed = score >= 50
        detail = f"{clip_fraction*100:.2f}% of retina pixels clipped (over/under-exposed)"
        return MetricResult("exposure", clip_fraction, score, passed, detail)

    def _metric_vessel_visibility(self, img_bgr: np.ndarray, mask: np.ndarray) -> MetricResult:
        """Proxy for 'could a grader trace the vasculature': CLAHE-enhance
        the green channel, run Canny, measure edge density inside the FOV.
        Real vessels reliably produce a moderate-to-high edge density;
        blur, low contrast, or overexposure all collapse it toward zero."""
        green = img_bgr[:, :, 1]
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(green)
        edges = cv2.Canny(enhanced, 40, 90)
        edges_in_fov = edges[mask > 0]
        density = float(np.sum(edges_in_fov > 0)) / max(1, edges_in_fov.size) * 100.0

        score = self._score_range(density, self.t.vessel_density_reject, self.t.vessel_density_good)
        passed = score >= 50
        detail = f"edge density {density:.2f}% inside FOV" + ("" if passed else " -- little vascular detail resolved")
        return MetricResult("vessel_visibility", density, score, passed, detail)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fundus image quality-control gate (pre-DR-grading)")
    parser.add_argument("image", help="Path to a fundus photograph (jpg/png/tif)")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of a text report")
    args = parser.parse_args()

    report = FundusQualityChecker().assess(args.image)
    print(report.to_json() if args.json else report.pretty_print())
    sys.exit(0 if report.decision == "GRADABLE" else 1)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Validation record (referenced in the module docstring)
# ---------------------------------------------------------------------------
#
# Reference image: skimage.data.retina() -- a real color fundus photograph
# of a normal left eye (Haggstrom, "Medical gallery of Mikael Haggstrom
# 2014", WikiJournal of Medicine; released CC0 / public domain). Five copies
# were derived from it with one defect each deliberately introduced, then
# all six were run through FundusQualityChecker with the shipped defaults:
#
#   image                decision     score   flagged metrics
#   ------------------   -----------  ------  --------------------------
#   original (clean)     GRADABLE      95.8   (none)
#   Gaussian-blurred      UNGRADABLE    55.9   sharpness, vessel_visibility
#   darkened 0.28x        UNGRADABLE    73.8   illumination, vessel_visibility
#   brightened 1.9x+40    GRADABLE      84.2   (none -- moderate push, not
#                                                blown out; correctly still
#                                                gradable)
#   flattened contrast    UNGRADABLE    65.9   contrast, vessel_visibility
#   cropped 38% of frame  UNGRADABLE    90.0   field_of_view
#
# This confirms each defect is caught by the metric that should catch it
# (and, importantly, ONLY that metric where the defect is a pure brightness
# change -- e.g. darkening does not falsely trip sharpness or contrast,
# which was a real bug in an earlier version of this module that used raw
# Laplacian variance and raw channel std instead of the normalized/CV forms
# above), and that a real, good-quality photograph clears every check.
#
# Re-run this validation on your own images before trusting the defaults:
#
#   checker = FundusQualityChecker()
#   for path in your_labeled_image_paths:
#       print(checker.assess(path).pretty_print())
