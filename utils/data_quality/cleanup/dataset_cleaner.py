"""Automated dataset cleaning and quality improvement pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class ImageQualityMetrics:
    """Image quality assessment results."""

    is_valid: bool
    image_size: tuple[int, int] | None
    aspect_ratio: float | None
    brightness: float | None
    contrast: float | None
    sharpness_score: float | None
    color_entropy: float | None
    issues: list[str]
    perceptual_hash: str | None


def compute_perceptual_hash(image: Image.Image, hash_size: int = 8) -> str:
    """Compute perceptual hash for image deduplication.

    Uses average hash algorithm - resize and compare to mean brightness.
    """
    img_small = image.resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(img_small.convert("L"))
    avg = pixels.mean()
    bits = (pixels > avg).flatten()
    return "".join(str(int(b)) for b in bits)


def assess_image_quality(image_path: str | Path) -> ImageQualityMetrics:
    """Comprehensive image quality assessment.

    Checks for:
    - Corrupted/unloadable images
    - Extreme aspect ratios
    - Very small/large images
    - Low brightness/contrast
    - Blurriness
    - Low color diversity
    """
    issues = []
    image_path = Path(image_path)

    try:
        image = Image.open(image_path)
        image.load()
    except Exception as e:
        return ImageQualityMetrics(
            is_valid=False,
            image_size=None,
            aspect_ratio=None,
            brightness=None,
            contrast=None,
            sharpness_score=None,
            color_entropy=None,
            issues=[f"Cannot load image: {e}"],
            perceptual_hash=None,
        )

    size = image.size
    w, h = size
    aspect_ratio = w / h if h > 0 else 0

    if size[0] < 256 or size[1] < 256:
        issues.append(f"Image too small: {size}")

    if size[0] > 4096 or size[1] > 4096:
        issues.append(f"Image too large: {size}")

    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        issues.append(f"Extreme aspect ratio: {aspect_ratio:.2f}")

    img_array = np.array(image.convert("RGB"))

    brightness = float(img_array.mean()) / 255.0
    if brightness < 0.05 or brightness > 0.95:
        issues.append(f"Extreme brightness: {brightness:.2f}")

    contrast = float(img_array.std()) / 255.0
    if contrast < 0.02:
        issues.append(f"Very low contrast: {contrast:.2f}")

    try:
        from PIL import ImageFilter

        blurred = image.filter(ImageFilter.GaussianBlur(radius=2))
        blur_diff = float(np.array(image).astype(float).std() - np.array(blurred).astype(float).std())
        sharpness = max(0, blur_diff / 255.0)
        if sharpness < 0.01:
            issues.append(f"Blurry image: sharpness={sharpness:.4f}")
    except Exception:
        sharpness = None

    color_entropy = float(np.std(img_array))
    if color_entropy < 5:
        issues.append(f"Low color diversity: entropy={color_entropy:.2f}")

    try:
        phash = compute_perceptual_hash(image)
    except Exception:
        phash = None

    return ImageQualityMetrics(
        is_valid=len(issues) == 0,
        image_size=size,
        aspect_ratio=aspect_ratio,
        brightness=brightness,
        contrast=contrast,
        sharpness_score=sharpness,
        color_entropy=color_entropy,
        issues=issues,
        perceptual_hash=phash,
    )


def find_duplicate_images(image_paths: list[str | Path], hash_threshold: float = 0.9) -> list[list[str]]:
    """Find perceptually similar images using hash similarity.

    Args:
        image_paths: List of image file paths
        hash_threshold: Similarity threshold (0-1) for considering images duplicates

    Returns:
        List of groups of duplicate images
    """
    hashes = {}
    for img_path in image_paths:
        try:
            image = Image.open(img_path)
            phash = compute_perceptual_hash(image)
            if phash not in hashes:
                hashes[phash] = []
            hashes[phash].append(str(img_path))
        except Exception:
            continue

    duplicates = [group for group in hashes.values() if len(group) > 1]
    return duplicates


def compute_exact_hash(file_path: str | Path) -> str:
    """Compute MD5 hash for exact deduplication."""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


class DatasetCleaner:
    """Orchestrates dataset cleaning and quality improvement."""

    def __init__(self, dataset_dir: str | Path, output_dir: str | Path | None = None):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir) if output_dir else self.dataset_dir / "cleaned"
        self.quality_report = []
        self.duplicates = []

    def scan_and_assess(self) -> dict:
        """Scan all images and assess quality.

        Returns:
            Summary statistics of quality assessment
        """
        image_files = (
            list(self.dataset_dir.glob("*.png"))
            + list(self.dataset_dir.glob("*.jpg"))
            + list(self.dataset_dir.glob("*.jpeg"))
        )

        valid_count = 0
        invalid_count = 0
        issues_by_type = {}

        for img_path in image_files:
            metrics = assess_image_quality(img_path)
            self.quality_report.append((str(img_path), metrics))

            if metrics.is_valid:
                valid_count += 1
            else:
                invalid_count += 1

            for issue in metrics.issues:
                issue_type = issue.split(":")[0]
                issues_by_type[issue_type] = issues_by_type.get(issue_type, 0) + 1

        return {
            "total_images": len(image_files),
            "valid": valid_count,
            "invalid": invalid_count,
            "issues_by_type": issues_by_type,
        }

    def find_and_report_duplicates(self) -> dict:
        """Find duplicate images and report results.

        Returns:
            Statistics on duplicates found
        """
        image_files = (
            list(self.dataset_dir.glob("*.png"))
            + list(self.dataset_dir.glob("*.jpg"))
            + list(self.dataset_dir.glob("*.jpeg"))
        )

        self.duplicates = find_duplicate_images([str(f) for f in image_files])

        total_duplicated = sum(len(group) - 1 for group in self.duplicates)

        return {
            "duplicate_groups": len(self.duplicates),
            "total_duplicated_images": total_duplicated,
            "can_save_space": f"~{total_duplicated * 2}MB if removed",
        }

    def remove_invalid_images(self, dry_run: bool = True) -> list[str]:
        """Remove or move invalid images.

        Args:
            dry_run: If True, only report what would be removed

        Returns:
            List of removed image paths
        """
        removed = []
        invalid_dir = self.output_dir / "invalid" if not dry_run else None

        for img_path, metrics in self.quality_report:
            if not metrics.is_valid:
                if dry_run:
                    removed.append(img_path)
                else:
                    if invalid_dir:
                        invalid_dir.mkdir(parents=True, exist_ok=True)
                        Path(img_path).rename(invalid_dir / Path(img_path).name)
                    else:
                        Path(img_path).unlink()
                    removed.append(img_path)

        return removed

    def remove_duplicates(self, keep_largest: bool = True, dry_run: bool = True) -> list[str]:
        """Remove duplicate images, keeping the best one.

        Args:
            keep_largest: If True, keep the largest image in each duplicate group
            dry_run: If True, only report what would be removed

        Returns:
            List of removed image paths
        """
        removed = []
        dup_dir = self.output_dir / "duplicates" if not dry_run else None

        for group in self.duplicates:
            if keep_largest:
                largest = max(group, key=lambda p: Path(p).stat().st_size)
                to_remove = [p for p in group if p != largest]
            else:
                to_remove = group[1:]

            for img_path in to_remove:
                if dry_run:
                    removed.append(img_path)
                else:
                    if dup_dir:
                        dup_dir.mkdir(parents=True, exist_ok=True)
                        Path(img_path).rename(dup_dir / Path(img_path).name)
                    else:
                        Path(img_path).unlink()
                    removed.append(img_path)

        return removed

    def generate_report(self) -> str:
        """Generate detailed cleaning report."""
        assessment = self.scan_and_assess()
        dup_report = self.find_and_report_duplicates()

        report = f"""
Dataset Cleaning Report
======================

Quality Assessment:
  Total images: {assessment["total_images"]}
  Valid: {assessment["valid"]}
  Invalid: {assessment["invalid"]}

Issues Found:
"""
        for issue_type, count in sorted(assessment["issues_by_type"].items()):
            report += f"  - {issue_type}: {count}\n"

        report += f"""
Duplicate Analysis:
  Duplicate groups: {dup_report["duplicate_groups"]}
  Total duplicated images: {dup_report["total_duplicated_images"]}
  Potential space savings: {dup_report["can_save_space"]}
"""

        return report
