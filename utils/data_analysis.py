"""
Data quality analysis and dataset statistics utilities.
"""
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
from PIL import Image


class DatasetAnalyzer:
    """Analyze dataset quality and statistics."""
    
    def __init__(self, data_path: str = None, manifest_path: str = None):
        self.data_path = Path(data_path) if data_path else None
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.stats = {}
    
    def analyze_images(self, max_samples: int = 1000) -> Dict[str, Any]:
        """Analyze image properties (resolution, aspect ratio, format)."""
        image_stats = {
            "total_images": 0,
            "resolutions": Counter(),
            "aspect_ratios": Counter(),
            "formats": Counter(),
            "sizes_mb": [],
            "corrupted_images": []
        }
        
        image_paths = self._get_image_paths()
        
        for i, img_path in enumerate(image_paths):
            if i >= max_samples:
                break
                
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    format_name = img.format or "Unknown"
                    
                    # Calculate aspect ratio (rounded to 2 decimals)
                    aspect_ratio = round(width / height, 2)
                    
                    # File size
                    size_mb = img_path.stat().st_size / (1024 * 1024)
                    
                    image_stats["resolutions"][f"{width}x{height}"] += 1
                    image_stats["aspect_ratios"][aspect_ratio] += 1
                    image_stats["formats"][format_name] += 1
                    image_stats["sizes_mb"].append(size_mb)
                    image_stats["total_images"] += 1
                    
            except Exception:
                image_stats["corrupted_images"].append(str(img_path))
        
        # Calculate statistics
        if image_stats["sizes_mb"]:
            image_stats["avg_size_mb"] = np.mean(image_stats["sizes_mb"])
            image_stats["median_size_mb"] = np.median(image_stats["sizes_mb"])
            image_stats["total_size_gb"] = sum(image_stats["sizes_mb"]) / 1024
        
        return image_stats
    
    def analyze_captions(self, max_samples: int = 1000) -> Dict[str, Any]:
        """Analyze caption quality and statistics."""
        caption_stats = {
            "total_captions": 0,
            "lengths": [],
            "word_counts": [],
            "tag_counts": [],
            "common_words": Counter(),
            "common_tags": Counter(),
            "empty_captions": 0,
            "has_emphasis": 0,
            "has_quality_tags": 0,
            "languages": Counter()
        }
        
        captions = self._get_captions()
        
        quality_tags = ["masterpiece", "best quality", "high quality", "highres", "8k", "ultra detailed"]
        
        for i, caption in enumerate(captions):
            if i >= max_samples:
                break
            
            if not caption or caption.strip() == "":
                caption_stats["empty_captions"] += 1
                continue
            
            caption = caption.strip()
            caption_stats["total_captions"] += 1
            caption_stats["lengths"].append(len(caption))
            
            # Word analysis
            words = caption.lower().split()
            caption_stats["word_counts"].append(len(words))
            caption_stats["common_words"].update(words)
            
            # Tag analysis (comma-separated)
            if "," in caption:
                tags = [tag.strip() for tag in caption.split(",")]
                caption_stats["tag_counts"].append(len(tags))
                caption_stats["common_tags"].update([tag.lower() for tag in tags])
            
            # Emphasis detection
            if "(" in caption or "[" in caption:
                caption_stats["has_emphasis"] += 1
            
            # Quality tags detection
            if any(tag in caption.lower() for tag in quality_tags):
                caption_stats["has_quality_tags"] += 1
            
            # Simple language detection (very basic)
            if any(ord(char) > 127 for char in caption):
                caption_stats["languages"]["non_ascii"] += 1
            else:
                caption_stats["languages"]["ascii"] += 1
        
        # Calculate statistics
        if caption_stats["lengths"]:
            caption_stats["avg_length"] = np.mean(caption_stats["lengths"])
            caption_stats["median_length"] = np.median(caption_stats["lengths"])
            caption_stats["avg_word_count"] = np.mean(caption_stats["word_counts"])
            caption_stats["median_word_count"] = np.median(caption_stats["word_counts"])
        
        if caption_stats["tag_counts"]:
            caption_stats["avg_tag_count"] = np.mean(caption_stats["tag_counts"])
            caption_stats["median_tag_count"] = np.median(caption_stats["tag_counts"])
        
        return caption_stats
    
    def check_data_quality(self) -> Dict[str, Any]:
        """Check overall data quality and identify issues."""
        issues = []
        warnings = []
        
        # Check if data exists
        if self.data_path and not self.data_path.exists():
            issues.append(f"Data path does not exist: {self.data_path}")
        
        if self.manifest_path and not self.manifest_path.exists():
            issues.append(f"Manifest path does not exist: {self.manifest_path}")
        
        # Analyze images and captions
        try:
            image_stats = self.analyze_images(max_samples=500)
            caption_stats = self.analyze_captions(max_samples=500)
            
            # Check for issues
            if image_stats["corrupted_images"]:
                issues.append(f"Found {len(image_stats['corrupted_images'])} corrupted images")
            
            if caption_stats["empty_captions"] > 0:
                warnings.append(f"Found {caption_stats['empty_captions']} empty captions")
            
            if caption_stats["avg_length"] < 20:
                warnings.append("Average caption length is very short (< 20 characters)")
            
            if caption_stats["has_quality_tags"] / caption_stats["total_captions"] < 0.1:
                warnings.append("Very few captions contain quality tags")
            
            # Check resolution diversity
            if len(image_stats["resolutions"]) == 1:
                warnings.append("All images have the same resolution - consider adding diversity")
            
            # Check aspect ratio diversity
            square_ratio = image_stats["aspect_ratios"].get(1.0, 0)
            if square_ratio / image_stats["total_images"] > 0.9:
                warnings.append("Most images are square - consider adding different aspect ratios")
            
        except Exception as e:
            issues.append(f"Error analyzing data: {str(e)}")
            image_stats = {}
            caption_stats = {}
        
        return {
            "issues": issues,
            "warnings": warnings,
            "image_stats": image_stats,
            "caption_stats": caption_stats
        }
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """Generate a comprehensive data quality report."""
        quality_check = self.check_data_quality()
        
        report = []
        report.append("=" * 80)
        report.append("DATASET QUALITY REPORT")
        report.append("=" * 80)
        
        # Issues and warnings
        if quality_check["issues"]:
            report.append("\n🚨 CRITICAL ISSUES:")
            for issue in quality_check["issues"]:
                report.append(f"  - {issue}")
        
        if quality_check["warnings"]:
            report.append("\n⚠️  WARNINGS:")
            for warning in quality_check["warnings"]:
                report.append(f"  - {warning}")
        
        # Image statistics
        img_stats = quality_check.get("image_stats", {})
        if img_stats:
            report.append("\n📸 IMAGE STATISTICS:")
            report.append(f"  Total Images: {img_stats['total_images']:,}")
            report.append(f"  Average Size: {img_stats.get('avg_size_mb', 0):.2f} MB")
            report.append(f"  Total Dataset Size: {img_stats.get('total_size_gb', 0):.2f} GB")
            
            report.append("\n  Top Resolutions:")
            for res, count in img_stats["resolutions"].most_common(5):
                percentage = (count / img_stats["total_images"]) * 100
                report.append(f"    {res}: {count:,} ({percentage:.1f}%)")
            
            report.append("\n  Top Aspect Ratios:")
            for ratio, count in img_stats["aspect_ratios"].most_common(5):
                percentage = (count / img_stats["total_images"]) * 100
                report.append(f"    {ratio}: {count:,} ({percentage:.1f}%)")
        
        # Caption statistics
        cap_stats = quality_check.get("caption_stats", {})
        if cap_stats:
            report.append("\n📝 CAPTION STATISTICS:")
            report.append(f"  Total Captions: {cap_stats['total_captions']:,}")
            report.append(f"  Average Length: {cap_stats.get('avg_length', 0):.1f} characters")
            report.append(f"  Average Words: {cap_stats.get('avg_word_count', 0):.1f}")
            report.append(f"  Average Tags: {cap_stats.get('avg_tag_count', 0):.1f}")
            report.append(f"  Has Emphasis: {cap_stats['has_emphasis']:,} ({cap_stats['has_emphasis']/cap_stats['total_captions']*100:.1f}%)")
            report.append(f"  Has Quality Tags: {cap_stats['has_quality_tags']:,} ({cap_stats['has_quality_tags']/cap_stats['total_captions']*100:.1f}%)")
            
            report.append("\n  Most Common Tags:")
            for tag, count in cap_stats["common_tags"].most_common(10):
                percentage = (count / cap_stats["total_captions"]) * 100
                report.append(f"    {tag}: {count:,} ({percentage:.1f}%)")
        
        report.append("\n" + "=" * 80)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {save_path}")
        
        return report_text
    
    def _get_image_paths(self) -> List[Path]:
        """Get list of image paths from data directory or manifest."""
        if self.manifest_path:
            # Load from JSONL manifest
            image_paths = []
            with open(self.manifest_path, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    img_path = data.get("image_path") or data.get("path")
                    if img_path:
                        image_paths.append(Path(img_path))
            return image_paths
        
        elif self.data_path:
            # Scan directory for images
            extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            image_paths = []
            
            for ext in extensions:
                image_paths.extend(self.data_path.rglob(f"*{ext}"))
                image_paths.extend(self.data_path.rglob(f"*{ext.upper()}"))
            
            return image_paths
        
        return []
    
    def _get_captions(self) -> List[str]:
        """Get list of captions from data directory or manifest."""
        if self.manifest_path:
            # Load from JSONL manifest
            captions = []
            with open(self.manifest_path, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    caption = data.get("caption") or data.get("text", "")
                    captions.append(caption)
            return captions
        
        elif self.data_path:
            # Load from .txt/.caption files
            captions = []
            image_paths = self._get_image_paths()
            
            for img_path in image_paths:
                # Try different caption file extensions
                for ext in ['.txt', '.caption']:
                    caption_path = img_path.with_suffix(ext)
                    if caption_path.exists():
                        try:
                            with open(caption_path, 'r', encoding='utf-8') as f:
                                caption = f.read().strip()
                                captions.append(caption)
                            break
                        except Exception:
                            captions.append("")
                    else:
                        captions.append("")
            
            return captions
        
        return []


def analyze_caption_emphasis(caption: str) -> Dict[str, Any]:
    """Analyze emphasis patterns in a single caption."""
    emphasis_patterns = {
        "parentheses_single": r'\(([^)]+)\)',
        "parentheses_double": r'\(\(([^)]+)\)\)',
        "brackets_single": r'\[([^\]]+)\]',
        "brackets_double": r'\[\[([^\]]+)\]\]',
    }
    
    results = {
        "has_emphasis": False,
        "emphasis_count": 0,
        "emphasis_types": [],
        "emphasized_terms": []
    }
    
    for pattern_name, pattern in emphasis_patterns.items():
        matches = re.findall(pattern, caption)
        if matches:
            results["has_emphasis"] = True
            results["emphasis_count"] += len(matches)
            results["emphasis_types"].append(pattern_name)
            results["emphasized_terms"].extend(matches)
    
    return results


def suggest_caption_improvements(caption: str) -> List[str]:
    """Suggest improvements for a caption."""
    suggestions = []
    
    if len(caption) < 20:
        suggestions.append("Caption is very short - consider adding more descriptive details")
    
    if not any(tag in caption.lower() for tag in ["masterpiece", "best quality", "high quality"]):
        suggestions.append("Consider adding quality tags like 'masterpiece, best quality'")
    
    if "," not in caption:
        suggestions.append("Consider using comma-separated tags for better structure")
    
    if not re.search(r'[()[\]]', caption):
        suggestions.append("Consider using emphasis syntax: (important tag) or [less important]")
    
    # Check for common issues
    if caption.count("(") != caption.count(")"):
        suggestions.append("Unmatched parentheses - check emphasis syntax")
    
    if caption.count("[") != caption.count("]"):
        suggestions.append("Unmatched brackets - check emphasis syntax")
    
    return suggestions