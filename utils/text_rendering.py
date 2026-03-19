"""
Advanced text rendering system inspired by Ideogram and Google Imagen.
Handles typography, text placement, and style integration in generated images.
"""
import re
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import difflib


class TextRenderingEngine:
    """Advanced text rendering for AI image generation."""
    
    def __init__(self):
        self.font_cache = {}
        self.text_styles = {
            "modern": {"weight": "bold", "spacing": 1.2, "kerning": 0.1},
            "classic": {"weight": "normal", "spacing": 1.0, "kerning": 0.05},
            "display": {"weight": "black", "spacing": 1.5, "kerning": 0.2},
            "script": {"weight": "normal", "spacing": 0.9, "kerning": -0.05},
            "monospace": {"weight": "normal", "spacing": 1.0, "kerning": 0.0}
        }
        
        self.typography_prompts = {
            "logo": "clean typography, professional logo design, vector style, crisp text",
            "poster": "bold typography, eye-catching text, poster design, readable font",
            "banner": "large text, banner style, promotional typography, clear lettering",
            "sign": "signage typography, outdoor sign, readable from distance, bold letters",
            "book": "book typography, serif font, readable text, literary design",
            "magazine": "magazine layout, editorial typography, modern font, clean design",
            "business": "corporate typography, professional font, business card style",
            "artistic": "artistic typography, creative lettering, stylized text, decorative font"
        }
    
    def extract_text_requirements(self, prompt: str) -> Dict[str, Any]:
        """Extract text rendering requirements from prompt."""
        text_info = {
            "has_text": False,
            "text_content": [],
            "text_style": "modern",
            "text_placement": "center",
            "text_size": "medium",
            "text_color": "auto",
            "typography_type": "general"
        }
        
        # Look for quoted text that should be rendered
        quoted_text = re.findall(r'"([^"]*)"', prompt)
        if quoted_text:
            text_info["has_text"] = True
            text_info["text_content"] = quoted_text
        
        # Look for text in brackets
        bracketed_text = re.findall(r'\[text:\s*([^\]]*)\]', prompt, re.IGNORECASE)
        if bracketed_text:
            text_info["has_text"] = True
            text_info["text_content"].extend(bracketed_text)
        
        # Detect typography type
        for typ, keywords in {
            "logo": ["logo", "brand", "company", "business"],
            "poster": ["poster", "advertisement", "ad", "promotional"],
            "banner": ["banner", "header", "title"],
            "sign": ["sign", "signage", "street", "shop"],
            "book": ["book", "novel", "literature", "cover"],
            "magazine": ["magazine", "editorial", "article"],
            "business": ["business card", "corporate", "professional"],
            "artistic": ["artistic", "creative", "decorative", "stylized"]
        }.items():
            if any(keyword in prompt.lower() for keyword in keywords):
                text_info["typography_type"] = typ
                break
        
        # Detect text style
        style_keywords = {
            "modern": ["modern", "contemporary", "clean", "minimal"],
            "classic": ["classic", "traditional", "serif", "elegant"],
            "display": ["bold", "large", "display", "headline"],
            "script": ["script", "handwritten", "cursive", "calligraphy"],
            "monospace": ["monospace", "code", "technical", "digital"]
        }
        
        for style, keywords in style_keywords.items():
            if any(keyword in prompt.lower() for keyword in keywords):
                text_info["text_style"] = style
                break
        
        # Detect placement
        placement_keywords = {
            "top": ["top", "header", "above"],
            "bottom": ["bottom", "footer", "below"],
            "left": ["left", "side"],
            "right": ["right", "side"],
            "center": ["center", "middle", "centered"]
        }
        
        for placement, keywords in placement_keywords.items():
            if any(keyword in prompt.lower() for keyword in keywords):
                text_info["text_placement"] = placement
                break
        
        # Detect size
        size_keywords = {
            "small": ["small", "tiny", "fine", "subtitle"],
            "medium": ["medium", "normal", "regular"],
            "large": ["large", "big", "headline", "title"],
            "huge": ["huge", "massive", "giant", "billboard"]
        }
        
        for size, keywords in size_keywords.items():
            if any(keyword in prompt.lower() for keyword in keywords):
                text_info["text_size"] = size
                break
        
        return text_info
    
    def enhance_prompt_for_text(self, prompt: str, text_info: Dict[str, Any]) -> str:
        """Enhance prompt with text rendering instructions."""
        if not text_info["has_text"]:
            return prompt
        
        # Add typography-specific prompts
        typography_enhancement = self.typography_prompts.get(
            text_info["typography_type"], 
            "clear text, readable typography, professional font"
        )
        
        # Add style-specific enhancements
        style_enhancement = ""
        if text_info["text_style"] == "modern":
            style_enhancement = "sans-serif font, clean lines, modern typography"
        elif text_info["text_style"] == "classic":
            style_enhancement = "serif font, traditional typography, elegant lettering"
        elif text_info["text_style"] == "display":
            style_enhancement = "bold font, display typography, strong lettering"
        elif text_info["text_style"] == "script":
            style_enhancement = "script font, handwritten style, flowing letters"
        elif text_info["text_style"] == "monospace":
            style_enhancement = "monospace font, fixed-width, technical typography"
        
        # Combine enhancements
        enhanced_prompt = f"{prompt}, {typography_enhancement}, {style_enhancement}"
        
        # Add text quality boosters
        text_quality_tags = [
            "sharp text", "crisp typography", "readable font", "clear lettering",
            "high resolution text", "professional typography", "perfect text rendering"
        ]
        
        enhanced_prompt += f", {', '.join(text_quality_tags[:3])}"
        
        return enhanced_prompt
    
    def create_text_mask(self, image_size: Tuple[int, int], 
                        text_info: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Create a mask for text placement guidance."""
        if not text_info["has_text"] or not text_info["text_content"]:
            return None
        
        width, height = image_size
        mask = torch.zeros((1, height, width), dtype=torch.float32)
        
        # Define text regions based on placement
        placement = text_info["text_placement"]
        
        if placement == "top":
            mask[:, :height//4, :] = 1.0
        elif placement == "bottom":
            mask[:, 3*height//4:, :] = 1.0
        elif placement == "left":
            mask[:, :, :width//3] = 1.0
        elif placement == "right":
            mask[:, :, 2*width//3:] = 1.0
        elif placement == "center":
            mask[:, height//3:2*height//3, width//4:3*width//4] = 1.0
        else:
            # Default center placement
            mask[:, height//3:2*height//3, width//4:3*width//4] = 1.0
        
        return mask
    
    def validate_text_rendering(self, image: Image.Image, 
                               expected_text: List[str]) -> Dict[str, Any]:
        """Validate text rendering quality in generated image."""
        try:
            import pytesseract
            
            # Extract text from image
            extracted_text = pytesseract.image_to_string(image)
            
            validation_results = {
                "text_detected": bool(extracted_text.strip()),
                "extracted_text": extracted_text.strip(),
                "expected_text": expected_text,
                "accuracy_score": 0.0,
                "readability_score": 0.0,
                "issues": []
            }
            
            if expected_text and extracted_text.strip():
                # Calculate accuracy
                expected_combined = " ".join(expected_text).lower()
                extracted_lower = extracted_text.lower()
                
                # Simple word-based accuracy
                expected_words = set(expected_combined.split())
                extracted_words = set(extracted_lower.split())
                
                word_accuracy = 0.0
                if expected_words:
                    word_accuracy = len(expected_words & extracted_words) / len(expected_words)

                # Fuzzy string similarity helps tolerate OCR tokenization differences.
                ratio = difflib.SequenceMatcher(None, expected_combined, extracted_lower).ratio()

                validation_results["accuracy_score"] = max(float(word_accuracy), float(ratio))
                
                # Readability assessment
                readability = self._assess_readability(extracted_text)
                validation_results["readability_score"] = readability
                
                # Identify issues
                if accuracy < 0.5:
                    validation_results["issues"].append("Low text accuracy")
                if readability < 0.6:
                    validation_results["issues"].append("Poor readability")
                if len(extracted_text) < 3:
                    validation_results["issues"].append("Very short text detected")
            
            return validation_results
            
        except ImportError:
            return {
                "text_detected": False,
                "error": "pytesseract not available for text validation"
            }
    
    def _assess_readability(self, text: str) -> float:
        """Assess text readability based on character patterns."""
        if not text.strip():
            return 0.0
        
        # Basic readability metrics
        total_chars = len(text)
        alpha_chars = sum(1 for c in text if c.isalpha())
        space_chars = sum(1 for c in text if c.isspace())
        
        # Readability score based on character distribution
        alpha_ratio = alpha_chars / total_chars if total_chars > 0 else 0
        space_ratio = space_chars / total_chars if total_chars > 0 else 0
        
        # Good text should have high alpha ratio and reasonable spacing
        readability = (alpha_ratio * 0.7) + (min(space_ratio * 5, 0.3) * 0.3)
        
        return min(readability, 1.0)


class TypographyStyler:
    """Typography styling and enhancement system."""
    
    def __init__(self):
        self.font_families = {
            "sans_serif": ["Arial", "Helvetica", "Open Sans", "Roboto", "Lato"],
            "serif": ["Times", "Georgia", "Playfair", "Merriweather", "Crimson"],
            "display": ["Impact", "Bebas", "Oswald", "Montserrat", "Raleway"],
            "script": ["Pacifico", "Dancing Script", "Great Vibes", "Allura"],
            "monospace": ["Courier", "Monaco", "Consolas", "Source Code Pro"]
        }
        
        self.color_schemes = {
            "high_contrast": [("#000000", "#FFFFFF"), ("#FFFFFF", "#000000")],
            "warm": [("#8B4513", "#FFF8DC"), ("#CD853F", "#FFFACD")],
            "cool": [("#191970", "#F0F8FF"), ("#4682B4", "#E6F3FF")],
            "vibrant": [("#FF6347", "#FFFAF0"), ("#32CD32", "#F0FFF0")],
            "elegant": [("#2F4F4F", "#F5F5DC"), ("#696969", "#F8F8FF")]
        }
    
    def suggest_typography_improvements(self, prompt: str, 
                                      text_info: Dict[str, Any]) -> List[str]:
        """Suggest typography improvements for better text rendering."""
        suggestions = []
        
        if text_info["has_text"]:
            # Font suggestions
            typography_type = text_info["typography_type"]
            if typography_type == "logo":
                suggestions.append("Use bold, sans-serif font for logo clarity")
                suggestions.append("Ensure high contrast between text and background")
            elif typography_type == "poster":
                suggestions.append("Use display font for maximum impact")
                suggestions.append("Consider hierarchical text sizing")
            elif typography_type == "book":
                suggestions.append("Use serif font for better readability")
                suggestions.append("Maintain consistent line spacing")
            
            # Style suggestions
            if "gradient" in prompt.lower():
                suggestions.append("Avoid gradients on text for better readability")
            
            if "small" in prompt.lower() and "text" in prompt.lower():
                suggestions.append("Increase text size for better visibility")
            
            # Color suggestions
            if any(color in prompt.lower() for color in ["red", "blue", "green"]):
                suggestions.append("Ensure sufficient contrast with background color")
        
        return suggestions
    
    def generate_text_variations(self, base_prompt: str, 
                                text_content: List[str]) -> List[str]:
        """Generate prompt variations optimized for text rendering."""
        variations = []
        
        for style in ["modern", "classic", "bold", "elegant"]:
            for placement in ["centered", "top", "bottom"]:
                variation = f"{base_prompt}, {style} typography, {placement} text"
                if text_content:
                    variation += f', text says "{text_content[0]}"'
                variation += ", sharp text, readable font, high quality typography"
                variations.append(variation)
        
        return variations[:6]  # Return top 6 variations


class TextAwareInpainting:
    """Text-aware inpainting for editing text in images."""
    
    def __init__(self):
        # pytesseract.image_to_data conf is usually an integer in [0, 100]
        self.text_detection_threshold = 60
    
    def detect_text_regions(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect text regions in an image for targeted editing."""
        try:
            import pytesseract
            from pytesseract import Output
            
            # Get detailed text detection data
            data = pytesseract.image_to_data(image, output_type=Output.DICT)
            
            text_regions = []
            n_boxes = len(data['level'])
            
            for i in range(n_boxes):
                confidence = int(data['conf'][i])
                if confidence >= int(self.text_detection_threshold):  # Good confidence threshold
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    text = data['text'][i].strip()
                    
                    if text:  # Only include regions with actual text
                        text_regions.append({
                            'bbox': (x, y, x + w, y + h),
                            'text': text,
                            'confidence': confidence,
                            'area': w * h
                        })
            
            return text_regions
            
        except ImportError:
            return []
    
    def create_text_edit_mask(self, image: Image.Image, 
                             target_text: str = None) -> Optional[Image.Image]:
        """Create a mask for text editing operations."""
        text_regions = self.detect_text_regions(image)
        
        if not text_regions:
            return None

        # Decide which regions to use (target_text acts as a focus hint).
        if target_text is not None and str(target_text).strip():
            t = str(target_text).strip().lower()
            focused = [r for r in text_regions if t in str(r.get("text", "")).lower()]
            # If we didn't match anything, fall back to all detected regions.
            if focused:
                regions_to_use = focused
            else:
                regions_to_use = text_regions
        else:
            regions_to_use = text_regions

        # "FreeText-like": build line-level masks rather than per-word rectangles.
        # Cluster word boxes by their vertical centers and then take the union bbox per cluster.
        def _bbox_area(b):
            x1, y1, x2, y2 = b
            return max(0, x2 - x1) * max(0, y2 - y1)

        # Compute typical line height for clustering tolerance.
        heights = [max(1, (r["bbox"][3] - r["bbox"][1])) for r in regions_to_use]
        avg_h = float(np.median(heights)) if heights else 12.0
        # Tolerance for considering two words part of the same line.
        y_tol = max(8.0, avg_h * 0.55)

        regions_sorted = sorted(regions_to_use, key=lambda r: (r["bbox"][1] + r["bbox"][3]) * 0.5)
        clusters: List[List[Dict[str, Any]]] = []
        cluster_centers: List[float] = []

        for r in regions_sorted:
            x1, y1, x2, y2 = r["bbox"]
            yc = (y1 + y2) * 0.5
            placed = False
            for ci in range(len(clusters)):
                if abs(yc - cluster_centers[ci]) <= y_tol:
                    clusters[ci].append(r)
                    # Update cluster center (mean)
                    cluster_centers[ci] = float(
                        np.mean([((rr["bbox"][1] + rr["bbox"][3]) * 0.5) for rr in clusters[ci]])
                    )
                    placed = True
                    break
            if not placed:
                clusters.append([r])
                cluster_centers.append(yc)

        # Create mask image: white regions are inpainted; black elsewhere is kept.
        mask = Image.new("L", image.size, 0)  # Black background
        draw = ImageDraw.Draw(mask)

        # Padding proportional to bbox size to better catch broken glyphs / kerning issues.
        pad_x_ratio = 0.08
        pad_y_ratio = 0.10
        min_pad_px = 2

        for cluster in clusters:
            # Optional: ignore tiny clusters (noise).
            union = [10 ** 9, 10 ** 9, -1, -1]
            for r in cluster:
                x1, y1, x2, y2 = r["bbox"]
                union[0] = min(union[0], x1)
                union[1] = min(union[1], y1)
                union[2] = max(union[2], x2)
                union[3] = max(union[3], y2)
            if union[2] <= union[0] or union[3] <= union[1]:
                continue
            ux1, uy1, ux2, uy2 = union
            if _bbox_area((ux1, uy1, ux2, uy2)) < 50:
                continue

            w = ux2 - ux1
            h = uy2 - uy1
            pad_x = max(min_pad_px, int(w * pad_x_ratio))
            pad_y = max(min_pad_px, int(h * pad_y_ratio))
            nx1 = max(0, ux1 - pad_x)
            ny1 = max(0, uy1 - pad_y)
            nx2 = min(image.size[0], ux2 + pad_x)
            ny2 = min(image.size[1], uy2 + pad_y)
            draw.rectangle((nx1, ny1, nx2, ny2), fill=255)

        # Topology refinement: small morphological closing to connect broken glyph strokes.
        try:
            import cv2

            arr = np.array(mask.convert("L"))
            if arr.max() > 0:
                # Close small gaps inside text regions.
                kernel = np.ones((3, 3), dtype=np.uint8)
                arr = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel, iterations=1)
        except Exception:
            pass

        return mask
    
    def suggest_text_edits(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Suggest possible text edits for an image."""
        text_regions = self.detect_text_regions(image)
        
        suggestions = []
        for region in text_regions:
            suggestions.append({
                'original_text': region['text'],
                'bbox': region['bbox'],
                'edit_types': ['replace', 'remove', 'style_change'],
                'confidence': region['confidence']
            })
        
        return suggestions


def create_text_rendering_pipeline():
    """Create a complete text rendering pipeline."""
    return {
        'engine': TextRenderingEngine(),
        'styler': TypographyStyler(),
        'inpainting': TextAwareInpainting()
    }