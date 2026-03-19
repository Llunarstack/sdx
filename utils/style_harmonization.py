"""
Style Harmonization System - Handles mixed styles to prevent weird-looking images.
Manages conflicts between 2D/3D styles, multiple LoRAs, and conflicting prompts.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from dataclasses import dataclass
from enum import Enum
import re
import json


class StyleType(Enum):
    """Style type classifications."""
    REALISTIC_3D = "realistic_3d"
    ANIME_2D = "anime_2d"
    CARTOON_2D = "cartoon_2d"
    ARTISTIC_2D = "artistic_2d"
    PHOTOGRAPHIC = "photographic"
    PAINTERLY = "painterly"
    SKETCH = "sketch"
    MIXED = "mixed"


class StyleConflictLevel(Enum):
    """Levels of style conflict severity."""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    INCOMPATIBLE = "incompatible"


@dataclass
class StyleComponent:
    """Individual style component (LoRA, prompt element, etc.)."""
    name: str
    type: StyleType
    strength: float
    source: str  # "lora", "prompt", "embedding", etc.
    priority: int = 1
    compatibility_tags: List[str] = None
    
    def __post_init__(self):
        if self.compatibility_tags is None:
            self.compatibility_tags = []


@dataclass
class StyleAnalysis:
    """Analysis of style components and conflicts."""
    components: List[StyleComponent]
    dominant_style: StyleType
    conflict_level: StyleConflictLevel
    conflicts: List[Dict[str, Any]]
    harmonization_needed: bool
    recommended_adjustments: List[Dict[str, Any]]


class StyleCompatibilityMatrix:
    """Defines compatibility between different style types."""
    
    def __init__(self):
        # Compatibility scores (0.0 = incompatible, 1.0 = perfect match)
        self.compatibility = {
            (StyleType.REALISTIC_3D, StyleType.REALISTIC_3D): 1.0,
            (StyleType.REALISTIC_3D, StyleType.PHOTOGRAPHIC): 0.9,
            (StyleType.REALISTIC_3D, StyleType.ANIME_2D): 0.2,
            (StyleType.REALISTIC_3D, StyleType.CARTOON_2D): 0.1,
            (StyleType.REALISTIC_3D, StyleType.ARTISTIC_2D): 0.4,
            (StyleType.REALISTIC_3D, StyleType.PAINTERLY): 0.6,
            (StyleType.REALISTIC_3D, StyleType.SKETCH): 0.3,
            
            (StyleType.ANIME_2D, StyleType.ANIME_2D): 1.0,
            (StyleType.ANIME_2D, StyleType.CARTOON_2D): 0.7,
            (StyleType.ANIME_2D, StyleType.ARTISTIC_2D): 0.8,
            (StyleType.ANIME_2D, StyleType.PAINTERLY): 0.6,
            (StyleType.ANIME_2D, StyleType.SKETCH): 0.5,
            (StyleType.ANIME_2D, StyleType.PHOTOGRAPHIC): 0.1,
            
            (StyleType.CARTOON_2D, StyleType.CARTOON_2D): 1.0,
            (StyleType.CARTOON_2D, StyleType.ARTISTIC_2D): 0.8,
            (StyleType.CARTOON_2D, StyleType.SKETCH): 0.7,
            (StyleType.CARTOON_2D, StyleType.PAINTERLY): 0.5,
            (StyleType.CARTOON_2D, StyleType.PHOTOGRAPHIC): 0.1,
            
            (StyleType.ARTISTIC_2D, StyleType.ARTISTIC_2D): 1.0,
            (StyleType.ARTISTIC_2D, StyleType.PAINTERLY): 0.9,
            (StyleType.ARTISTIC_2D, StyleType.SKETCH): 0.8,
            (StyleType.ARTISTIC_2D, StyleType.PHOTOGRAPHIC): 0.3,
            
            (StyleType.PHOTOGRAPHIC, StyleType.PHOTOGRAPHIC): 1.0,
            (StyleType.PHOTOGRAPHIC, StyleType.PAINTERLY): 0.4,
            (StyleType.PHOTOGRAPHIC, StyleType.SKETCH): 0.2,
            
            (StyleType.PAINTERLY, StyleType.PAINTERLY): 1.0,
            (StyleType.PAINTERLY, StyleType.SKETCH): 0.8,
            
            (StyleType.SKETCH, StyleType.SKETCH): 1.0,
        }
        
        # Make matrix symmetric
        symmetric_compatibility = {}
        for (style1, style2), score in self.compatibility.items():
            symmetric_compatibility[(style1, style2)] = score
            symmetric_compatibility[(style2, style1)] = score
        
        self.compatibility = symmetric_compatibility
    
    def get_compatibility(self, style1: StyleType, style2: StyleType) -> float:
        """Get compatibility score between two styles."""
        return self.compatibility.get((style1, style2), 0.0)


class StyleDetector:
    """Detects style types from prompts, LoRAs, and other inputs."""
    
    def __init__(self):
        # Style detection patterns
        self.style_patterns = {
            StyleType.REALISTIC_3D: [
                r'\b(realistic|photorealistic|3d|render|cgi|unreal engine)\b',
                r'\b(hyperrealistic|lifelike|detailed|high resolution)\b',
                r'\b(octane render|blender|maya|cinema 4d)\b'
            ],
            StyleType.ANIME_2D: [
                r'\b(anime|manga|japanese animation|cel shading)\b',
                r'\b(kawaii|moe|bishojo|bishonen)\b',
                r'\b(studio ghibli|makoto shinkai|anime style)\b'
            ],
            StyleType.CARTOON_2D: [
                r'\b(cartoon|animated|disney|pixar style)\b',
                r'\b(toon|cel animation|flat colors)\b',
                r'\b(comic book|graphic novel|illustration)\b'
            ],
            StyleType.ARTISTIC_2D: [
                r'\b(artistic|stylized|abstract|impressionist)\b',
                r'\b(watercolor|oil painting|acrylic|digital art)\b',
                r'\b(concept art|fantasy art|sci-fi art)\b'
            ],
            StyleType.PHOTOGRAPHIC: [
                r'\b(photograph|photo|camera|lens|aperture)\b',
                r'\b(portrait|landscape|macro|street photography)\b',
                r'\b(canon|nikon|sony|dslr|mirrorless)\b'
            ],
            StyleType.PAINTERLY: [
                r'\b(painting|painted|brush strokes|canvas)\b',
                r'\b(van gogh|monet|picasso|renaissance)\b',
                r'\b(oil on canvas|watercolor|gouache|tempera)\b'
            ],
            StyleType.SKETCH: [
                r'\b(sketch|drawing|pencil|charcoal|ink)\b',
                r'\b(line art|outline|draft|study)\b',
                r'\b(graphite|conte|pastel drawing)\b'
            ]
        }
        
        # LoRA style classifications (would be loaded from config)
        self.lora_styles = {
            "realistic_vision": StyleType.REALISTIC_3D,
            "anime_diffusion": StyleType.ANIME_2D,
            "cartoon_style": StyleType.CARTOON_2D,
            "artistic_vision": StyleType.ARTISTIC_2D,
            "photo_realistic": StyleType.PHOTOGRAPHIC,
            "oil_painting": StyleType.PAINTERLY,
            "sketch_style": StyleType.SKETCH,
        }
    
    def detect_prompt_styles(self, prompt: str) -> List[StyleComponent]:
        """Detect styles from text prompt."""
        components = []
        prompt_lower = prompt.lower()
        
        for style_type, patterns in self.style_patterns.items():
            strength = 0.0
            matches = []
            
            for pattern in patterns:
                pattern_matches = re.findall(pattern, prompt_lower, re.IGNORECASE)
                if pattern_matches:
                    strength += len(pattern_matches) * 0.3
                    matches.extend(pattern_matches)
            
            if strength > 0:
                # Normalize strength
                strength = min(strength, 1.0)
                
                component = StyleComponent(
                    name=f"prompt_{style_type.value}",
                    type=style_type,
                    strength=strength,
                    source="prompt",
                    priority=2,
                    compatibility_tags=matches
                )
                components.append(component)
        
        return components
    
    def detect_lora_styles(self, lora_configs: List[Dict[str, Any]]) -> List[StyleComponent]:
        """Detect styles from LoRA configurations."""
        components = []
        
        for lora_config in lora_configs:
            lora_name = lora_config.get("name", "").lower()
            lora_strength = lora_config.get("strength", 1.0)
            
            # Try to match LoRA name to known styles
            detected_style = None
            for known_lora, style_type in self.lora_styles.items():
                if known_lora in lora_name:
                    detected_style = style_type
                    break
            
            # If no direct match, try pattern matching on LoRA name
            if not detected_style:
                for style_type, patterns in self.style_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, lora_name, re.IGNORECASE):
                            detected_style = style_type
                            break
                    if detected_style:
                        break
            
            if detected_style:
                component = StyleComponent(
                    name=lora_name,
                    type=detected_style,
                    strength=lora_strength,
                    source="lora",
                    priority=3,  # LoRAs have higher priority than prompts
                    compatibility_tags=[lora_name]
                )
                components.append(component)
        
        return components
    
    def detect_embedding_styles(self, embeddings: List[Dict[str, Any]]) -> List[StyleComponent]:
        """Detect styles from textual inversions/embeddings."""
        components = []
        
        for embedding in embeddings:
            embedding_name = embedding.get("name", "").lower()
            embedding_strength = embedding.get("strength", 1.0)
            
            # Similar pattern matching as LoRAs
            detected_style = None
            for style_type, patterns in self.style_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, embedding_name, re.IGNORECASE):
                        detected_style = style_type
                        break
                if detected_style:
                    break
            
            if detected_style:
                component = StyleComponent(
                    name=embedding_name,
                    type=detected_style,
                    strength=embedding_strength,
                    source="embedding",
                    priority=2,
                    compatibility_tags=[embedding_name]
                )
                components.append(component)
        
        return components


class StyleConflictAnalyzer:
    """Analyzes conflicts between different style components."""
    
    def __init__(self):
        self.compatibility_matrix = StyleCompatibilityMatrix()
    
    def analyze_conflicts(self, components: List[StyleComponent]) -> StyleAnalysis:
        """Analyze style conflicts and provide recommendations."""
        if not components:
            return StyleAnalysis(
                components=[],
                dominant_style=StyleType.REALISTIC_3D,
                conflict_level=StyleConflictLevel.NONE,
                conflicts=[],
                harmonization_needed=False,
                recommended_adjustments=[]
            )
        
        # Group components by style type
        style_groups = {}
        for component in components:
            if component.type not in style_groups:
                style_groups[component.type] = []
            style_groups[component.type].append(component)
        
        # Calculate total strength per style
        style_strengths = {}
        for style_type, group_components in style_groups.items():
            total_strength = sum(comp.strength * comp.priority for comp in group_components)
            style_strengths[style_type] = total_strength
        
        # Determine dominant style
        dominant_style = max(style_strengths.keys(), key=lambda x: style_strengths[x])
        
        # Analyze conflicts
        conflicts = []
        max_conflict_score = 0.0
        
        style_types = list(style_groups.keys())
        for i, style1 in enumerate(style_types):
            for style2 in style_types[i+1:]:
                compatibility = self.compatibility_matrix.get_compatibility(style1, style2)
                conflict_score = 1.0 - compatibility
                
                if conflict_score > 0.3:  # Significant conflict
                    strength1 = style_strengths[style1]
                    strength2 = style_strengths[style2]
                    
                    conflict = {
                        "style1": style1,
                        "style2": style2,
                        "conflict_score": conflict_score,
                        "strength1": strength1,
                        "strength2": strength2,
                        "severity": self._get_conflict_severity(conflict_score, strength1, strength2)
                    }
                    conflicts.append(conflict)
                    max_conflict_score = max(max_conflict_score, conflict_score)
        
        # Determine overall conflict level
        conflict_level = self._determine_conflict_level(max_conflict_score, len(conflicts))
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            components, style_groups, style_strengths, conflicts, dominant_style
        )
        
        return StyleAnalysis(
            components=components,
            dominant_style=dominant_style,
            conflict_level=conflict_level,
            conflicts=conflicts,
            harmonization_needed=conflict_level != StyleConflictLevel.NONE,
            recommended_adjustments=recommendations
        )
    
    def _get_conflict_severity(self, conflict_score: float, strength1: float, strength2: float) -> str:
        """Determine severity of individual conflict."""
        combined_strength = strength1 + strength2
        weighted_conflict = conflict_score * combined_strength
        
        if weighted_conflict > 2.0:
            return "severe"
        elif weighted_conflict > 1.0:
            return "moderate"
        else:
            return "minor"
    
    def _determine_conflict_level(self, max_conflict_score: float, num_conflicts: int) -> StyleConflictLevel:
        """Determine overall conflict level."""
        if max_conflict_score == 0.0:
            return StyleConflictLevel.NONE
        elif max_conflict_score < 0.3:
            return StyleConflictLevel.MINOR
        elif max_conflict_score < 0.6:
            return StyleConflictLevel.MODERATE
        elif max_conflict_score < 0.8:
            return StyleConflictLevel.SEVERE
        else:
            return StyleConflictLevel.INCOMPATIBLE
    
    def _generate_recommendations(self, 
                                components: List[StyleComponent],
                                style_groups: Dict[StyleType, List[StyleComponent]],
                                style_strengths: Dict[StyleType, float],
                                conflicts: List[Dict[str, Any]],
                                dominant_style: StyleType) -> List[Dict[str, Any]]:
        """Generate harmonization recommendations."""
        recommendations = []
        
        # If there are severe conflicts, recommend reducing conflicting styles
        for conflict in conflicts:
            if conflict["severity"] in ["severe", "moderate"]:
                weaker_style = conflict["style1"] if conflict["strength1"] < conflict["strength2"] else conflict["style2"]
                stronger_style = conflict["style2"] if weaker_style == conflict["style1"] else conflict["style1"]
                
                if weaker_style != dominant_style:
                    recommendations.append({
                        "type": "reduce_strength",
                        "target_style": weaker_style,
                        "reason": f"Conflicts with dominant {stronger_style.value} style",
                        "suggested_reduction": 0.3 if conflict["severity"] == "moderate" else 0.5
                    })
        
        # Recommend style bridging for compatible styles
        compatible_pairs = []
        for conflict in conflicts:
            if conflict["conflict_score"] < 0.4:  # Somewhat compatible
                compatible_pairs.append((conflict["style1"], conflict["style2"]))
        
        if compatible_pairs:
            recommendations.append({
                "type": "style_bridging",
                "compatible_pairs": compatible_pairs,
                "reason": "Use transitional elements to blend compatible styles"
            })
        
        # Recommend dominant style reinforcement
        if len(style_groups) > 2:
            recommendations.append({
                "type": "reinforce_dominant",
                "target_style": dominant_style,
                "reason": "Strengthen dominant style to reduce conflicts",
                "suggested_boost": 0.2
            })
        
        return recommendations

class StyleHarmonizer:
    """Harmonizes conflicting styles to prevent weird-looking images."""
    
    def __init__(self):
        self.detector = StyleDetector()
        self.analyzer = StyleConflictAnalyzer()
        
        # Style transition mappings for bridging incompatible styles
        self.style_bridges = {
            (StyleType.REALISTIC_3D, StyleType.ANIME_2D): [
                "semi-realistic", "stylized realism", "3d anime style"
            ],
            (StyleType.REALISTIC_3D, StyleType.CARTOON_2D): [
                "stylized 3d", "pixar style", "semi-realistic cartoon"
            ],
            (StyleType.ANIME_2D, StyleType.PHOTOGRAPHIC): [
                "photorealistic anime", "live action anime style", "realistic manga"
            ],
            (StyleType.CARTOON_2D, StyleType.PHOTOGRAPHIC): [
                "live action cartoon", "realistic cartoon style", "stylized photography"
            ]
        }
    
    def harmonize_styles(self, 
                        prompt: str,
                        lora_configs: List[Dict[str, Any]] = None,
                        embeddings: List[Dict[str, Any]] = None,
                        user_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main harmonization function that analyzes and resolves style conflicts.
        
        Returns:
            Dictionary with harmonized prompt, adjusted LoRA strengths, and metadata
        """
        lora_configs = lora_configs or []
        embeddings = embeddings or []
        user_preferences = user_preferences or {}
        
        # Detect all style components
        prompt_styles = self.detector.detect_prompt_styles(prompt)
        lora_styles = self.detector.detect_lora_styles(lora_configs)
        embedding_styles = self.detector.detect_embedding_styles(embeddings)
        
        all_components = prompt_styles + lora_styles + embedding_styles
        
        # Analyze conflicts
        analysis = self.analyzer.analyze_conflicts(all_components)
        
        # Apply harmonization if needed
        if analysis.harmonization_needed:
            harmonized_result = self._apply_harmonization(
                prompt, lora_configs, embeddings, analysis, user_preferences
            )
        else:
            harmonized_result = {
                "harmonized_prompt": prompt,
                "adjusted_loras": lora_configs,
                "adjusted_embeddings": embeddings,
                "changes_made": []
            }
        
        # Add analysis metadata
        harmonized_result.update({
            "style_analysis": {
                "dominant_style": analysis.dominant_style.value,
                "conflict_level": analysis.conflict_level.value,
                "num_conflicts": len(analysis.conflicts),
                "harmonization_applied": analysis.harmonization_needed
            },
            "detected_styles": [
                {
                    "name": comp.name,
                    "type": comp.type.value,
                    "strength": comp.strength,
                    "source": comp.source
                }
                for comp in all_components
            ],
            "conflicts": [
                {
                    "style1": conflict["style1"].value,
                    "style2": conflict["style2"].value,
                    "severity": conflict["severity"],
                    "conflict_score": conflict["conflict_score"]
                }
                for conflict in analysis.conflicts
            ]
        })
        
        return harmonized_result
    
    def _apply_harmonization(self,
                           prompt: str,
                           lora_configs: List[Dict[str, Any]],
                           embeddings: List[Dict[str, Any]],
                           analysis: StyleAnalysis,
                           user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Apply harmonization based on analysis recommendations."""
        
        harmonized_prompt = prompt
        adjusted_loras = lora_configs.copy()
        adjusted_embeddings = embeddings.copy()
        changes_made = []
        
        # Get user preferences
        harmonization_mode = user_preferences.get("harmonization_mode", "balanced")  # balanced, preserve_dominant, blend_all
        max_strength_reduction = user_preferences.get("max_strength_reduction", 0.5)
        allow_prompt_modification = user_preferences.get("allow_prompt_modification", True)
        
        # Apply recommendations
        for recommendation in analysis.recommended_adjustments:
            if recommendation["type"] == "reduce_strength":
                target_style = recommendation["target_style"]
                reduction = min(recommendation["suggested_reduction"], max_strength_reduction)
                
                # Reduce LoRA strengths for this style
                for lora in adjusted_loras:
                    lora_name = lora.get("name", "").lower()
                    if self._lora_matches_style(lora_name, target_style):
                        original_strength = lora.get("strength", 1.0)
                        new_strength = original_strength * (1.0 - reduction)
                        lora["strength"] = max(new_strength, 0.1)  # Minimum strength
                        changes_made.append(f"Reduced {lora_name} LoRA strength from {original_strength:.2f} to {new_strength:.2f}")
                
                # Reduce embedding strengths for this style
                for embedding in adjusted_embeddings:
                    embedding_name = embedding.get("name", "").lower()
                    if self._embedding_matches_style(embedding_name, target_style):
                        original_strength = embedding.get("strength", 1.0)
                        new_strength = original_strength * (1.0 - reduction)
                        embedding["strength"] = max(new_strength, 0.1)
                        changes_made.append(f"Reduced {embedding_name} embedding strength from {original_strength:.2f} to {new_strength:.2f}")
            
            elif recommendation["type"] == "style_bridging" and allow_prompt_modification:
                # Add bridging terms to prompt
                bridge_terms = self._get_bridge_terms(recommendation["compatible_pairs"])
                if bridge_terms:
                    harmonized_prompt = self._add_bridge_terms(harmonized_prompt, bridge_terms)
                    changes_made.append(f"Added style bridging terms: {', '.join(bridge_terms)}")
            
            elif recommendation["type"] == "reinforce_dominant" and allow_prompt_modification:
                dominant_style = recommendation["target_style"]
                reinforcement_terms = self._get_reinforcement_terms(dominant_style)
                if reinforcement_terms:
                    harmonized_prompt = self._add_reinforcement_terms(harmonized_prompt, reinforcement_terms)
                    changes_made.append(f"Added {dominant_style.value} reinforcement terms")
        
        # Apply conflict-specific fixes
        if analysis.conflict_level == StyleConflictLevel.SEVERE:
            harmonized_prompt, additional_changes = self._apply_severe_conflict_fixes(
                harmonized_prompt, analysis, user_preferences
            )
            changes_made.extend(additional_changes)
        
        return {
            "harmonized_prompt": harmonized_prompt,
            "adjusted_loras": adjusted_loras,
            "adjusted_embeddings": adjusted_embeddings,
            "changes_made": changes_made
        }
    
    def _lora_matches_style(self, lora_name: str, style_type: StyleType) -> bool:
        """Check if LoRA name matches a style type."""
        patterns = self.detector.style_patterns.get(style_type, [])
        for pattern in patterns:
            if re.search(pattern, lora_name, re.IGNORECASE):
                return True
        return False
    
    def _embedding_matches_style(self, embedding_name: str, style_type: StyleType) -> bool:
        """Check if embedding name matches a style type."""
        return self._lora_matches_style(embedding_name, style_type)
    
    def _get_bridge_terms(self, compatible_pairs: List[Tuple[StyleType, StyleType]]) -> List[str]:
        """Get bridging terms for compatible style pairs."""
        bridge_terms = []
        for style1, style2 in compatible_pairs:
            pair_bridges = self.style_bridges.get((style1, style2)) or self.style_bridges.get((style2, style1))
            if pair_bridges:
                bridge_terms.extend(pair_bridges[:1])  # Add one bridge term per pair
        return bridge_terms
    
    def _add_bridge_terms(self, prompt: str, bridge_terms: List[str]) -> str:
        """Add bridging terms to prompt."""
        if not bridge_terms:
            return prompt
        
        # Add terms at the beginning with moderate weight
        bridge_text = ", ".join(f"({term}:0.8)" for term in bridge_terms)
        return f"{bridge_text}, {prompt}"
    
    def _get_reinforcement_terms(self, style_type: StyleType) -> List[str]:
        """Get reinforcement terms for a style type."""
        reinforcement_map = {
            StyleType.REALISTIC_3D: ["highly detailed", "photorealistic", "sharp focus"],
            StyleType.ANIME_2D: ["anime style", "cel shading", "clean lines"],
            StyleType.CARTOON_2D: ["cartoon style", "flat colors", "simple shading"],
            StyleType.ARTISTIC_2D: ["artistic", "stylized", "creative"],
            StyleType.PHOTOGRAPHIC: ["photograph", "professional photography", "high quality"],
            StyleType.PAINTERLY: ["painted", "brush strokes", "artistic medium"],
            StyleType.SKETCH: ["sketch", "line art", "drawing"]
        }
        return reinforcement_map.get(style_type, [])
    
    def _add_reinforcement_terms(self, prompt: str, reinforcement_terms: List[str]) -> str:
        """Add reinforcement terms to prompt."""
        if not reinforcement_terms:
            return prompt
        
        # Add terms with higher weight to reinforce dominant style
        reinforcement_text = ", ".join(f"({term}:1.1)" for term in reinforcement_terms[:2])
        return f"{prompt}, {reinforcement_text}"
    
    def _apply_severe_conflict_fixes(self, 
                                   prompt: str, 
                                   analysis: StyleAnalysis,
                                   user_preferences: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Apply fixes for severe style conflicts."""
        changes_made = []
        
        # For severe conflicts, add negative prompts to suppress conflicting elements
        conflicting_styles = set()
        for conflict in analysis.conflicts:
            if conflict["severity"] == "severe":
                # Suppress the weaker conflicting style
                weaker_style = conflict["style1"] if conflict["strength1"] < conflict["strength2"] else conflict["style2"]
                conflicting_styles.add(weaker_style)
        
        if conflicting_styles and user_preferences.get("allow_negative_prompts", True):
            negative_terms = []
            for style in conflicting_styles:
                if style != analysis.dominant_style:
                    style_negative_terms = self._get_negative_terms(style)
                    negative_terms.extend(style_negative_terms[:2])  # Limit negative terms
            
            if negative_terms:
                # Add negative terms to suppress conflicting styles
                negative_text = ", ".join(negative_terms)
                prompt = f"{prompt} [negative: {negative_text}]"
                changes_made.append(f"Added negative terms to suppress conflicts: {negative_text}")
        
        return prompt, changes_made
    
    def _get_negative_terms(self, style_type: StyleType) -> List[str]:
        """Get negative terms to suppress a style type."""
        negative_map = {
            StyleType.REALISTIC_3D: ["flat", "2d", "cartoon", "anime"],
            StyleType.ANIME_2D: ["realistic", "photographic", "3d render"],
            StyleType.CARTOON_2D: ["realistic", "detailed", "photographic"],
            StyleType.PHOTOGRAPHIC: ["painted", "drawn", "artistic", "stylized"],
            StyleType.PAINTERLY: ["photographic", "digital", "clean"],
            StyleType.SKETCH: ["colored", "painted", "photographic"]
        }
        return negative_map.get(style_type, [])


def create_style_harmonization_system() -> StyleHarmonizer:
    """Create and initialize the style harmonization system."""
    return StyleHarmonizer()


# Example usage and testing functions
def test_style_harmonization():
    """Test the style harmonization system."""
    harmonizer = StyleHarmonizer()
    
    # Test case 1: Conflicting 2D/3D styles
    test_prompt = "realistic 3d render of anime girl in cartoon style"
    test_loras = [
        {"name": "realistic_vision_v2", "strength": 1.0},
        {"name": "anime_diffusion", "strength": 0.8}
    ]
    
    result = harmonizer.harmonize_styles(
        prompt=test_prompt,
        lora_configs=test_loras,
        user_preferences={"harmonization_mode": "balanced"}
    )
    
    print("Test Case 1 - Conflicting 2D/3D:")
    print(f"Original: {test_prompt}")
    print(f"Harmonized: {result['harmonized_prompt']}")
    print(f"Conflict Level: {result['style_analysis']['conflict_level']}")
    print(f"Changes: {result['changes_made']}")
    print()
    
    return result


if __name__ == "__main__":
    test_style_harmonization()