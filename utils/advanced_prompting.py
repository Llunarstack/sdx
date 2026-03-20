"""
Advanced Prompt Engineering System - Address over-specification and prompt optimization.
Implements intelligent prompt structuring, priority weighting, and adaptive prompting.
"""
import re
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class PromptElement:
    """Represents a single element in a prompt."""
    content: str
    category: str
    priority: float
    weight: float = 1.0
    conflicts: tuple = None
    
    def __post_init__(self):
        if self.conflicts is None:
            self.conflicts = ()
    
    def __hash__(self):
        return hash((self.content, self.category, self.priority, self.weight, self.conflicts))
    
    def __eq__(self, other):
        if not isinstance(other, PromptElement):
            return False
        return (self.content == other.content and 
                self.category == other.category and 
                self.priority == other.priority and 
                self.weight == other.weight and 
                self.conflicts == other.conflicts)


@dataclass
class PromptStructure:
    """Represents the structure of an optimized prompt."""
    subject: List[PromptElement]
    style: List[PromptElement]
    composition: List[PromptElement]
    quality: List[PromptElement]
    technical: List[PromptElement]
    negative: List[PromptElement]
    
    def get_all_elements(self) -> List[PromptElement]:
        """Get all elements in priority order."""
        all_elements = (
            self.subject + self.style + self.composition + 
            self.quality + self.technical
        )
        return sorted(all_elements, key=lambda x: x.priority, reverse=True)


class PromptAnalyzer:
    """Analyze and categorize prompt elements."""
    
    def __init__(self):
        self.categories = {
            "subject": {
                "keywords": ["person", "woman", "man", "girl", "boy", "character", "figure", 
                           "animal", "cat", "dog", "bird", "creature", "object"],
                "priority_base": 0.9
            },
            "action": {
                "keywords": ["standing", "sitting", "walking", "running", "dancing", "jumping",
                           "holding", "wearing", "looking", "smiling", "crying"],
                "priority_base": 0.8
            },
            "style": {
                "keywords": ["realistic", "anime", "cartoon", "photorealistic", "artistic",
                           "oil painting", "watercolor", "digital art", "concept art"],
                "priority_base": 0.7
            },
            "composition": {
                "keywords": ["portrait", "full body", "close up", "wide shot", "from above",
                           "from below", "side view", "front view", "three quarter"],
                "priority_base": 0.6
            },
            "environment": {
                "keywords": ["background", "forest", "city", "beach", "mountain", "indoor",
                           "outdoor", "studio", "landscape", "room"],
                "priority_base": 0.5
            },
            "lighting": {
                "keywords": ["bright", "dark", "soft light", "harsh light", "natural light",
                           "dramatic lighting", "sunset", "sunrise", "golden hour"],
                "priority_base": 0.4
            },
            "quality": {
                "keywords": ["masterpiece", "best quality", "high quality", "detailed",
                           "sharp", "clear", "professional", "8k", "4k"],
                "priority_base": 0.3
            },
            "technical": {
                "keywords": ["depth of field", "bokeh", "lens flare", "motion blur",
                           "film grain", "chromatic aberration", "vignette"],
                "priority_base": 0.2
            }
        }
        
        self.conflict_groups = {
            "style_conflicts": [
                ["realistic", "anime", "cartoon"],
                ["photorealistic", "artistic", "stylized"],
                ["detailed", "simple", "minimalist"]
            ],
            "composition_conflicts": [
                ["portrait", "full body", "close up"],
                ["from above", "from below", "eye level"],
                ["wide shot", "close up", "medium shot"]
            ],
            "lighting_conflicts": [
                ["bright", "dark", "dim"],
                ["soft light", "harsh light", "dramatic lighting"],
                ["natural light", "artificial light", "studio lighting"]
            ]
        }
        
        # Prompt length optimization
        self.length_thresholds = {
            "short": 50,      # Under 50 chars - too short
            "optimal": 200,   # 50-200 chars - optimal range
            "long": 400,      # 200-400 chars - acceptable
            "too_long": 400   # Over 400 chars - likely over-specified
        }
    
    def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """Analyze a prompt and return detailed breakdown."""
        elements = self._parse_prompt_elements(prompt)
        categorized = self._categorize_elements(elements)
        conflicts = self._detect_conflicts(categorized)
        complexity = self._assess_complexity(prompt, categorized)
        
        return {
            "original_prompt": prompt,
            "length": len(prompt),
            "word_count": len(prompt.split()),
            "elements": categorized,
            "conflicts": conflicts,
            "complexity_score": complexity,
            "optimization_needed": complexity > 0.7 or len(conflicts) > 0,
            "recommendations": self._generate_recommendations(prompt, categorized, conflicts, complexity)
        }
    
    def _parse_prompt_elements(self, prompt: str) -> List[str]:
        """Parse prompt into individual elements."""
        # Split by commas and clean up
        elements = [elem.strip() for elem in prompt.split(',')]
        
        # Remove empty elements
        elements = [elem for elem in elements if elem]
        
        # Handle parentheses and brackets (emphasis)
        processed_elements = []
        for elem in elements:
            # Extract emphasis level
            emphasis_level = 1.0
            clean_elem = elem
            
            # Count parentheses for emphasis
            open_parens = elem.count('(')
            close_parens = elem.count(')')
            if open_parens > 0 and close_parens > 0:
                emphasis_level = 1.0 + (min(open_parens, close_parens) * 0.2)
                clean_elem = re.sub(r'[()]', '', elem).strip()
            
            # Count brackets for de-emphasis
            open_brackets = elem.count('[')
            close_brackets = elem.count(']')
            if open_brackets > 0 and close_brackets > 0:
                emphasis_level = 1.0 - (min(open_brackets, close_brackets) * 0.2)
                clean_elem = re.sub(r'[\[\]]', '', elem).strip()
            
            if clean_elem:
                processed_elements.append((clean_elem, emphasis_level))
        
        return processed_elements
    
    def _categorize_elements(self, elements: List[Tuple[str, float]]) -> Dict[str, List[PromptElement]]:
        """Categorize prompt elements by type."""
        categorized = {category: [] for category in self.categories.keys()}
        
        for element_text, emphasis in elements:
            element_lower = element_text.lower()
            categorized_flag = False
            
            # Check each category
            for category, info in self.categories.items():
                for keyword in info["keywords"]:
                    if keyword in element_lower:
                        priority = info["priority_base"] * emphasis
                        
                        prompt_element = PromptElement(
                            content=element_text,
                            category=category,
                            priority=priority,
                            weight=emphasis
                        )
                        
                        categorized[category].append(prompt_element)
                        categorized_flag = True
                        break
                
                if categorized_flag:
                    break
            
            if not categorized_flag:
                # Default category based on position and content
                default_category = self._determine_default_category(element_text)
                priority = 0.5 * emphasis
                
                prompt_element = PromptElement(
                    content=element_text,
                    category=default_category,
                    priority=priority,
                    weight=emphasis
                )
                
                categorized[default_category].append(prompt_element)
        
        return categorized
    
    def _determine_default_category(self, element: str) -> str:
        """Determine default category for uncategorized elements."""
        element_lower = element.lower()
        
        # Simple heuristics
        if any(word in element_lower for word in ["color", "red", "blue", "green", "black", "white"]):
            return "style"
        elif any(word in element_lower for word in ["beautiful", "pretty", "handsome", "cute"]):
            return "quality"
        elif len(element.split()) == 1:
            return "subject"  # Single words often describe subjects
        else:
            return "environment"  # Multi-word phrases often describe environment
    
    def _detect_conflicts(self, categorized: Dict[str, List[PromptElement]]) -> List[Dict[str, Any]]:
        """Detect conflicting elements in the prompt."""
        conflicts = []
        
        for conflict_type, conflict_groups in self.conflict_groups.items():
            for conflict_group in conflict_groups:
                found_conflicts = []
                
                # Check all categories for conflicting terms
                for category, elements in categorized.items():
                    for element in elements:
                        element_lower = element.content.lower()
                        for conflict_term in conflict_group:
                            if conflict_term in element_lower:
                                found_conflicts.append({
                                    "element": element.content,
                                    "category": category,
                                    "conflict_term": conflict_term,
                                    "priority": element.priority
                                })
                
                # If we found multiple conflicting terms, report the conflict
                if len(found_conflicts) > 1:
                    conflicts.append({
                        "type": conflict_type,
                        "conflicting_elements": found_conflicts,
                        "severity": "high" if len(found_conflicts) > 2 else "medium"
                    })
        
        return conflicts
    
    def _assess_complexity(self, prompt: str, categorized: Dict[str, List[PromptElement]]) -> float:
        """Assess prompt complexity score (0-1)."""
        complexity_factors = {
            "length": min(len(prompt) / 500, 1.0) * 0.3,  # Length factor
            "element_count": min(sum(len(elements) for elements in categorized.values()) / 20, 1.0) * 0.2,
            "category_diversity": len([cat for cat, elements in categorized.items() if elements]) / len(self.categories) * 0.2,
            "emphasis_usage": self._count_emphasis_markers(prompt) / 10 * 0.1,
            "technical_terms": self._count_technical_terms(prompt) / 5 * 0.2
        }
        
        return sum(complexity_factors.values())
    
    def _count_emphasis_markers(self, prompt: str) -> int:
        """Count emphasis markers in prompt."""
        return prompt.count('(') + prompt.count('[')
    
    def _count_technical_terms(self, prompt: str) -> int:
        """Count technical photography/art terms."""
        technical_terms = [
            "depth of field", "bokeh", "aperture", "iso", "shutter speed",
            "composition", "rule of thirds", "golden ratio", "leading lines",
            "chromatic aberration", "vignette", "film grain", "lens flare"
        ]
        
        count = 0
        prompt_lower = prompt.lower()
        for term in technical_terms:
            if term in prompt_lower:
                count += 1
        
        return count
    
    def _generate_recommendations(self, prompt: str, categorized: Dict[str, List[PromptElement]], 
                                conflicts: List[Dict[str, Any]], complexity: float) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Length recommendations
        prompt_length = len(prompt)
        if prompt_length < self.length_thresholds["short"]:
            recommendations.append("Prompt is too short - add more descriptive details")
        elif prompt_length > self.length_thresholds["too_long"]:
            recommendations.append("Prompt is too long - consider simplifying or using multi-stage generation")
        
        # Conflict recommendations
        if conflicts:
            recommendations.append(f"Found {len(conflicts)} conflicts - resolve contradictory terms")
            for conflict in conflicts:
                if conflict["severity"] == "high":
                    recommendations.append(f"High priority: resolve {conflict['type']} conflicts")
        
        # Complexity recommendations
        if complexity > 0.8:
            recommendations.append("Very high complexity - consider breaking into multiple prompts")
        elif complexity > 0.6:
            recommendations.append("High complexity - prioritize most important elements")
        
        # Category balance recommendations
        element_counts = {cat: len(elements) for cat, elements in categorized.items() if elements}
        
        if "subject" not in element_counts or element_counts["subject"] == 0:
            recommendations.append("Add clear subject description")
        
        if element_counts.get("quality", 0) == 0:
            recommendations.append("Consider adding quality tags (masterpiece, best quality)")
        
        if element_counts.get("style", 0) == 0:
            recommendations.append("Consider specifying art style")
        
        # Over-specification warnings
        if element_counts.get("technical", 0) > 3:
            recommendations.append("Too many technical terms - may confuse the model")
        
        if sum(element_counts.values()) > 15:
            recommendations.append("Too many elements - focus on the most important ones")
        
        return recommendations


class PromptOptimizer:
    """Optimize prompts for better generation results."""
    
    def __init__(self):
        self.analyzer = PromptAnalyzer()
        
        # Optimization strategies
        self.optimization_strategies = {
            "simplify": self._simplify_prompt,
            "prioritize": self._prioritize_elements,
            "resolve_conflicts": self._resolve_conflicts,
            "balance_categories": self._balance_categories,
            "enhance_clarity": self._enhance_clarity
        }
        
        # Quality enhancers by category
        self.quality_enhancers = {
            "general": ["masterpiece", "best quality", "high quality"],
            "realistic": ["photorealistic", "detailed", "sharp focus"],
            "artistic": ["artistic", "creative", "expressive"],
            "anime": ["anime style", "clean lines", "vibrant colors"],
            "portrait": ["detailed face", "clear eyes", "natural expression"],
            "landscape": ["scenic", "atmospheric", "wide view"]
        }
    
    def optimize_prompt(self, prompt: str, 
                       optimization_level: str = "balanced",
                       target_style: str = None,
                       max_length: int = 300) -> Dict[str, Any]:
        """Optimize a prompt using various strategies."""
        # Analyze original prompt
        analysis = self.analyzer.analyze_prompt(prompt)
        
        # Apply optimization strategies based on level
        if optimization_level == "minimal":
            strategies = ["resolve_conflicts"]
        elif optimization_level == "balanced":
            strategies = ["resolve_conflicts", "prioritize", "enhance_clarity"]
        elif optimization_level == "aggressive":
            strategies = ["simplify", "resolve_conflicts", "prioritize", "balance_categories"]
        else:
            strategies = ["prioritize", "enhance_clarity"]
        
        # Apply strategies
        optimized_elements = analysis["elements"]
        optimization_log = []
        
        for strategy in strategies:
            if strategy in self.optimization_strategies:
                optimized_elements, log_entry = self.optimization_strategies[strategy](
                    optimized_elements, analysis, target_style
                )
                optimization_log.append(log_entry)
        
        # Build optimized prompt
        optimized_prompt = self._build_optimized_prompt(
            optimized_elements, max_length, target_style
        )
        
        # Analyze optimized prompt
        optimized_analysis = self.analyzer.analyze_prompt(optimized_prompt)
        
        return {
            "original_prompt": prompt,
            "optimized_prompt": optimized_prompt,
            "original_analysis": analysis,
            "optimized_analysis": optimized_analysis,
            "optimization_log": optimization_log,
            "improvement_score": self._calculate_improvement_score(analysis, optimized_analysis)
        }
    
    def _simplify_prompt(self, elements: Dict[str, List[PromptElement]], 
                        analysis: Dict[str, Any], target_style: str) -> Tuple[Dict[str, List[PromptElement]], str]:
        """Simplify prompt by removing low-priority elements."""
        simplified = {}
        removed_count = 0
        
        for category, element_list in elements.items():
            # Keep only high-priority elements
            simplified[category] = [
                elem for elem in element_list 
                if elem.priority > 0.5 or category in ["subject", "action"]
            ]
            removed_count += len(element_list) - len(simplified[category])
        
        log_entry = f"Simplified: removed {removed_count} low-priority elements"
        return simplified, log_entry
    
    def _prioritize_elements(self, elements: Dict[str, List[PromptElement]], 
                           analysis: Dict[str, Any], target_style: str) -> Tuple[Dict[str, List[PromptElement]], str]:
        """Prioritize elements by importance and relevance."""
        prioritized = {}
        
        # Priority order for categories
        category_priority = {
            "subject": 1.0,
            "action": 0.9,
            "style": 0.8,
            "composition": 0.7,
            "environment": 0.6,
            "lighting": 0.5,
            "quality": 0.4,
            "technical": 0.3
        }
        
        for category, element_list in elements.items():
            # Adjust priorities based on category importance
            category_multiplier = category_priority.get(category, 0.5)
            
            prioritized[category] = []
            for elem in element_list:
                new_priority = elem.priority * category_multiplier
                
                # Boost priority if matches target style
                if target_style and target_style.lower() in elem.content.lower():
                    new_priority *= 1.3
                
                elem.priority = new_priority
                prioritized[category].append(elem)
            
            # Sort by priority within category
            prioritized[category].sort(key=lambda x: x.priority, reverse=True)
        
        log_entry = "Prioritized: adjusted element priorities by category and relevance"
        return prioritized, log_entry
    
    def _resolve_conflicts(self, elements: Dict[str, List[PromptElement]], 
                          analysis: Dict[str, Any], target_style: str) -> Tuple[Dict[str, List[PromptElement]], str]:
        """Resolve conflicting elements by keeping highest priority ones."""
        resolved = {}
        conflicts_resolved = 0
        
        for category, element_list in elements.items():
            resolved[category] = []
            
            # Group potentially conflicting elements
            conflict_groups = self._identify_element_conflicts(element_list)
            
            for group in conflict_groups:
                if len(group) > 1:
                    # Keep only the highest priority element from conflicting group
                    best_element = max(group, key=lambda x: x.priority)
                    resolved[category].append(best_element)
                    conflicts_resolved += len(group) - 1
                else:
                    resolved[category].extend(group)
            
            # Add non-conflicting elements
            all_conflicting = set()
            for group in conflict_groups:
                all_conflicting.update(group)
            
            for elem in element_list:
                if elem not in all_conflicting:
                    resolved[category].append(elem)
        
        log_entry = f"Resolved conflicts: removed {conflicts_resolved} conflicting elements"
        return resolved, log_entry
    
    def _balance_categories(self, elements: Dict[str, List[PromptElement]], 
                           analysis: Dict[str, Any], target_style: str) -> Tuple[Dict[str, List[PromptElement]], str]:
        """Balance elements across categories."""
        balanced = {}
        
        # Target distribution (max elements per category)
        max_per_category = {
            "subject": 3,
            "action": 2,
            "style": 2,
            "composition": 2,
            "environment": 2,
            "lighting": 1,
            "quality": 2,
            "technical": 1
        }
        
        total_trimmed = 0
        
        for category, element_list in elements.items():
            max_allowed = max_per_category.get(category, 2)
            
            if len(element_list) > max_allowed:
                # Keep top elements by priority
                balanced[category] = sorted(element_list, key=lambda x: x.priority, reverse=True)[:max_allowed]
                total_trimmed += len(element_list) - max_allowed
            else:
                balanced[category] = element_list
        
        log_entry = f"Balanced categories: trimmed {total_trimmed} elements for better distribution"
        return balanced, log_entry
    
    def _enhance_clarity(self, elements: Dict[str, List[PromptElement]], 
                        analysis: Dict[str, Any], target_style: str) -> Tuple[Dict[str, List[PromptElement]], str]:
        """Enhance prompt clarity by adding missing essential elements."""
        enhanced = dict(elements)  # Copy
        additions = 0
        
        # Check for missing essential elements
        element_counts = {cat: len(elem_list) for cat, elem_list in elements.items()}
        
        # Add quality tags if missing
        if element_counts.get("quality", 0) == 0:
            quality_tags = self.quality_enhancers["general"][:2]
            for tag in quality_tags:
                enhanced["quality"].append(PromptElement(
                    content=tag,
                    category="quality",
                    priority=0.4,
                    weight=1.0
                ))
                additions += 1
        
        # Add style clarity if target style specified
        if target_style and element_counts.get("style", 0) == 0:
            enhanced["style"].append(PromptElement(
                content=target_style,
                category="style",
                priority=0.8,
                weight=1.0
            ))
            additions += 1
        
        log_entry = f"Enhanced clarity: added {additions} essential elements"
        return enhanced, log_entry
    
    def _identify_element_conflicts(self, elements: List[PromptElement]) -> List[List[PromptElement]]:
        """Identify conflicting elements within a category."""
        # This is simplified - would be more sophisticated in practice
        conflict_keywords = [
            ["realistic", "anime", "cartoon"],
            ["detailed", "simple", "minimalist"],
            ["bright", "dark", "dim"],
            ["large", "small", "tiny"],
            ["old", "young", "new"]
        ]
        
        conflict_groups = []
        used_elements = set()
        
        for keywords in conflict_keywords:
            conflicting_elements = []
            
            for element in elements:
                if element in used_elements:
                    continue
                
                element_lower = element.content.lower()
                for keyword in keywords:
                    if keyword in element_lower:
                        conflicting_elements.append(element)
                        used_elements.add(element)
                        break
            
            if conflicting_elements:
                conflict_groups.append(conflicting_elements)
        
        # Add non-conflicting elements as individual groups
        for element in elements:
            if element not in used_elements:
                conflict_groups.append([element])
        
        return conflict_groups
    
    def _build_optimized_prompt(self, elements: Dict[str, List[PromptElement]], 
                               max_length: int, target_style: str) -> str:
        """Build the final optimized prompt."""
        # Collect all elements and sort by priority
        all_elements = []
        for category, element_list in elements.items():
            all_elements.extend(element_list)
        
        all_elements.sort(key=lambda x: x.priority, reverse=True)
        
        # Build prompt respecting length limit
        prompt_parts = []
        current_length = 0
        
        for element in all_elements:
            element_text = element.content
            
            # Add emphasis markers based on weight
            if element.weight > 1.2:
                element_text = f"({element_text})"
            elif element.weight < 0.8:
                element_text = f"[{element_text}]"
            
            # Check if adding this element would exceed length limit
            if current_length + len(element_text) + 2 > max_length:  # +2 for ", "
                break
            
            prompt_parts.append(element_text)
            current_length += len(element_text) + 2
        
        return ", ".join(prompt_parts)
    
    def _calculate_improvement_score(self, original_analysis: Dict[str, Any], 
                                   optimized_analysis: Dict[str, Any]) -> float:
        """Calculate improvement score between original and optimized prompts."""
        improvements = 0.0
        
        # Conflict reduction
        original_conflicts = len(original_analysis.get("conflicts", []))
        optimized_conflicts = len(optimized_analysis.get("conflicts", []))
        if original_conflicts > optimized_conflicts:
            improvements += 0.3
        
        # Complexity reduction
        original_complexity = original_analysis.get("complexity_score", 0)
        optimized_complexity = optimized_analysis.get("complexity_score", 0)
        if original_complexity > optimized_complexity:
            improvements += 0.3
        
        # Length optimization
        original_length = original_analysis.get("length", 0)
        optimized_length = optimized_analysis.get("length", 0)
        if 50 <= optimized_length <= 300 and abs(optimized_length - 200) < abs(original_length - 200):
            improvements += 0.2
        
        # Recommendation reduction
        original_recs = len(original_analysis.get("recommendations", []))
        optimized_recs = len(optimized_analysis.get("recommendations", []))
        if original_recs > optimized_recs:
            improvements += 0.2
        
        return min(improvements, 1.0)


def create_advanced_prompting_system():
    """Create complete advanced prompting system."""
    return {
        'analyzer': PromptAnalyzer(),
        'optimizer': PromptOptimizer()
    }