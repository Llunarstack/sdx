"""
Anatomy & Pose Correction System - Address anatomical accuracy and complex pose generation.
Implements pose validation, hand correction, and multi-person interaction handling.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BodyPart:
    """Represents a body part with anatomical constraints."""

    name: str
    position: Tuple[float, float]
    joints: List[Tuple[float, float]]
    constraints: Dict[str, Any]
    confidence: float = 1.0


@dataclass
class Pose:
    """Represents a complete human pose."""

    body_parts: Dict[str, BodyPart]
    pose_type: str = "standing"
    difficulty: float = 0.5
    issues: List[str] = None

    def __post_init__(self):
        if self.issues is None:
            self.issues = []


class AnatomyValidator:
    """Validate anatomical correctness in generated images."""

    def __init__(self):
        self.body_part_rules = {
            "hands": {
                "finger_count": 5,
                "thumb_position": "opposite",
                "joint_count": 14,  # 5 fingers × 3 joints - 1 (thumb has 2)
                "common_issues": ["extra_fingers", "missing_fingers", "twisted_joints", "impossible_bends"],
            },
            "arms": {
                "joint_count": 3,  # shoulder, elbow, wrist
                "bend_direction": "forward",
                "length_ratio": 1.0,  # arm length should be consistent
                "common_issues": ["wrong_bend_direction", "impossible_twist", "length_mismatch"],
            },
            "legs": {
                "joint_count": 3,  # hip, knee, ankle
                "bend_direction": "backward",
                "length_ratio": 1.0,
                "common_issues": ["wrong_bend_direction", "impossible_pose", "length_mismatch"],
            },
            "torso": {
                "proportions": {"shoulder_width": 2.0, "waist_ratio": 0.7},
                "common_issues": ["twisted_spine", "impossible_bend", "proportion_error"],
            },
            "head": {
                "features": ["eyes", "nose", "mouth", "ears"],
                "proportions": {"eye_distance": 1.0, "face_symmetry": 0.95},
                "common_issues": ["asymmetric_features", "missing_features", "distorted_proportions"],
            },
        }

        self.pose_templates = {
            "standing": {
                "description": "natural standing pose",
                "difficulty": 0.2,
                "key_points": ["upright_torso", "balanced_stance", "relaxed_arms"],
            },
            "sitting": {
                "description": "sitting pose with proper posture",
                "difficulty": 0.4,
                "key_points": ["bent_knees", "supported_back", "natural_arm_position"],
            },
            "walking": {
                "description": "natural walking motion",
                "difficulty": 0.6,
                "key_points": ["alternating_legs", "arm_swing", "forward_lean"],
            },
            "running": {
                "description": "dynamic running pose",
                "difficulty": 0.7,
                "key_points": ["extended_stride", "pumping_arms", "forward_momentum"],
            },
            "jumping": {
                "description": "mid-air jumping pose",
                "difficulty": 0.8,
                "key_points": ["bent_knees", "raised_arms", "airborne_position"],
            },
            "dancing": {
                "description": "expressive dance pose",
                "difficulty": 0.9,
                "key_points": ["fluid_movement", "artistic_expression", "balance"],
            },
        }

    def analyze_pose_complexity(self, pose_description: str) -> float:
        """Analyze the complexity/difficulty of a requested pose."""
        complexity_factors = {
            "multiple_people": 0.3,
            "interaction": 0.2,
            "complex_hand_pose": 0.2,
            "unusual_angle": 0.15,
            "dynamic_motion": 0.15,
        }

        total_complexity = 0.0
        description_lower = pose_description.lower()

        # Check for multiple people
        people_indicators = ["two people", "couple", "group", "multiple", "together"]
        if any(indicator in description_lower for indicator in people_indicators):
            total_complexity += complexity_factors["multiple_people"]

        # Check for interactions
        interaction_words = ["holding", "touching", "hugging", "shaking hands", "dancing together"]
        if any(word in description_lower for word in interaction_words):
            total_complexity += complexity_factors["interaction"]

        # Check for complex hand poses
        hand_complexity = ["playing instrument", "typing", "writing", "gesturing", "sign language"]
        if any(pose in description_lower for pose in hand_complexity):
            total_complexity += complexity_factors["complex_hand_pose"]

        # Check for unusual angles
        angle_words = ["from below", "from above", "side view", "back view", "three quarter"]
        if any(angle in description_lower for angle in angle_words):
            total_complexity += complexity_factors["unusual_angle"]

        # Check for dynamic motion
        motion_words = ["jumping", "running", "dancing", "falling", "flying", "spinning"]
        if any(motion in description_lower for motion in motion_words):
            total_complexity += complexity_factors["dynamic_motion"]

        return min(total_complexity, 1.0)

    def generate_anatomy_aware_prompt(self, base_prompt: str, focus_areas: List[str] = None) -> str:
        """Generate prompt with anatomy-specific enhancements."""
        if focus_areas is None:
            focus_areas = ["hands", "face", "posture"]

        anatomy_enhancers = {
            "hands": [
                "perfect hands",
                "correct fingers",
                "five fingers",
                "natural hand pose",
                "detailed hands",
                "anatomically correct hands",
                "well-formed fingers",
            ],
            "face": [
                "symmetrical face",
                "natural expression",
                "correct proportions",
                "detailed facial features",
                "realistic face",
                "proper eye placement",
            ],
            "posture": [
                "natural posture",
                "correct anatomy",
                "realistic proportions",
                "proper body mechanics",
                "anatomically accurate",
                "believable pose",
            ],
            "arms": [
                "natural arm position",
                "correct arm length",
                "proper elbow bend",
                "realistic arm pose",
                "anatomically correct arms",
            ],
            "legs": [
                "natural leg position",
                "correct proportions",
                "proper knee bend",
                "realistic stance",
                "anatomically correct legs",
            ],
        }

        enhancements = []
        for area in focus_areas:
            if area in anatomy_enhancers:
                # Add 2-3 enhancers per focus area
                enhancements.extend(anatomy_enhancers[area][:3])

        # Add general anatomy enhancers
        general_enhancers = [
            "anatomically correct",
            "realistic human anatomy",
            "proper proportions",
            "natural pose",
            "believable human figure",
        ]

        enhancements.extend(general_enhancers[:2])

        return f"{base_prompt}, {', '.join(enhancements)}"

    def create_pose_reference_prompt(self, pose_type: str, person_description: str = "person") -> str:
        """Create a reference prompt for a specific pose type."""
        if pose_type not in self.pose_templates:
            pose_type = "standing"

        template = self.pose_templates[pose_type]

        prompt_parts = [
            f"{person_description} in {template['description']}",
            ", ".join(template["key_points"]),
            "anatomically correct",
            "natural pose",
            "realistic proportions",
            "professional reference pose",
        ]

        return ", ".join(prompt_parts)

    def suggest_pose_corrections(self, pose_description: str) -> List[str]:
        """Suggest corrections for problematic pose descriptions."""
        suggestions = []
        description_lower = pose_description.lower()

        # Check for hand-related issues
        if "hands" in description_lower:
            suggestions.extend(
                [
                    "Specify 'five fingers on each hand' for accuracy",
                    "Add 'natural hand pose' to avoid twisted fingers",
                    "Consider 'detailed hands' for better finger definition",
                ]
            )

        # Check for complex interactions
        if any(word in description_lower for word in ["holding", "touching", "grabbing"]):
            suggestions.extend(
                [
                    "Break complex interactions into simpler components",
                    "Specify exact hand positions for object interactions",
                    "Consider generating objects and people separately, then compositing",
                ]
            )

        # Check for multiple people
        if any(word in description_lower for word in ["two people", "couple", "group"]):
            suggestions.extend(
                [
                    "Generate each person separately for better anatomy",
                    "Use clear spatial descriptions (left person, right person)",
                    "Avoid overlapping bodies which cause anatomy issues",
                ]
            )

        # Check for unusual poses
        unusual_poses = ["upside down", "twisted", "contorted", "impossible"]
        if any(pose in description_lower for pose in unusual_poses):
            suggestions.extend(
                [
                    "Consider if the pose is physically possible",
                    "Break complex poses into multiple generation steps",
                    "Use reference images for unusual poses",
                ]
            )

        return suggestions


class HandCorrector:
    """Specialized system for hand generation and correction."""

    def __init__(self):
        self.hand_poses = {
            "relaxed": "natural relaxed hand, fingers slightly curved, thumb visible",
            "fist": "closed fist, thumb outside fingers, natural knuckle position",
            "pointing": "index finger extended, other fingers curled, natural gesture",
            "open_palm": "open palm facing forward, five fingers spread naturally",
            "holding": "hand in grasping position, fingers curved around object",
            "peace_sign": "index and middle finger extended in V shape, other fingers down",
            "thumbs_up": "thumb extended upward, fingers curled naturally",
            "waving": "hand raised, fingers slightly spread, natural wave gesture",
        }

        self.hand_enhancement_prompts = [
            "perfect hands",
            "five fingers",
            "correct finger count",
            "natural hand pose",
            "detailed fingers",
            "anatomically correct hands",
            "well-formed thumbs",
            "proper finger joints",
            "realistic hand proportions",
            "clear finger definition",
        ]

    def generate_hand_focused_prompt(
        self, base_prompt: str, hand_pose: str = "relaxed", emphasis_level: str = "high"
    ) -> str:
        """Generate prompt with strong focus on hand accuracy."""
        hand_description = self.hand_poses.get(hand_pose, self.hand_poses["relaxed"])

        if emphasis_level == "high":
            hand_enhancers = self.hand_enhancement_prompts[:6]
        elif emphasis_level == "medium":
            hand_enhancers = self.hand_enhancement_prompts[:4]
        else:
            hand_enhancers = self.hand_enhancement_prompts[:2]

        # Structure prompt to prioritize hands
        prompt_parts = [
            base_prompt,
            hand_description,
            ", ".join(hand_enhancers),
            "hands in focus",
            "clear hand details",
            "professional hand reference",
        ]

        return ", ".join(prompt_parts)

    def create_hand_validation_checklist(self) -> Dict[str, List[str]]:
        """Create checklist for validating hand generation."""
        return {
            "finger_count": [
                "Count exactly 5 fingers per hand",
                "Check thumb is separate from other fingers",
                "Verify no extra or missing digits",
            ],
            "finger_joints": [
                "Each finger has proper joint segments",
                "Joints bend in natural directions",
                "No impossible finger positions",
            ],
            "proportions": [
                "Fingers are proportional to hand size",
                "Thumb reaches to first joint of index finger",
                "Hand size matches body proportions",
            ],
            "pose_accuracy": [
                "Hand pose matches description",
                "Fingers follow natural curves",
                "Thumb position is anatomically correct",
            ],
        }


class MultiPersonComposer:
    """Handle multi-person scenes with proper interactions."""

    def __init__(self):
        self.interaction_types = {
            "conversation": {
                "positioning": "facing each other, appropriate distance",
                "body_language": "open posture, eye contact direction",
                "difficulty": 0.4,
            },
            "handshake": {
                "positioning": "standing close, right hands meeting",
                "body_language": "formal posture, direct engagement",
                "difficulty": 0.7,
            },
            "hugging": {
                "positioning": "close proximity, arms around each other",
                "body_language": "intimate gesture, bodies touching",
                "difficulty": 0.8,
            },
            "dancing": {
                "positioning": "coordinated movement, synchronized poses",
                "body_language": "fluid motion, artistic expression",
                "difficulty": 0.9,
            },
            "walking_together": {
                "positioning": "side by side, matching pace",
                "body_language": "synchronized movement, casual interaction",
                "difficulty": 0.5,
            },
        }

    def decompose_multi_person_scene(self, scene_description: str) -> List[Dict[str, Any]]:
        """Break down multi-person scene into manageable components."""
        components = []

        # Extract number of people
        people_count = self._extract_people_count(scene_description)

        # Extract interaction type
        interaction = self._identify_interaction(scene_description)

        # Create individual person components
        for i in range(people_count):
            person_component = {
                "type": "person",
                "id": f"person_{i + 1}",
                "description": self._extract_person_description(scene_description, i),
                "pose": self._determine_individual_pose(interaction, i, people_count),
                "position": self._calculate_person_position(interaction, i, people_count),
            }
            components.append(person_component)

        # Add interaction component if needed
        if interaction and interaction != "separate":
            interaction_component = {
                "type": "interaction",
                "interaction_type": interaction,
                "participants": [f"person_{i + 1}" for i in range(people_count)],
                "description": self._generate_interaction_description(interaction),
            }
            components.append(interaction_component)

        return components

    def _extract_people_count(self, description: str) -> int:
        """Extract number of people from description."""
        count_indicators = {
            "two people": 2,
            "couple": 2,
            "pair": 2,
            "duo": 2,
            "three people": 3,
            "trio": 3,
            "group of three": 3,
            "four people": 4,
            "group of four": 4,
            "quartet": 4,
            "group": 3,
            "crowd": 5,  # Default assumptions
        }

        description_lower = description.lower()
        for indicator, count in count_indicators.items():
            if indicator in description_lower:
                return count

        return 1  # Default to single person

    def _identify_interaction(self, description: str) -> Optional[str]:
        """Identify the type of interaction between people."""
        description_lower = description.lower()

        for interaction_type in self.interaction_types.keys():
            if interaction_type.replace("_", " ") in description_lower:
                return interaction_type

        # Check for interaction keywords
        interaction_keywords = {
            "talking": "conversation",
            "speaking": "conversation",
            "chatting": "conversation",
            "shaking hands": "handshake",
            "greeting": "handshake",
            "embracing": "hugging",
            "holding": "hugging",
            "dancing": "dancing",
            "walking": "walking_together",
        }

        for keyword, interaction in interaction_keywords.items():
            if keyword in description_lower:
                return interaction

        return None

    def _extract_person_description(self, description: str, person_index: int) -> str:
        """Extract description for a specific person."""
        # This is simplified - would be more sophisticated in practice
        base_description = description

        # Remove interaction terms to focus on person attributes
        interaction_terms = ["talking", "dancing", "hugging", "walking together", "shaking hands"]
        for term in interaction_terms:
            base_description = base_description.replace(term, "")

        return base_description.strip()

    def _determine_individual_pose(self, interaction: str, person_index: int, total_people: int) -> str:
        """Determine appropriate pose for individual in interaction."""
        if not interaction:
            return "standing"

        interaction_poses = {
            "conversation": "standing, facing partner, relaxed posture",
            "handshake": "standing, extending right hand, formal posture",
            "hugging": "standing, arms extended for embrace",
            "dancing": "dynamic dance pose, expressive movement",
            "walking_together": "walking pose, synchronized step",
        }

        return interaction_poses.get(interaction, "standing")

    def _calculate_person_position(self, interaction: str, person_index: int, total_people: int) -> Tuple[float, float]:
        """Calculate position for person in multi-person scene."""
        if total_people == 1:
            return (0.5, 0.5)  # Center

        if total_people == 2:
            if person_index == 0:
                return (0.35, 0.5)  # Left person
            else:
                return (0.65, 0.5)  # Right person

        # For more people, arrange in a line or arc
        spacing = 0.8 / total_people
        x_position = 0.1 + (person_index + 0.5) * spacing
        return (x_position, 0.5)

    def _generate_interaction_description(self, interaction: str) -> str:
        """Generate description for the interaction component."""
        if interaction in self.interaction_types:
            interaction_info = self.interaction_types[interaction]
            return f"{interaction_info['positioning']}, {interaction_info['body_language']}"

        return f"{interaction} interaction, natural body language"

    def generate_multi_person_prompt(self, components: List[Dict[str, Any]]) -> str:
        """Generate optimized prompt for multi-person scene."""
        prompt_parts = []

        # Add person descriptions
        person_descriptions = []
        for component in components:
            if component["type"] == "person":
                person_desc = f"{component['description']} in {component['pose']}"
                person_descriptions.append(person_desc)

        if len(person_descriptions) > 1:
            prompt_parts.append(f"{len(person_descriptions)} people: " + ", and ".join(person_descriptions))
        else:
            prompt_parts.extend(person_descriptions)

        # Add interaction descriptions
        for component in components:
            if component["type"] == "interaction":
                prompt_parts.append(component["description"])

        # Add multi-person specific enhancers
        prompt_parts.extend(
            [
                "clear separation between people",
                "distinct individuals",
                "proper spatial relationships",
                "natural interaction",
                "anatomically correct for all people",
                "well-composed group scene",
            ]
        )

        return ", ".join(prompt_parts)


def create_anatomy_correction_system():
    """Create complete anatomy correction system."""
    return {
        "anatomy_validator": AnatomyValidator(),
        "hand_corrector": HandCorrector(),
        "multi_person_composer": MultiPersonComposer(),
    }
