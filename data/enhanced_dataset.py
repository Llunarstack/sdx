"""
Enhanced Dataset for Advanced DiT Training
Processes images with spatial layouts, anatomy annotations, text content, and consistency data.
"""
import json
import torch
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Any
import cv2
import re
from pathlib import Path

from training.enhanced_trainer import EnhancedTrainingBatch
from utils.enhanced_utils import (
    SimpleTextEngine,
    SimpleAnatomyValidator,
    SimpleSceneComposer
)


class EnhancedT2IDataset(Dataset):
    """Enhanced Text-to-Image dataset with advanced feature annotations."""
    
    def __init__(self, 
                 data_path: str,
                 manifest_path: Optional[str] = None,
                 image_size: int = 512,
                 enable_spatial_control: bool = True,
                 enable_anatomy_awareness: bool = True,
                 enable_text_rendering: bool = True,
                 enable_consistency: bool = True,
                 max_objects: int = 10,
                 max_text_length: int = 50):
        
        self.data_path = Path(data_path)
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.image_size = image_size
        
        # Feature flags
        self.enable_spatial_control = enable_spatial_control
        self.enable_anatomy_awareness = enable_anatomy_awareness
        self.enable_text_rendering = enable_text_rendering
        self.enable_consistency = enable_consistency
        
        # Limits
        self.max_objects = max_objects
        self.max_text_length = max_text_length
        
        # Load data
        self.data_items = self._load_data()
        
        # Initialize feature processors
        self.text_engine = SimpleTextEngine()
        self.anatomy_validator = SimpleAnatomyValidator()
        self.scene_composer = SimpleSceneComposer()
        
        # Character and style mappings
        self.character_to_id = {}
        self.style_to_id = {}
        self._build_consistency_mappings()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load dataset items from manifest or directory structure."""
        data_items = []
        
        if self.manifest_path and self.manifest_path.exists():
            # Load from JSONL manifest
            with open(self.manifest_path, 'r') as f:
                for line in f:
                    item = json.loads(line.strip())
                    data_items.append(item)
        else:
            # Scan directory structure
            for img_path in self.data_path.rglob("*.jpg"):
                # Look for corresponding annotation files
                base_name = img_path.stem
                
                # Caption file
                caption_file = img_path.with_suffix('.txt')
                caption = ""
                if caption_file.exists():
                    with open(caption_file, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                
                # Enhanced annotation file
                annotation_file = img_path.with_suffix('.json')
                annotations = {}
                if annotation_file.exists():
                    with open(annotation_file, 'r') as f:
                        annotations = json.load(f)
                
                item = {
                    "image_path": str(img_path),
                    "caption": caption,
                    "annotations": annotations
                }
                data_items.append(item)
        
        return data_items
    
    def _build_consistency_mappings(self):
        """Build character and style ID mappings."""
        characters = set()
        styles = set()
        
        for item in self.data_items:
            annotations = item.get("annotations", {})
            
            # Extract character names
            character = annotations.get("character_name")
            if character:
                characters.add(character)
            
            # Extract style names
            style = annotations.get("style_name")
            if style:
                styles.add(style)
        
        # Create ID mappings
        self.character_to_id = {char: i for i, char in enumerate(sorted(characters))}
        self.style_to_id = {style: i for i, style in enumerate(sorted(styles))}
    
    def __len__(self) -> int:
        return len(self.data_items)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get enhanced training item."""
        item = self.data_items[idx]
        
        # Load image
        image = Image.open(item["image_path"]).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # Get caption and annotations
        caption = item.get("caption", "")
        annotations = item.get("annotations", {})
        
        # Prepare enhanced features
        enhanced_data = {
            "image": image_tensor,
            "caption": caption,
            "class_label": 0,  # Would be actual class if using class conditioning
        }
        
        # Process spatial control data
        if self.enable_spatial_control:
            spatial_data = self._process_spatial_control(caption, annotations, image)
            enhanced_data.update(spatial_data)
        
        # Process anatomy data
        if self.enable_anatomy_awareness:
            anatomy_data = self._process_anatomy_awareness(caption, annotations, image)
            enhanced_data.update(anatomy_data)
        
        # Process text rendering data
        if self.enable_text_rendering:
            text_data = self._process_text_rendering(caption, annotations)
            enhanced_data.update(text_data)
        
        # Process consistency data
        if self.enable_consistency:
            consistency_data = self._process_consistency(annotations)
            enhanced_data.update(consistency_data)
        
        return enhanced_data
    
    def _process_spatial_control(self, caption: str, annotations: Dict[str, Any], 
                                image: Image.Image) -> Dict[str, Any]:
        """Process spatial control and object layout data."""
        spatial_data = {}
        
        # Extract object layout from annotations or generate from caption
        object_layout = annotations.get("object_layout")
        
        if object_layout is None:
            # Generate layout from caption using scene composer
            objects = self._extract_objects_from_caption(caption)
            if objects:
                scene_objects = self.scene_composer.create_scene_layout(
                    objects[:self.max_objects],
                    layout_type="balanced"
                )
                
                # Convert to tensor format
                layout_tensor = torch.zeros(self.max_objects, 4)
                for i, obj in enumerate(scene_objects):
                    if i < self.max_objects:
                        layout_tensor[i] = torch.tensor([
                            obj.position[0], obj.position[1],
                            obj.size[0], obj.size[1]
                        ])
                
                spatial_data["spatial_layout"] = layout_tensor
                spatial_data["object_count"] = len(objects)
        else:
            # Use provided layout
            layout_tensor = torch.tensor(object_layout, dtype=torch.float32)
            if layout_tensor.shape[0] < self.max_objects:
                # Pad with zeros
                padding = torch.zeros(self.max_objects - layout_tensor.shape[0], 4)
                layout_tensor = torch.cat([layout_tensor, padding], dim=0)
            
            spatial_data["spatial_layout"] = layout_tensor[:self.max_objects]
            spatial_data["object_count"] = annotations.get("object_count", 0)
        
        return spatial_data
    
    def _process_anatomy_awareness(self, caption: str, annotations: Dict[str, Any],
                                 image: Image.Image) -> Dict[str, Any]:
        """Process anatomy awareness data."""
        anatomy_data = {}
        
        # Check if image contains humans
        has_human = self._detect_human_in_caption(caption)
        
        if has_human:
            # Generate anatomy mask (simplified - would use actual pose detection)
            anatomy_mask = self._generate_anatomy_mask(image)
            anatomy_data["anatomy_mask"] = anatomy_mask
            
            # Extract keypoints if available
            keypoints = annotations.get("anatomy_keypoints")
            if keypoints:
                anatomy_data["anatomy_keypoints"] = torch.tensor(keypoints, dtype=torch.float32)
        
        return anatomy_data
    
    def _process_text_rendering(self, caption: str, annotations: Dict[str, Any]) -> Dict[str, Any]:
        """Process text rendering data."""
        text_data = {}
        
        # Extract text requirements from caption
        text_info = self.text_engine.extract_text_requirements(caption)
        
        if text_info["has_text"] or annotations.get("has_text", False):
            # Get text content
            text_content = text_info.get("text_content", [])
            if not text_content:
                text_content = annotations.get("text_content", [])
            
            if text_content:
                # Tokenize text (simplified - would use actual tokenizer)
                text_tokens = self._tokenize_text(text_content)
                text_data["text_tokens"] = text_tokens
                
                # Get text positions
                text_positions = annotations.get("text_positions")
                if text_positions:
                    text_data["text_positions"] = torch.tensor(text_positions, dtype=torch.float32)
                else:
                    # Generate default positions
                    text_data["text_positions"] = self._generate_text_positions(len(text_content))
                
                # Get typography style
                typography_style = annotations.get("typography_style", 0)
                text_data["typography_style"] = torch.tensor(typography_style, dtype=torch.long)
                
                text_data["text_content"] = text_content
        
        return text_data
    
    def _process_consistency(self, annotations: Dict[str, Any]) -> Dict[str, Any]:
        """Process character and style consistency data."""
        consistency_data = {}
        
        # Character ID
        character_name = annotations.get("character_name")
        if character_name and character_name in self.character_to_id:
            consistency_data["character_id"] = torch.tensor(
                self.character_to_id[character_name], dtype=torch.long
            )
        
        # Style ID
        style_name = annotations.get("style_name")
        if style_name and style_name in self.style_to_id:
            consistency_data["style_id"] = torch.tensor(
                self.style_to_id[style_name], dtype=torch.long
            )
        
        return consistency_data
    
    def _extract_objects_from_caption(self, caption: str) -> List[str]:
        """Extract object names from caption."""
        # Simple object extraction - would be more sophisticated
        common_objects = [
            "person", "woman", "man", "girl", "boy", "child",
            "car", "house", "tree", "flower", "chair", "table",
            "cat", "dog", "bird", "book", "cup", "bottle"
        ]
        
        objects = []
        caption_lower = caption.lower()
        
        for obj in common_objects:
            if obj in caption_lower:
                objects.append(obj)
        
        return objects
    
    def _detect_human_in_caption(self, caption: str) -> bool:
        """Detect if caption describes humans."""
        human_keywords = [
            "person", "woman", "man", "girl", "boy", "child", "people",
            "human", "figure", "character", "portrait"
        ]
        
        caption_lower = caption.lower()
        return any(keyword in caption_lower for keyword in human_keywords)
    
    def _generate_anatomy_mask(self, image: Image.Image) -> torch.Tensor:
        """Generate anatomy mask for image (simplified)."""
        # This would use actual pose detection/segmentation
        # For now, create a simple mask
        mask = torch.zeros(self.image_size // 8, self.image_size // 8)  # Downsampled
        
        # Create simple human-like region in center
        h, w = mask.shape
        center_h, center_w = h // 2, w // 2
        mask[center_h-10:center_h+10, center_w-5:center_w+5] = 1.0
        
        return mask.flatten()  # Flatten for attention
    
    def _tokenize_text(self, text_content: List[str]) -> torch.Tensor:
        """Tokenize text content (simplified)."""
        # This would use actual tokenizer
        # For now, create dummy tokens
        max_tokens = self.max_text_length
        tokens = torch.zeros(max_tokens, dtype=torch.long)
        
        # Simple character-based tokenization
        text_combined = " ".join(text_content)[:max_tokens]
        for i, char in enumerate(text_combined):
            if i < max_tokens:
                tokens[i] = ord(char) % 1000  # Simple mapping
        
        return tokens
    
    def _generate_text_positions(self, num_texts: int) -> torch.Tensor:
        """Generate default text positions."""
        positions = torch.zeros(self.max_text_length, 2)
        
        # Place texts in center region
        for i in range(min(num_texts, self.max_text_length)):
            x = 0.3 + 0.4 * (i / max(1, num_texts - 1))  # Spread horizontally
            y = 0.5  # Center vertically
            positions[i] = torch.tensor([x, y])
        
        return positions


def collate_enhanced_batch(batch: List[Dict[str, Any]]) -> EnhancedTrainingBatch:
    """Collate function for enhanced training batches."""
    
    # Standard data
    images = torch.stack([item["image"] for item in batch])
    class_labels = torch.tensor([item["class_label"] for item in batch], dtype=torch.long)
    
    # Generate timesteps and noise for diffusion
    batch_size = len(batch)
    timesteps = torch.randint(0, 1000, (batch_size,), dtype=torch.long)
    noise = torch.randn_like(images)
    
    # Enhanced features (optional)
    spatial_layouts = None
    if "spatial_layout" in batch[0]:
        spatial_layouts = torch.stack([item["spatial_layout"] for item in batch])
    
    anatomy_masks = None
    if "anatomy_mask" in batch[0]:
        anatomy_masks = torch.stack([item["anatomy_mask"] for item in batch])
    
    text_tokens = None
    if "text_tokens" in batch[0]:
        text_tokens = torch.stack([item["text_tokens"] for item in batch])
    
    text_positions = None
    if "text_positions" in batch[0]:
        text_positions = torch.stack([item["text_positions"] for item in batch])
    
    typography_styles = None
    if "typography_style" in batch[0]:
        typography_styles = torch.stack([item["typography_style"] for item in batch])
    
    character_ids = None
    if "character_id" in batch[0]:
        character_ids = torch.stack([item["character_id"] for item in batch])
    
    style_ids = None
    if "style_id" in batch[0]:
        style_ids = torch.stack([item["style_id"] for item in batch])
    
    # Ground truth data
    object_counts = None
    if "object_count" in batch[0]:
        object_counts = torch.tensor([item["object_count"] for item in batch], dtype=torch.long)
    
    anatomy_keypoints = None
    if "anatomy_keypoints" in batch[0]:
        anatomy_keypoints = torch.stack([item["anatomy_keypoints"] for item in batch])
    
    text_content = None
    if "text_content" in batch[0]:
        text_content = [item["text_content"] for item in batch]
    
    return EnhancedTrainingBatch(
        images=images,
        timesteps=timesteps,
        noise=noise,
        class_labels=class_labels,
        spatial_layouts=spatial_layouts,
        anatomy_masks=anatomy_masks,
        text_tokens=text_tokens,
        text_positions=text_positions,
        typography_styles=typography_styles,
        character_ids=character_ids,
        style_ids=style_ids,
        object_counts=object_counts,
        anatomy_keypoints=anatomy_keypoints,
        text_content=text_content
    )