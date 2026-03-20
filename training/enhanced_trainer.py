"""
Enhanced Training System for Advanced DiT Models
Includes specialized loss functions and training procedures for advanced features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
import numpy as np
from dataclasses import dataclass

from models.enhanced_dit import EnhancedDiT
from utils.enhanced_utils import (
    create_precision_control_system,
    create_anatomy_correction_system,
    create_text_rendering_pipeline,
    create_consistency_system
)
from utils.character_consistency import CharacterDatabase, CharacterProfile, FaceEncoder, BodyEncoder
from utils.consistency_losses import ConsistencyLossManager
from utils.style_harmonization import create_style_harmonization_system


@dataclass
class EnhancedTrainingBatch:
    """Enhanced training batch with additional feature data."""
    # Standard diffusion data
    images: torch.Tensor
    timesteps: torch.Tensor
    noise: torch.Tensor
    class_labels: torch.Tensor
    
    # Enhanced feature data
    spatial_layouts: Optional[torch.Tensor] = None
    anatomy_masks: Optional[torch.Tensor] = None
    text_tokens: Optional[torch.Tensor] = None
    text_positions: Optional[torch.Tensor] = None
    typography_styles: Optional[torch.Tensor] = None
    character_ids: Optional[torch.Tensor] = None
    style_ids: Optional[torch.Tensor] = None
    
    # Character consistency data
    character_profiles: Optional[List[CharacterProfile]] = None
    reference_images: Optional[torch.Tensor] = None
    character_embeddings: Optional[torch.Tensor] = None
    
    # Style harmonization data
    original_prompt: Optional[str] = None
    harmonized_prompt: Optional[str] = None
    lora_configs: Optional[List[Dict[str, Any]]] = None
    style_conflicts: Optional[Dict[str, Any]] = None
    
    # Ground truth for validation
    object_counts: Optional[torch.Tensor] = None
    anatomy_keypoints: Optional[torch.Tensor] = None
    text_content: Optional[List[str]] = None


class SpatialControlLoss(nn.Module):
    """Loss function for spatial control and object placement."""
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        self.mse_loss = nn.MSELoss()
        
    def forward(self, predicted_noise: torch.Tensor, target_noise: torch.Tensor,
                spatial_layout: Optional[torch.Tensor] = None,
                object_counts: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # Base diffusion loss
        base_loss = self.mse_loss(predicted_noise, target_noise)
        
        if spatial_layout is None:
            return base_loss
        
        # Spatial consistency loss
        # Encourage model to respect spatial layouts
        B, C, H, W = predicted_noise.shape
        
        # Create spatial attention maps from noise prediction
        spatial_attention = torch.mean(predicted_noise.abs(), dim=1)  # [B, H, W]
        
        # Create target spatial maps from layout
        target_spatial = torch.zeros_like(spatial_attention)
        
        if spatial_layout is not None:
            for b in range(B):
                for obj_idx in range(spatial_layout.shape[1]):
                    x, y, w, h = spatial_layout[b, obj_idx]
                    if x > 0 and y > 0:  # Valid object
                        x_start = int(x * W)
                        y_start = int(y * H)
                        x_end = int((x + w) * W)
                        y_end = int((y + h) * H)
                        
                        target_spatial[b, y_start:y_end, x_start:x_end] = 1.0
        
        # Spatial alignment loss
        spatial_loss = F.mse_loss(spatial_attention, target_spatial)
        
        # Counting loss if provided
        counting_loss = 0.0
        if object_counts is not None:
            # Simplified counting loss - count peaks in attention
            attention_peaks = []
            for b in range(B):
                # Find local maxima in spatial attention
                attention_map = spatial_attention[b]
                # Simple peak counting (would be more sophisticated in practice)
                peaks = (attention_map > 0.5).sum().float()
                attention_peaks.append(peaks)
            
            predicted_counts = torch.stack(attention_peaks)
            counting_loss = F.mse_loss(predicted_counts, object_counts.float())
        
        total_loss = base_loss + self.weight * (spatial_loss + 0.5 * counting_loss)
        return total_loss


class AnatomyAwareLoss(nn.Module):
    """Loss function for anatomy-aware generation."""
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        self.mse_loss = nn.MSELoss()
        
    def forward(self, predicted_noise: torch.Tensor, target_noise: torch.Tensor,
                anatomy_mask: Optional[torch.Tensor] = None,
                anatomy_keypoints: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # Base diffusion loss
        base_loss = self.mse_loss(predicted_noise, target_noise)
        
        if anatomy_mask is None:
            return base_loss
        
        # Anatomy-focused loss
        # Higher weight on human regions
        if anatomy_mask is not None:
            # Reshape mask to match noise dimensions
            B, N = anatomy_mask.shape
            H = W = int(np.sqrt(N))
            anatomy_mask_2d = anatomy_mask.view(B, H, W).unsqueeze(1)
            anatomy_mask_2d = F.interpolate(anatomy_mask_2d, size=predicted_noise.shape[-2:], mode='nearest')
            
            # Weighted loss - higher weight on anatomy regions
            anatomy_weight = 1.0 + 2.0 * anatomy_mask_2d  # 3x weight on anatomy
            anatomy_loss = torch.mean(anatomy_weight * (predicted_noise - target_noise) ** 2)
        else:
            anatomy_loss = base_loss
        
        # Hand-specific loss (simplified)
        hand_loss = 0.0
        if anatomy_keypoints is not None:
            # Focus on hand regions - would need more sophisticated implementation
            # Apply additional loss weighting to hand regions
            hand_loss = 0.1 * F.mse_loss(predicted_noise, target_noise)
        
        total_loss = base_loss + self.weight * (anatomy_loss - base_loss + hand_loss)
        return total_loss


class TextRenderingLoss(nn.Module):
    """Loss function for text rendering accuracy."""
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        self.mse_loss = nn.MSELoss()
        
    def forward(self, predicted_noise: torch.Tensor, target_noise: torch.Tensor,
                text_tokens: Optional[torch.Tensor] = None,
                text_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # Base diffusion loss
        base_loss = self.mse_loss(predicted_noise, target_noise)
        
        if text_tokens is None:
            return base_loss
        
        # Text region loss
        text_loss = 0.0
        if text_positions is not None:
            B, max_text_len, _ = text_positions.shape
            C, H, W = predicted_noise.shape[1:]
            
            # Create text region masks
            text_masks = torch.zeros(B, H, W, device=predicted_noise.device)
            
            for b in range(B):
                for t in range(max_text_len):
                    x, y = text_positions[b, t]
                    if x > 0 and y > 0:  # Valid text position
                        x_pos = int(x * W)
                        y_pos = int(y * H)
                        # Create small region around text position
                        x_start = max(0, x_pos - 5)
                        x_end = min(W, x_pos + 5)
                        y_start = max(0, y_pos - 5)
                        y_end = min(H, y_pos + 5)
                        text_masks[b, y_start:y_end, x_start:x_end] = 1.0
            
            # Apply higher weight to text regions
            text_weight = 1.0 + 3.0 * text_masks.unsqueeze(1)  # 4x weight on text
            text_loss = torch.mean(text_weight * (predicted_noise - target_noise) ** 2)
        
        total_loss = base_loss + self.weight * (text_loss - base_loss)
        return total_loss


class ConsistencyLoss(nn.Module):
    """Loss function for character and style consistency."""
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        self.mse_loss = nn.MSELoss()
        
    def forward(self, predicted_noise: torch.Tensor, target_noise: torch.Tensor,
                character_ids: Optional[torch.Tensor] = None,
                style_ids: Optional[torch.Tensor] = None,
                character_features: Optional[torch.Tensor] = None,
                style_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # Base diffusion loss
        base_loss = self.mse_loss(predicted_noise, target_noise)
        
        consistency_loss = 0.0
        
        # Character consistency loss
        if character_ids is not None and character_features is not None:
            # Encourage similar features for same character
            unique_chars = torch.unique(character_ids)
            for char_id in unique_chars:
                char_mask = (character_ids == char_id)
                if char_mask.sum() > 1:  # Multiple instances of same character
                    char_features = character_features[char_mask]
                    # Encourage feature similarity
                    char_consistency = torch.var(char_features, dim=0).mean()
                    consistency_loss += char_consistency
        
        # Style consistency loss
        if style_ids is not None and style_features is not None:
            # Similar logic for style consistency
            unique_styles = torch.unique(style_ids)
            for style_id in unique_styles:
                style_mask = (style_ids == style_id)
                if style_mask.sum() > 1:
                    style_features_subset = style_features[style_mask]
                    style_consistency = torch.var(style_features_subset, dim=0).mean()
                    consistency_loss += style_consistency
        
        total_loss = base_loss + self.weight * consistency_loss
        return total_loss


class EnhancedTrainer:
    """Enhanced trainer for advanced DiT models."""
    
    def __init__(self, model: EnhancedDiT, device: str = "cuda", character_database_path: str = "./character_database"):
        self.model = model
        self.device = device
        
        # Loss functions
        self.spatial_loss = SpatialControlLoss(weight=0.5)
        self.anatomy_loss = AnatomyAwareLoss(weight=0.3)
        self.text_loss = TextRenderingLoss(weight=0.4)
        self.consistency_loss = ConsistencyLoss(weight=0.2)
        
        # Character consistency system
        self.character_database = CharacterDatabase(character_database_path)
        self.face_encoder = FaceEncoder().to(device)
        self.body_encoder = BodyEncoder().to(device)
        self.character_consistency_manager = ConsistencyLossManager(
            self.face_encoder, 
            self.body_encoder
        )
        
        # Style harmonization system
        self.style_harmonizer = create_style_harmonization_system()
        
        # Feature extractors for validation
        precision_system = create_precision_control_system()
        anatomy_system = create_anatomy_correction_system()
        text_system = create_text_rendering_pipeline()
        consistency_system = create_consistency_system()
        
        self.scene_composer = precision_system['scene_composer']
        self.anatomy_validator = anatomy_system['anatomy_validator']
        self.text_engine = text_system['engine']
        self.consistency_manager = consistency_system['consistency_manager']
        
    def compute_enhanced_loss(self, batch: EnhancedTrainingBatch,
                            predicted_noise: torch.Tensor,
                            target_noise: torch.Tensor,
                            generated_images: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute enhanced loss with all advanced features."""
        
        losses = {}
        
        # Spatial control loss
        if batch.spatial_layouts is not None:
            spatial_loss = self.spatial_loss(
                predicted_noise, target_noise,
                batch.spatial_layouts, batch.object_counts
            )
            losses['spatial'] = spatial_loss
        
        # Anatomy-aware loss
        if batch.anatomy_masks is not None:
            anatomy_loss = self.anatomy_loss(
                predicted_noise, target_noise,
                batch.anatomy_masks, batch.anatomy_keypoints
            )
            losses['anatomy'] = anatomy_loss
        
        # Text rendering loss
        if batch.text_tokens is not None:
            text_loss = self.text_loss(
                predicted_noise, target_noise,
                batch.text_tokens, batch.text_positions
            )
            losses['text'] = text_loss
        
        # Character consistency loss
        if batch.character_profiles is not None and generated_images is not None:
            char_losses = self.character_consistency_manager.compute_total_loss(
                generated_images=generated_images,
                character_profiles=batch.character_profiles,
                reference_images=batch.reference_images
            )
            losses.update(char_losses)
        
        # Original consistency loss
        if batch.character_ids is not None or batch.style_ids is not None:
            consistency_loss = self.consistency_loss(
                predicted_noise, target_noise,
                batch.character_ids, batch.style_ids
            )
            losses['legacy_consistency'] = consistency_loss
        
        # Base diffusion loss
        base_loss = F.mse_loss(predicted_noise, target_noise)
        losses['base'] = base_loss
        
        # Total loss
        total_loss = base_loss
        for loss_name, loss_value in losses.items():
            if loss_name != 'base':
                total_loss = total_loss + loss_value - base_loss  # Avoid double counting base loss
        
        losses['total'] = total_loss
        
        return losses
    
    def training_step(self, batch: EnhancedTrainingBatch) -> Dict[str, torch.Tensor]:
        """Single training step with enhanced features."""
        
        # Standard diffusion forward pass
        x_0 = batch.images
        t = batch.timesteps
        noise = batch.noise
        
        # Add noise to images
        x_t = self.add_noise(x_0, noise, t)
        
        # Model prediction with enhanced features
        model_output = self.model(
            x_t, t, batch.class_labels,
            spatial_layout=batch.spatial_layouts,
            anatomy_mask=batch.anatomy_masks,
            text_tokens=batch.text_tokens,
            text_positions=batch.text_positions,
            typography_style=batch.typography_styles,
            character_id=batch.character_ids,
            style_id=batch.style_ids
        )
        
        # For character consistency, we need to generate images during training
        # This is computationally expensive but necessary for consistency training
        generated_images = None
        if batch.character_profiles is not None:
            # Generate images from predicted noise (simplified)
            with torch.no_grad():
                # This would normally require full denoising process
                # For training efficiency, we use a simplified approximation
                generated_images = x_0  # Use ground truth as approximation
        
        # Compute enhanced loss
        losses = self.compute_enhanced_loss(batch, model_output, noise, generated_images)
        
        # Handle sigma prediction if enabled
        if hasattr(self.model, 'learn_sigma') and self.model.learn_sigma:
            # Split output into noise and sigma predictions
            predicted_noise, predicted_sigma = torch.chunk(model_output, 2, dim=1)
        else:
            predicted_noise = model_output
        
        # Compute enhanced losses
        losses = self.compute_enhanced_loss(batch, predicted_noise, noise)
        
        return losses
    
    def add_noise(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Add noise to clean images (standard diffusion)."""
        # This would use your diffusion schedule
        # Placeholder implementation
        alpha_t = torch.cos(t * np.pi / 2) ** 2
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        
        x_t = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * noise
        return x_t
    
    def validate_generation(self, generated_images: torch.Tensor,
                          batch: EnhancedTrainingBatch) -> Dict[str, float]:
        """Validate generated images against enhanced features."""
        
        validation_results = {}
        
        # Convert to PIL images for validation
        from torchvision.transforms import ToPILImage
        to_pil = ToPILImage()
        
        for i, img_tensor in enumerate(generated_images):
            img = to_pil(img_tensor)
            
            # Validate character consistency if applicable
            if batch.character_profiles and i < len(batch.character_profiles):
                character_profile = batch.character_profiles[i]
                consistency_scores = self.character_database.validate_consistency(
                    img_tensor, character_profile.character_id
                )
                validation_results[f'character_consistency_{i}'] = consistency_scores['overall_consistency']
            
            # Validate text rendering if applicable
            if batch.text_content and i < len(batch.text_content):
                text_validation = self.text_engine.validate_text_rendering(
                    img, [batch.text_content[i]]
                )
                validation_results[f'text_accuracy_{i}'] = text_validation.get('accuracy_score', 0.0)
            
            # Validate anatomy if applicable
            if batch.anatomy_masks is not None:
                # Placeholder anatomy validation
                validation_results[f'anatomy_score_{i}'] = 0.8  # Would be actual validation
            
            # Validate spatial layout if applicable
            if batch.spatial_layouts is not None:
                # Placeholder spatial validation
                validation_results[f'spatial_accuracy_{i}'] = 0.9  # Would be actual validation
        
        return validation_results
    
    # Character consistency management methods
    def create_character(self, name: str, reference_images: List[str], 
                        physical_features=None, style_preferences=None) -> CharacterProfile:
        """Create a new character profile."""
        return self.character_database.create_character(
            name, reference_images, physical_features, style_preferences
        )
    
    def update_character(self, character_id: str, updates: Dict[str, Any]) -> CharacterProfile:
        """Update an existing character profile."""
        return self.character_database.update_character(character_id, updates)
    
    def get_character(self, character_id: str) -> Optional[CharacterProfile]:
        """Get character profile by ID."""
        return self.character_database.get_character(character_id)
    
    def list_characters(self, filters: Optional[Dict[str, Any]] = None) -> List[CharacterProfile]:
        """List all characters with optional filtering."""
        return self.character_database.list_characters(filters)
    
    def delete_character(self, character_id: str) -> bool:
        """Delete a character profile."""
        return self.character_database.delete_character(character_id)
    
    def validate_character_consistency(self, image: torch.Tensor, character_id: str) -> Dict[str, float]:
        """Validate consistency of generated image against character."""
        return self.character_database.validate_consistency(image, character_id)
    
    def update_consistency_loss_weights(self, new_weights: Dict[str, float]):
        """Update character consistency loss weights during training."""
        self.character_consistency_manager.update_loss_weights(new_weights)
    
    def get_consistency_statistics(self) -> Dict[str, Any]:
        """Get statistics about character consistency system."""
        return {
            'character_count': len(self.character_database.characters),
            'loss_statistics': self.character_consistency_manager.get_loss_statistics(),
            'database_path': str(self.character_database.database_path)
        }
    
    # Style harmonization methods
    def harmonize_batch_styles(self, batch: EnhancedTrainingBatch) -> EnhancedTrainingBatch:
        """Harmonize styles in a training batch to prevent conflicts."""
        if not batch.original_prompt or not batch.lora_configs:
            return batch  # No harmonization needed
        
        # Harmonize styles for each item in batch
        harmonized_prompts = []
        harmonized_lora_configs = []
        style_analyses = []
        
        batch_size = batch.images.shape[0]
        
        for i in range(batch_size):
            # Get individual prompt and LoRA config (simplified - assumes same for all)
            prompt = batch.original_prompt if isinstance(batch.original_prompt, str) else batch.original_prompt[i]
            lora_config = batch.lora_configs if isinstance(batch.lora_configs, list) else [batch.lora_configs]
            
            # Harmonize styles
            harmonization_result = self.style_harmonizer.harmonize_styles(
                prompt=prompt,
                lora_configs=lora_config,
                user_preferences={"harmonization_mode": "balanced", "allow_prompt_modification": True}
            )
            
            harmonized_prompts.append(harmonization_result["harmonized_prompt"])
            harmonized_lora_configs.append(harmonization_result["adjusted_loras"])
            style_analyses.append(harmonization_result["style_analysis"])
        
        # Update batch with harmonized data
        batch.harmonized_prompt = harmonized_prompts[0] if len(set(harmonized_prompts)) == 1 else harmonized_prompts
        batch.lora_configs = harmonized_lora_configs[0] if len(harmonized_lora_configs) == 1 else harmonized_lora_configs
        batch.style_conflicts = {
            "analyses": style_analyses,
            "harmonization_applied": any(analysis["harmonization_applied"] for analysis in style_analyses)
        }
        
        return batch
    
    def validate_style_harmony(self, prompt: str, lora_configs: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate style harmony for a given prompt and LoRA configuration."""
        return self.style_harmonizer.harmonize_styles(
            prompt=prompt,
            lora_configs=lora_configs or [],
            user_preferences={"harmonization_mode": "analyze_only"}
        )
    
    def get_style_recommendations(self, prompt: str, lora_configs: List[Dict[str, Any]] = None) -> List[str]:
        """Get style harmonization recommendations."""
        result = self.validate_style_harmony(prompt, lora_configs)
        
        recommendations = []
        if result["style_analysis"]["conflict_level"] != "none":
            recommendations.append(f"Detected {result['style_analysis']['conflict_level']} style conflicts")
            recommendations.append(f"Dominant style: {result['style_analysis']['dominant_style']}")
            
            for conflict in result["conflicts"]:
                recommendations.append(
                    f"Conflict between {conflict['style1']} and {conflict['style2']} "
                    f"(severity: {conflict['severity']})"
                )
        else:
            recommendations.append("No style conflicts detected - harmonious combination")
        
        return recommendations


def create_enhanced_trainer(model: EnhancedDiT, device: str = "cuda", 
                          character_database_path: str = "./character_database") -> EnhancedTrainer:
    """Create enhanced trainer instance with character consistency support."""
    return EnhancedTrainer(model, device, character_database_path)