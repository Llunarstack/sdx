"""
Model architecture visualization and analysis utilities.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any
import json


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count parameters by module type."""
    param_counts = {}
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            module_type = type(module).__name__
            param_count = sum(p.numel() for p in module.parameters())
            
            if module_type not in param_counts:
                param_counts[module_type] = 0
            param_counts[module_type] += param_count
    
    return param_counts


def analyze_model_architecture(model: nn.Module) -> Dict[str, Any]:
    """Analyze model architecture and return detailed information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    param_counts = count_parameters(model)
    
    # Estimate model size in MB (assuming float32)
    model_size_mb = total_params * 4 / (1024 * 1024)
    
    # Get layer information
    layers_info = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules
            module_params = sum(p.numel() for p in module.parameters())
            if module_params > 0:
                layers_info.append({
                    "name": name,
                    "type": type(module).__name__,
                    "parameters": module_params,
                    "trainable": any(p.requires_grad for p in module.parameters())
                })
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": model_size_mb,
        "parameter_counts_by_type": param_counts,
        "layers_info": layers_info,
        "num_layers": len(layers_info)
    }


def print_model_summary(model: nn.Module, input_shape: Tuple[int, ...] = None):
    """Print a detailed model summary."""
    analysis = analyze_model_architecture(model)
    
    print("=" * 80)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 80)
    
    print(f"Total Parameters: {analysis['total_parameters']:,}")
    print(f"Trainable Parameters: {analysis['trainable_parameters']:,}")
    print(f"Model Size: {analysis['model_size_mb']:.2f} MB")
    print(f"Number of Layers: {analysis['num_layers']}")
    
    print("\nParameter Distribution by Module Type:")
    print("-" * 50)
    for module_type, count in sorted(analysis['parameter_counts_by_type'].items(), 
                                   key=lambda x: x[1], reverse=True):
        percentage = (count / analysis['total_parameters']) * 100
        print(f"{module_type:20s}: {count:>12,} ({percentage:5.1f}%)")
    
    if input_shape:
        print(f"\nInput Shape: {input_shape}")
        try:
            # Try to estimate memory usage
            batch_size = input_shape[0] if len(input_shape) > 0 else 1
            memory_per_sample = 1  # Rough estimate in MB
            total_memory = batch_size * memory_per_sample + analysis['model_size_mb']
            print(f"Estimated Memory Usage: {total_memory:.2f} MB")
        except Exception:
            pass
    
    print("=" * 80)


def save_model_graph(model: nn.Module, save_path: str, input_shape: Tuple[int, ...]):
    """Save model architecture graph (requires torchviz)."""
    try:
        from torchviz import make_dot
        
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        if next(model.parameters()).is_cuda:
            dummy_input = dummy_input.cuda()
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        # Create graph
        dot = make_dot(output, params=dict(model.named_parameters()))
        dot.render(save_path, format='png', cleanup=True)
        print(f"Model graph saved to {save_path}.png")
        
    except ImportError:
        print("torchviz not installed. Install with: pip install torchviz")
    except Exception as e:
        print(f"Error creating model graph: {e}")


def compare_models(model1: nn.Module, model2: nn.Module, 
                  name1: str = "Model 1", name2: str = "Model 2"):
    """Compare two models side by side."""
    analysis1 = analyze_model_architecture(model1)
    analysis2 = analyze_model_architecture(model2)
    
    print("=" * 100)
    print(f"MODEL COMPARISON: {name1} vs {name2}")
    print("=" * 100)
    
    print(f"{'Metric':<30} {'Model 1':<20} {'Model 2':<20} {'Difference':<20}")
    print("-" * 100)
    
    metrics = [
        ("Total Parameters", "total_parameters"),
        ("Trainable Parameters", "trainable_parameters"),
        ("Model Size (MB)", "model_size_mb"),
        ("Number of Layers", "num_layers")
    ]
    
    for metric_name, key in metrics:
        val1 = analysis1[key]
        val2 = analysis2[key]
        
        if isinstance(val1, float):
            diff = f"{val2 - val1:+.2f}"
            val1_str = f"{val1:.2f}"
            val2_str = f"{val2:.2f}"
        else:
            diff = f"{val2 - val1:+,}"
            val1_str = f"{val1:,}"
            val2_str = f"{val2:,}"
        
        print(f"{metric_name:<30} {val1_str:<20} {val2_str:<20} {diff:<20}")
    
    print("=" * 100)


def get_layer_wise_lr_groups(model: nn.Module, base_lr: float = 1e-4) -> List[Dict]:
    """Create layer-wise learning rate groups for fine-tuning."""
    param_groups = []
    
    # Group parameters by layer depth
    layer_groups = {}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Determine layer depth based on name
        depth = name.count('.')
        
        if depth not in layer_groups:
            layer_groups[depth] = []
        
        layer_groups[depth].append(param)
    
    # Create parameter groups with different learning rates
    for depth in sorted(layer_groups.keys()):
        # Lower learning rate for deeper layers (closer to output)
        lr_multiplier = 0.1 ** (max(0, depth - 2) / 10)  # Gradual decay
        group_lr = base_lr * lr_multiplier
        
        param_groups.append({
            'params': layer_groups[depth],
            'lr': group_lr,
            'layer_depth': depth
        })
    
    return param_groups


def find_unused_parameters(model: nn.Module, input_tensor: torch.Tensor) -> List[str]:
    """Find parameters that don't contribute to gradients."""
    model.train()
    
    # Forward pass
    output = model(input_tensor)
    
    # Backward pass
    if hasattr(output, 'loss'):
        loss = output.loss
    else:
        loss = output.mean()  # Dummy loss
    
    loss.backward()
    
    # Find parameters with None gradients
    unused_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            unused_params.append(name)
    
    # Clear gradients
    model.zero_grad()
    
    return unused_params


def export_model_info(model: nn.Module, save_path: str):
    """Export detailed model information to JSON."""
    analysis = analyze_model_architecture(model)
    
    # Convert to JSON-serializable format
    export_data = {
        "model_summary": {
            "total_parameters": analysis["total_parameters"],
            "trainable_parameters": analysis["trainable_parameters"],
            "model_size_mb": analysis["model_size_mb"],
            "num_layers": analysis["num_layers"]
        },
        "parameter_distribution": analysis["parameter_counts_by_type"],
        "layers": analysis["layers_info"]
    }
    
    with open(save_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Model information exported to {save_path}")


def estimate_training_memory(model: nn.Module, batch_size: int, 
                           sequence_length: int, use_gradient_checkpointing: bool = False) -> Dict[str, float]:
    """Estimate memory usage during training."""
    model_params = sum(p.numel() for p in model.parameters())
    
    # Model weights (float32)
    model_memory = model_params * 4 / (1024**3)  # GB
    
    # Gradients (same size as model)
    gradient_memory = model_memory
    
    # Optimizer states (AdamW: 2x model size)
    optimizer_memory = model_memory * 2
    
    # Activations (rough estimate)
    # This is very approximate and depends on model architecture
    activation_memory = batch_size * sequence_length * 1024 * 4 / (1024**3)  # GB
    
    if use_gradient_checkpointing:
        activation_memory *= 0.5  # Rough reduction
    
    total_memory = model_memory + gradient_memory + optimizer_memory + activation_memory
    
    return {
        "model_memory_gb": model_memory,
        "gradient_memory_gb": gradient_memory,
        "optimizer_memory_gb": optimizer_memory,
        "activation_memory_gb": activation_memory,
        "total_memory_gb": total_memory
    }