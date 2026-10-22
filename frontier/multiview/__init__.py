"""SDX-Spatial: the 3D core. A tri-plane latent field that decodes to one consistent object from any camera.

``triplane`` is the field itself; ``training`` fits it to an SDF, ``neus_render`` renders and fits it from views,
``mesh_export`` / ``validate`` turn it into a checked mesh, ``radiance_sh`` adds view-dependent colour,
``rigging`` skeletons and skins the mesh, ``asset_graph`` splits it into semantic parts, ``crossbreed`` mixes
conditionings, ``bridge`` lifts 2D DiT features into the field, and ``views`` renders turntables / contact sheets.
"""

from __future__ import annotations

from .asset_graph import AABB, AssetGraph, SemanticPart
from .bridge import BridgeConfig, Lift2Dto3D
from .crossbreed import blend_conditioning, mutate_conditioning, slerp, spatial_graft
from .mesh_export import extract_mesh, write_obj
from .neus_render import NeusRenderer, View, fit_from_views, look_at_rays, volume_render_sdf
from .radiance_sh import SHRadianceDecoder, eval_sh_basis, sh_num_coeffs
from .rigging import (
    auto_skeleton_from_points,
    detokenize_skeleton,
    forward_kinematics,
    linear_blend_skinning,
    rig_mesh,
    skinning_matrices,
    skinning_weights,
    tokenize_skeleton,
)
from .training import eikonal_loss, fit_sdf, sdf_and_gradient, sdf_regression_loss, sphere_sdf, surface_normal
from .triplane import SpatialLatentField, TriPlaneConfig, build_spatial_field
from .validate import validate_mesh
from .views import contact_sheet, perfect_angles, viewpoint_rays

__all__ = [
    "AABB",
    "AssetGraph",
    "BridgeConfig",
    "Lift2Dto3D",
    "NeusRenderer",
    "SHRadianceDecoder",
    "SemanticPart",
    "SpatialLatentField",
    "TriPlaneConfig",
    "View",
    "auto_skeleton_from_points",
    "blend_conditioning",
    "build_spatial_field",
    "contact_sheet",
    "detokenize_skeleton",
    "eikonal_loss",
    "eval_sh_basis",
    "extract_mesh",
    "fit_from_views",
    "fit_sdf",
    "forward_kinematics",
    "linear_blend_skinning",
    "look_at_rays",
    "mutate_conditioning",
    "perfect_angles",
    "rig_mesh",
    "sdf_and_gradient",
    "sdf_regression_loss",
    "sh_num_coeffs",
    "skinning_matrices",
    "skinning_weights",
    "slerp",
    "spatial_graft",
    "sphere_sdf",
    "surface_normal",
    "tokenize_skeleton",
    "validate_mesh",
    "viewpoint_rays",
    "volume_render_sdf",
    "write_obj",
]
