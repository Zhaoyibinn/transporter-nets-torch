#!/usr/bin/env python3
"""Downsample a mesh and export it as STL.

Example:
    python scripts/downsample_mesh.py \
        --input ravens_torch/environments/assets/insertion/fuse_post.ply \
        --output ravens_torch/environments/assets/insertion/fuse_post.stl \
        --target-face-count 2000

This script requires `trimesh` (and its dependency `pyglet`). Install via:
    pip install trimesh
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh

try:
    import open3d as o3d  # type: ignore
except ImportError:
    o3d = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Downsample a mesh and save as STL")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("ravens_torch/environments/assets/insertion/fuse_post.ply"),
        help="Path to the source mesh (PLY, OBJ, etc.).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ravens_torch/environments/assets/insertion/fuse_post.stl"),
        help="Target path for the simplified STL mesh.",
    )
    parser.add_argument(
        "--target-face-count",
        type=int,
        default=2000,
        help="Approximate number of faces to keep after simplification.",
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="If set, require the mesh to have at least the target number of faces before simplifying.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Uniform scaling factor applied to the mesh before export.",
    )
    return parser.parse_args()


def simplify_mesh_trimesh(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    if hasattr(mesh, "simplify_quadratic_decimation"):
        simplified = mesh.simplify_quadratic_decimation(target_faces)
        if isinstance(simplified, trimesh.Trimesh):
            return simplified
    raise AttributeError("simplify_quadratic_decimation not available in current trimesh build.")


def simplify_mesh_open3d(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    if o3d is None:
        raise ImportError("open3d is required for simplification but is not installed.")

    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces),
    )
    o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_faces)
    o3d_mesh.remove_degenerate_triangles()
    o3d_mesh.remove_duplicated_triangles()
    o3d_mesh.remove_non_manifold_edges()
    new_mesh = trimesh.Trimesh(
        vertices=np.asarray(o3d_mesh.vertices),
        faces=np.asarray(o3d_mesh.triangles),
        process=True,
    )
    return new_mesh


def simplify_mesh(mesh: trimesh.Trimesh, target_faces: int, exact: bool) -> trimesh.Trimesh:
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("Loaded geometry is not a single mesh.")

    if mesh.faces.shape[0] <= target_faces:
        if exact:
            raise ValueError(
                f"Mesh already has {mesh.faces.shape[0]} faces, which is not more than the target {target_faces}."
            )
        return mesh

    try:
        return simplify_mesh_trimesh(mesh, target_faces)
    except AttributeError:
        return simplify_mesh_open3d(mesh, target_faces)


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input mesh not found: {args.input}")

    mesh = trimesh.load(args.input, force='mesh')

    mesh_simplified = simplify_mesh(mesh, args.target_face_count, args.exact)

    if not np.isclose(args.scale, 1.0):
        mesh_simplified = mesh_simplified.copy()
        mesh_simplified.apply_scale(args.scale)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh_simplified.export(args.output)
    scale_suffix = f", scale {args.scale:g}x" if not np.isclose(args.scale, 1.0) else ""
    print(
        f"Saved simplified mesh to {args.output} with {mesh_simplified.faces.shape[0]} faces ("
        f"original {mesh.faces.shape[0]} faces){scale_suffix}."
    )


if __name__ == "__main__":
    main()
