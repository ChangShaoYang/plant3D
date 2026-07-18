"""
Input:
    data/plant_3D.glb

Output:
    results/plant_with_nodes.glb

Run:
    python plant_anal

Use from another Python file:
    from plant_anal import analyze_plant_model

    result = analyze_plant_model()

    # Use these two values in your workflow:
    height = result["height"]
    node_count = result["node_count"]
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import DBSCAN


BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = Path("data/plant_3D.glb")
OUTPUT_GLB = Path("results/plant_with_nodes.glb")


@dataclass
class AnalysisConfig:
    up_axis: str = "y"
    sample_count: int = 80000
    seed: int = 42

    lower_percentile: float = 1.0
    upper_percentile: float = 99.0

    num_slices: int = 120
    stem_center_slices: int = 160

    vertical_window_ratio: float = 0.014
    dbscan_eps_ratio: float = 0.022
    dbscan_min_samples: int = 12

    stem_core_radius_ratio: float = 0.025
    near_stem_radius_ratio: float = 0.045
    min_outward_radius_ratio: float = 0.10
    min_branch_length_ratio: float = 0.05

    min_linearity: float = 0.60
    max_planarity: float = 0.50
    merge_distance_ratio: float = 0.04

    marker_radius_ratio: float = 0.012


# Convert a relative path to an absolute project path.
def resolve_project_path(path: Union[str, Path]) -> Path:
    path = Path(path)

    if path.is_absolute():
        return path

    return BASE_DIR / path


# Convert a path to a readable relative path.
def to_display_path(path: Union[str, Path]) -> str:
    path = Path(path).resolve()

    try:
        return str(path.relative_to(BASE_DIR)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# Load the input GLB file as one mesh.
def load_glb_as_single_mesh(input_path: Union[str, Path]) -> trimesh.Trimesh:
    input_path = resolve_project_path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input GLB file not found: {to_display_path(input_path)}")

    mesh = trimesh.load(str(input_path), force="mesh")

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("Failed to load GLB as a Trimesh object.")

    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError("The mesh has no vertices.")

    return mesh


# Get vertical axis and horizontal axes.
def get_axis_indices(up_axis: str) -> Tuple[int, List[int]]:
    axis_map = {
        "x": 0,
        "y": 1,
        "z": 2
    }

    up_axis = up_axis.lower()

    if up_axis not in axis_map:
        raise ValueError("up_axis must be one of: x, y, z")

    up_idx = axis_map[up_axis]
    horizontal_idx = [i for i in range(3) if i != up_idx]

    return up_idx, horizontal_idx


# Compute plant height from mesh vertices.
def compute_height_from_vertices(
    vertices: np.ndarray,
    config: AnalysisConfig
) -> Dict:
    up_idx, _ = get_axis_indices(config.up_axis)
    h_values = vertices[:, up_idx]

    h_min = float(np.percentile(h_values, config.lower_percentile))
    h_max = float(np.percentile(h_values, config.upper_percentile))
    height = h_max - h_min

    return {
        "height": float(height),
        "min_height": h_min,
        "max_height": h_max
    }


# Sample surface points from the mesh.
def sample_points_from_mesh(
    mesh: trimesh.Trimesh,
    config: AnalysisConfig
) -> np.ndarray:
    np.random.seed(config.seed)
    points = mesh.sample(config.sample_count)

    return np.asarray(points)


# Estimate the main stem centerline.
def estimate_main_stem_centerline(
    points: np.ndarray,
    config: AnalysisConfig
) -> Dict:
    up_idx, horizontal_idx = get_axis_indices(config.up_axis)

    h_values = points[:, up_idx]
    h_min = float(np.min(h_values))
    h_max = float(np.max(h_values))

    bins = np.linspace(h_min, h_max, config.stem_center_slices + 1)
    centers_h = 0.5 * (bins[:-1] + bins[1:])

    global_xy = np.median(points[:, horizontal_idx], axis=0)
    centerline_xy = np.full((config.stem_center_slices, 2), np.nan, dtype=np.float64)

    for i in range(config.stem_center_slices):
        mask = (h_values >= bins[i]) & (h_values < bins[i + 1])
        slice_points = points[mask]

        if len(slice_points) < 20:
            continue

        xy = slice_points[:, horizontal_idx]
        dist_to_global_center = np.linalg.norm(xy - global_xy, axis=1)

        keep_count = max(20, int(len(slice_points) * 0.25))
        keep_count = min(keep_count, len(slice_points))

        keep_idx = np.argsort(dist_to_global_center)[:keep_count]
        centerline_xy[i] = np.median(xy[keep_idx], axis=0)

    valid = ~np.isnan(centerline_xy[:, 0])

    if np.sum(valid) < 2:
        raise RuntimeError("Failed to estimate main stem centerline.")

    for dim in range(2):
        centerline_xy[:, dim] = np.interp(
            centers_h,
            centers_h[valid],
            centerline_xy[valid, dim]
        )

    centerline_xy[:, 0] = gaussian_filter1d(centerline_xy[:, 0], sigma=2.0)
    centerline_xy[:, 1] = gaussian_filter1d(centerline_xy[:, 1], sigma=2.0)

    return {
        "height_centers": centers_h,
        "centerline_xy": centerline_xy
    }


# Interpolate the stem center at one height.
def interpolate_stem_xy(
    centerline_info: Dict,
    height_value: float
) -> np.ndarray:
    height_centers = centerline_info["height_centers"]
    centerline_xy = centerline_info["centerline_xy"]

    x = np.interp(height_value, height_centers, centerline_xy[:, 0])
    y = np.interp(height_value, height_centers, centerline_xy[:, 1])

    return np.asarray([x, y], dtype=np.float64)


# Create one 3D point on the stem centerline.
def make_point_on_stem(
    height_value: float,
    stem_xy: np.ndarray,
    up_axis: str
) -> List[float]:
    up_idx, horizontal_idx = get_axis_indices(up_axis)

    point = np.zeros(3, dtype=np.float64)
    point[up_idx] = height_value
    point[horizontal_idx[0]] = stem_xy[0]
    point[horizontal_idx[1]] = stem_xy[1]

    return point.tolist()


# Measure whether a point cluster is line-like or plane-like.
def pca_shape_features(points: np.ndarray) -> Dict:
    if len(points) < 3:
        return {
            "linearity": 0.0,
            "planarity": 0.0
        }

    centered = points - np.mean(points, axis=0)
    cov = np.cov(centered.T)

    eigenvalues, _ = np.linalg.eigh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]

    l1 = max(float(eigenvalues[0]), 1e-12)
    l2 = max(float(eigenvalues[1]), 0.0)
    l3 = max(float(eigenvalues[2]), 0.0)

    linearity = (l1 - l2) / l1
    planarity = (l2 - l3) / l1

    return {
        "linearity": float(linearity),
        "planarity": float(planarity)
    }


# Merge nearby node candidates and place them on the stem.
def merge_node_candidates_on_stem(
    candidates: List[Dict],
    merge_distance: float,
    centerline_info: Dict,
    config: AnalysisConfig
) -> List[Dict]:
    if len(candidates) == 0:
        return []

    candidates = sorted(candidates, key=lambda item: item["height"])

    groups = []
    current_group = [candidates[0]]

    for candidate in candidates[1:]:
        current_mean = float(np.mean([item["height"] for item in current_group]))

        if abs(candidate["height"] - current_mean) <= merge_distance:
            current_group.append(candidate)
        else:
            groups.append(current_group)
            current_group = [candidate]

    groups.append(current_group)

    nodes = []

    for group in groups:
        node_height = float(np.mean([item["height"] for item in group]))

        stem_xy = interpolate_stem_xy(
            centerline_info=centerline_info,
            height_value=node_height
        )

        node_point = make_point_on_stem(
            height_value=node_height,
            stem_xy=stem_xy,
            up_axis=config.up_axis
        )

        nodes.append({
            "height": node_height,
            "point": node_point
        })

    return nodes


# Estimate node positions from stem-attached branch clusters.
def estimate_nodes_by_stem_attachment_pca(
    points: np.ndarray,
    config: AnalysisConfig
) -> List[Dict]:
    up_idx, horizontal_idx = get_axis_indices(config.up_axis)

    h_values = points[:, up_idx]
    h_min = float(np.min(h_values))
    h_max = float(np.max(h_values))
    height_range = h_max - h_min

    if height_range <= 0:
        raise ValueError("Invalid point cloud height range.")

    centerline_info = estimate_main_stem_centerline(points, config)
    slice_centers = np.linspace(h_min, h_max, config.num_slices)

    vertical_window = height_range * config.vertical_window_ratio
    dbscan_eps = height_range * config.dbscan_eps_ratio
    stem_core_radius = height_range * config.stem_core_radius_ratio
    near_stem_radius = height_range * config.near_stem_radius_ratio
    min_outward_radius = height_range * config.min_outward_radius_ratio
    min_branch_length = height_range * config.min_branch_length_ratio
    merge_distance = height_range * config.merge_distance_ratio

    candidates = []

    for h in slice_centers:
        window_mask = np.abs(h_values - h) <= vertical_window
        window_points = points[window_mask]

        if len(window_points) < config.dbscan_min_samples * 2:
            continue

        stem_xy = interpolate_stem_xy(centerline_info, h)

        xy = window_points[:, horizontal_idx]
        radial_dist = np.linalg.norm(xy - stem_xy, axis=1)

        non_core_mask = radial_dist >= stem_core_radius
        candidate_points = window_points[non_core_mask]

        if len(candidate_points) < config.dbscan_min_samples * 2:
            continue

        labels = DBSCAN(
            eps=dbscan_eps,
            min_samples=config.dbscan_min_samples
        ).fit_predict(candidate_points)

        valid_labels = set(labels.tolist())
        valid_labels.discard(-1)

        for label in valid_labels:
            cluster = candidate_points[labels == label]

            if len(cluster) < config.dbscan_min_samples:
                continue

            cluster_xy = cluster[:, horizontal_idx]
            cluster_radial = np.linalg.norm(cluster_xy - stem_xy, axis=1)

            min_radial = float(np.min(cluster_radial))
            max_radial = float(np.max(cluster_radial))
            radial_extent = max_radial - min_radial

            if min_radial > near_stem_radius:
                continue

            if max_radial < min_outward_radius:
                continue

            if radial_extent < min_branch_length:
                continue

            features = pca_shape_features(cluster)

            if features["linearity"] < config.min_linearity:
                continue

            if features["planarity"] > config.max_planarity:
                continue

            nearest_idx = int(np.argmin(cluster_radial))
            attachment_height = float(cluster[nearest_idx, up_idx])

            candidates.append({
                "height": attachment_height
            })

    nodes = merge_node_candidates_on_stem(
        candidates=candidates,
        merge_distance=merge_distance,
        centerline_info=centerline_info,
        config=config
    )

    return nodes


# Create one red sphere marker.
def create_sphere_marker(
    point: np.ndarray,
    radius: float
) -> trimesh.Trimesh:
    sphere = trimesh.creation.uv_sphere(
        radius=radius,
        count=[16, 16]
    )

    sphere.apply_translation(point)

    sphere.visual.vertex_colors = np.tile(
        np.array([255, 0, 0, 255], dtype=np.uint8),
        (len(sphere.vertices), 1)
    )

    return sphere


# Export the plant mesh with red node markers.
def export_annotated_glb(
    plant_mesh: trimesh.Trimesh,
    nodes: List[Dict],
    height: float,
    output_path: Union[str, Path],
    config: AnalysisConfig
) -> Path:
    output_path = resolve_project_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scene = trimesh.Scene()
    scene.add_geometry(
        plant_mesh.copy(),
        node_name="plant_mesh"
    )

    marker_radius = height * config.marker_radius_ratio

    for idx, node in enumerate(nodes):
        point = np.asarray(node["point"], dtype=np.float64)

        marker = create_sphere_marker(
            point=point,
            radius=marker_radius
        )

        scene.add_geometry(
            marker,
            node_name=f"node_marker_{idx + 1:02d}"
        )

    scene.export(str(output_path))

    return output_path


# Analyze one plant model and return height and node count.
def analyze_plant_model(
    input_path: Union[str, Path] = INPUT_FILE,
    output_glb: Union[str, Path] = OUTPUT_GLB,
    config: Optional[AnalysisConfig] = None
) -> Dict:
    if config is None:
        config = AnalysisConfig()

    input_path = resolve_project_path(input_path)

    mesh = load_glb_as_single_mesh(input_path)
    vertices = np.asarray(mesh.vertices)

    height_info = compute_height_from_vertices(
        vertices=vertices,
        config=config
    )

    points = sample_points_from_mesh(
        mesh=mesh,
        config=config
    )

    nodes = estimate_nodes_by_stem_attachment_pca(
        points=points,
        config=config
    )

    output_glb_path = export_annotated_glb(
        plant_mesh=mesh,
        nodes=nodes,
        height=height_info["height"],
        output_path=output_glb,
        config=config
    )

    result = {
        "input_file": to_display_path(input_path),
        "output_glb": to_display_path(output_glb_path),

        # Use these two fields in other workflows:
        "height": height_info["height"],
        "node_count": len(nodes),

        "node_heights": [node["height"] for node in nodes]
    }

    return result


# Run the default workflow.
def main() -> None:
    result = analyze_plant_model()

    print(f"Input file: {result['input_file']}")
    print(f"Height: {result['height']:.6f}")
    print(f"Estimated node count: {result['node_count']}")
    print(f"Annotated GLB output: {result['output_glb']}")


if __name__ == "__main__":
    main()