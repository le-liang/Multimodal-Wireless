import numpy as np
import yaml
import os
import sys
import tensorflow as tf
import pandas as pd

from sionna.rt.utils import mi_to_tf_tensor
from scipy.spatial.transform import Rotation


def find_scenario_folders(root_folder):
    """Return scenario folders below ``root_folder`` in relative-path order."""
    scenario_folders = []
    for folder, directory_names, _ in os.walk(root_folder):
        if "scenes" not in directory_names:
            continue

        scenario_folders.append(folder)
        directory_names[:] = []

    scenario_folders.sort(key=lambda folder: os.path.relpath(folder, root_folder))
    if not scenario_folders:
        raise ValueError(f"No scene folders found in {root_folder}")

    return scenario_folders


def find_scene_frames(scene_folder):
    """Return numeric scene indices and the XML for the earliest frame."""
    scene_indices = [
        os.path.splitext(filename)[0]
        for filename in os.listdir(scene_folder)
        if filename.endswith(".yaml")
    ]
    scene_indices.sort(key=int)
    if not scene_indices:
        raise ValueError(f"No scene YAML files found in {scene_folder}")

    initial_index = scene_indices[0]
    initial_scene_file = os.path.join(scene_folder, initial_index, f"{initial_index}.xml")
    return scene_indices, initial_scene_file


def update_scene(scene, scene_folder, current_index, new_index):
    """Advance a loaded scene from ``current_index`` to ``new_index``."""
    current_yaml = os.path.join(scene_folder, f"{current_index}.yaml")
    new_yaml = os.path.join(scene_folder, f"{new_index}.yaml")
    with open(current_yaml, "r", encoding="utf-8") as yaml_file:
        current_frame = yaml.safe_load(yaml_file)
    with open(new_yaml, "r", encoding="utf-8") as yaml_file:
        new_frame = yaml.safe_load(yaml_file)

    current_vehicles = current_frame["vehicles"]
    new_vehicles = new_frame["vehicles"]
    if len(current_vehicles) != len(new_vehicles):
        raise ValueError("Current and new scene YAMLs contain different vehicle counts")

    scene_params = scene.mi_scene_params
    solver_paths = scene.solver_paths
    rdtype = scene.dtype.real_dtype

    for index, (current_vehicle, new_vehicle) in enumerate(zip(current_vehicles, new_vehicles)):
        object_name = f"Vehicle_{index}"
        vehicle_object = scene.get(object_name)

        current_location = current_vehicle["pose"]["location"]
        new_location = new_vehicle["pose"]["location"]
        current_rotation = current_vehicle["pose"]["rotation"]
        new_rotation = new_vehicle["pose"]["rotation"]

        current_position = np.array([current_location["x"], -current_location["y"], current_location["z"]], dtype=np.float64)
        new_position = np.array([new_location["x"], -new_location["y"], new_location["z"]], dtype=np.float64)
        current_angles = np.deg2rad([-current_rotation["yaw"], -current_rotation["pitch"], current_rotation["roll"]])
        new_angles = np.deg2rad([-new_rotation["yaw"], -new_rotation["pitch"], new_rotation["roll"]])

        current_matrix = Rotation.from_euler("ZYX", current_angles).as_matrix()
        new_matrix = Rotation.from_euler("ZYX", new_angles).as_matrix()
        relative_matrix = new_matrix @ current_matrix.T
        previous_matrix = Rotation.from_euler("ZYX", vehicle_object.orientation.numpy()).as_matrix()
        scene_matrix = relative_matrix @ previous_matrix
        scene_angles = Rotation.from_matrix(scene_matrix).as_euler("ZYX")
        translation = tf.cast(new_position - relative_matrix @ current_position, rdtype)
        relative_matrix = tf.cast(relative_matrix, rdtype)

        vertex_key = vehicle_object._vertex_params_name(vehicle_object.name)
        vertices = mi_to_tf_tensor(scene_params[vertex_key], rdtype)
        vertices = tf.reshape(vertices, [-1, 3])
        vertices = tf.einsum("ij,...j->...i", relative_matrix, vertices)
        vertices += translation
        flat_vertices = tf.reshape(vertices, [-1])
        scene_params[vertex_key] = vehicle_object._mi_scalar_t(flat_vertices)

        object_id = vehicle_object.object_id
        primitive_offset = solver_paths.prim_offsets[object_id]
        primitive_indices = tf.range(primitive_offset, primitive_offset + vehicle_object._mi_shape.face_count(), dtype=tf.int32)
        primitive_indices = tf.expand_dims(primitive_indices, axis=1)
        primitives = solver_paths.primitives.gather_nd(primitive_indices)
        primitives = tf.einsum("ij,...j->...i", relative_matrix, primitives)
        primitives += translation
        solver_paths.primitives.scatter_nd_update(primitive_indices, primitives)

        normals = solver_paths.normals.gather_nd(primitive_indices)
        normals = tf.einsum("ij,...j->...i", relative_matrix, normals)
        solver_paths.normals.scatter_nd_update(primitive_indices, normals)

        wedge_indices, _ = tf.unique(tf.where(solver_paths.wedges_objects == object_id)[:, 0])
        wedge_indices = tf.expand_dims(wedge_indices, axis=1)
        wedge_origins = solver_paths.wedges_origin.gather_nd(wedge_indices)
        wedge_origins = tf.einsum("ij,...j->...i", relative_matrix, wedge_origins)
        wedge_origins += translation
        solver_paths.wedges_origin.scatter_nd_update(wedge_indices, wedge_origins)

        wedge_directions = solver_paths.wedges_e_hat.gather_nd(wedge_indices)
        wedge_directions = tf.einsum("ij,...j->...i", relative_matrix, wedge_directions)
        solver_paths.wedges_e_hat.scatter_nd_update(wedge_indices, wedge_directions)

        wedge_normals = solver_paths.wedges_normals.gather_nd(wedge_indices)
        wedge_normals = tf.einsum("ij,...j->...i", relative_matrix, wedge_normals)
        solver_paths.wedges_normals.scatter_nd_update(wedge_indices, wedge_normals)

        vehicle_object._orientation = tf.cast(scene_angles, rdtype)

    scene_params.update()
    scene.scene_geometry_updated()
