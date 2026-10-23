import os


# Configure the GPU before importing TensorFlow and Sionna.
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import sionna
import tensorflow as tf
from sionna.rt import LambertianPattern, PlanarArray, RadioMaterial, Receiver, Transmitter, load_scene

from utils.configuration_utils import centered_array_indices, group_configurations
from utils.data_utils import H_data_storage, aod_transform, dim_police, load_loc_speed
from utils.scene_utils import find_scenario_folders, find_scene_frames, update_scene


gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as error:
        print(error)
else:
    print("No GPU detected, using CPU.")

tf.get_logger().setLevel("ERROR")
sionna.config.seed = 42


ROOT_FOLDER = "/home/user5090/mth/scene_generator/sunny_yaml"
CONFIGURATIONS = [
    {"tx_rows": 1, "tx_cols": 16, "rx_rows": 1, "rx_cols": 16, "fc": 28e9},
]


if not CONFIGURATIONS:
    raise ValueError("CONFIGURATIONS must contain at least one configuration")

configuration_groups = group_configurations(CONFIGURATIONS)
scenario_folders = find_scenario_folders(ROOT_FOLDER)
print(f"Found {len(scenario_folders)} scenarios to process.")

for scenario_folder in scenario_folders:
    residual_path = os.path.relpath(scenario_folder, ROOT_FOLDER)
    scene_folder = os.path.join(scenario_folder, "scenes")
    scene_indices, initial_scene_file = find_scene_frames(scene_folder)
    if not os.path.isfile(initial_scene_file):
        print(f"{residual_path}: skipped (initial scene not found)")
        continue

    indexed_cav_paths = []
    for name in os.listdir(scenario_folder):
        path = os.path.join(scenario_folder, name)
        if name.startswith("cav_") and os.path.isdir(path):
            indexed_cav_paths.append((int(name.removeprefix("cav_")), path))
    indexed_cav_paths.sort()
    if not indexed_cav_paths:
        raise ValueError(f"No CAV folders found in {scenario_folder}")

    cav_paths = [path for _, path in indexed_cav_paths]
    cav_ids = [os.path.basename(path) for path in cav_paths]

    total_frames = len(scene_indices)
    collected_frames = 0
    print(f"{residual_path}: {collected_frames}/{total_frames}", end="", flush=True)
    scene = load_scene(initial_scene_file)

    ground_materials = [
        RadioMaterial(
            "itu_wet_ground_28",
            relative_permittivity=3,
            conductivity=2.5,
            scattering_coefficient=0.0,
            xpd_coefficient=0.0,
            scattering_pattern=LambertianPattern(),
            frequency_update_callback=None,
        ),
        RadioMaterial(
            "itu_medium_dry_ground_28",
            relative_permittivity=3,
            conductivity=0.4,
            scattering_coefficient=0.0,
            xpd_coefficient=0.0,
            scattering_pattern=LambertianPattern(),
        ),
        RadioMaterial(
            "itu_very_dry_ground_28",
            relative_permittivity=2.5,
            conductivity=0.03,
            scattering_coefficient=0.0,
            xpd_coefficient=0.0,
            scattering_pattern=LambertianPattern(),
        ),
    ]
    for material in ground_materials:
        scene.add(material)
    for material in scene.radio_materials.values():
        material.scattering_coefficient = 1 / np.sqrt(3)

    scene.synthetic_array = True
    scene_checked = False
    current_index = scene_indices[0]
    for scene_index in scene_indices:
        if scene_index != current_index:
            update_scene(scene, scene_folder, current_index, scene_index)
            current_index = scene_index

        endpoints = []
        for cav_id, cav_path in zip(cav_ids, cav_paths):
            cav_yaml = os.path.join(cav_path, f"{scene_index}.yaml")
            position, rotation, _ = load_loc_speed(cav_yaml)
            tx = Transmitter(name=f"tx_{cav_id}", position=position, orientation=rotation)
            rx = Receiver(name=f"rx_{cav_id}", position=position, orientation=rotation)
            scene.add(tx)
            scene.add(rx)
            endpoint = {
                "id": cav_id,
                "tx_name": tx.name,
                "rx_name": rx.name,
                "rotation": rotation,
            }
            endpoints.append(endpoint)

        # Synthetic arrays trace device centers independently of array dimensions.
        trace_cfg = CONFIGURATIONS[0]
        scene.tx_array = PlanarArray(
            num_rows=trace_cfg["tx_rows"],
            num_cols=trace_cfg["tx_cols"],
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="dipole",
            polarization="V",
        )
        scene.rx_array = PlanarArray(
            num_rows=trace_cfg["rx_rows"],
            num_cols=trace_cfg["rx_cols"],
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="dipole",
            polarization="V",
        )
        scene.frequency = trace_cfg["fc"]
        traced_paths = scene.trace_paths(max_depth=1, num_samples=1e6, check_scene=not scene_checked)
        scene_checked = True

        for group in configuration_groups:
            scene.tx_array = PlanarArray(
                num_rows=group["tx_rows"],
                num_cols=group["tx_cols"],
                vertical_spacing=0.5,
                horizontal_spacing=0.5,
                pattern="dipole",
                polarization="V",
            )
            scene.rx_array = PlanarArray(
                num_rows=group["rx_rows"],
                num_cols=group["rx_cols"],
                vertical_spacing=0.5,
                horizontal_spacing=0.5,
                pattern="dipole",
                polarization="V",
            )
            scene.frequency = group["fc"]
            paths = scene.compute_fields(*traced_paths, check_scene=False)
            coefficients, _ = paths.cir()

            for rx_index, receiver in enumerate(endpoints):
                for tx_index, transmitter in enumerate(endpoints):
                    if rx_index == tx_index:
                        continue

                    valid_indices = tf.where(paths.mask[0, rx_index, tx_index])[:, 0]
                    link_coefficients = coefficients[:, rx_index:rx_index + 1, :, tx_index:tx_index + 1]
                    link_coefficients = tf.gather(link_coefficients, valid_indices, axis=-2)
                    tau = paths.tau[:, rx_index:rx_index + 1, tx_index:tx_index + 1]
                    tau = tf.gather(tau, valid_indices, axis=-1)
                    glob_theta_t = paths.theta_t[:, rx_index:rx_index + 1, tx_index:tx_index + 1]
                    glob_theta_t = tf.gather(glob_theta_t, valid_indices, axis=-1)
                    glob_phi_t = paths.phi_t[:, rx_index:rx_index + 1, tx_index:tx_index + 1]
                    glob_phi_t = tf.gather(glob_phi_t, valid_indices, axis=-1)
                    glob_theta_r = paths.theta_r[:, rx_index:rx_index + 1, tx_index:tx_index + 1]
                    glob_theta_r = tf.gather(glob_theta_r, valid_indices, axis=-1)
                    glob_phi_r = paths.phi_r[:, rx_index:rx_index + 1, tx_index:tx_index + 1]
                    glob_phi_r = tf.gather(glob_phi_r, valid_indices, axis=-1)
                    theta_t, phi_t = aod_transform(glob_theta_t, glob_phi_t, transmitter["rotation"])
                    theta_r, phi_r = aod_transform(glob_theta_r, glob_phi_r, receiver["rotation"])

                    for cfg in group["configurations"]:
                        tx_indices = centered_array_indices(group["tx_rows"], group["tx_cols"], cfg["tx_rows"], cfg["tx_cols"])
                        rx_indices = centered_array_indices(group["rx_rows"], group["rx_cols"], cfg["rx_rows"], cfg["rx_cols"])
                        cfg_coefficients = tf.gather(link_coefficients, rx_indices, axis=2)
                        cfg_coefficients = tf.gather(cfg_coefficients, tx_indices, axis=4)
                        dim_police(cfg_coefficients.shape)

                        path_properties = {
                            "a": cfg_coefficients,
                            "tau": tau,
                            "theta_t": theta_t,
                            "phi_t": phi_t,
                            "theta_r": theta_r,
                            "phi_r": phi_r,
                            "glob_theta_t": glob_theta_t,
                            "glob_phi_t": glob_phi_t,
                            "glob_theta_r": glob_theta_r,
                            "glob_phi_r": glob_phi_r,
                        }

                        frequency_ghz = int(scene.frequency / 1e9)
                        output_root = os.path.join(f"Nt_{cfg['tx_rows']}_{cfg['tx_cols']}_Nr_{cfg['rx_rows']}_{cfg['rx_cols']}_fc_{frequency_ghz}GHz", residual_path)
                        save_path = os.path.join(output_root, transmitter["id"], receiver["id"])
                        os.makedirs(save_path, exist_ok=True)
                        H_data_storage(path_properties, save_path, scene_index, transmitter["id"])

        for endpoint in endpoints:
            scene.remove(endpoint["rx_name"])
            scene.remove(endpoint["tx_name"])

        collected_frames += 1
        print(f"\r{residual_path}: {collected_frames}/{total_frames}", end="", flush=True)

    print()
