import os # Configure which GPU 
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Sionna
import sionna
import sys
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
else:
        print("No GPU detected, using CPU.")

# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

try: # detect if the notebook runs in Colab
    import google.colab
    no_preview = True # deactivate preview
except:
    if os.getenv("SIONNA_NO_PREVIEW"):
        no_preview = True
    else:
        no_preview = False

resolution = [480,320] # increase for higher quality of renderings

# Define magic cell command to skip a cell if needed
from IPython.core.magic import register_cell_magic
from IPython import get_ipython

import matplotlib.pyplot as plt
import numpy as np
import time
import yaml

# Import Sionna RT components
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera, RadioMaterial, LambertianPattern

# For link-level simulations
from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies, OFDMChannel, ApplyOFDMChannel, CIRDataset
from sionna.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.utils import compute_ber, ebnodb2no, PlotBER
from sionna.ofdm import KBestDetector, LinearDetector
from sionna.mimo import StreamManagement
from data_utils import load_loc_speed, H_data_storage, dim_police, aod_transform
# Set random seed for reproducibility
sionna.config.seed = 42

# Discover CAV data folders dynamically under the data root directory
data_root = '/data/CARLA_dataset_sunny/Town03/Town03_5wayroad_seed28'
all_cav_data_paths = [
    os.path.join(data_root, d)
    for d in sorted(os.listdir(data_root))
    if os.path.isdir(os.path.join(data_root, d)) and d.startswith('cav_')
]

# Derive cav ids and count
cav_ids = [os.path.basename(p) for p in all_cav_data_paths]
num_cav = len(cav_ids)

scenes_folder = '/data/scene_generator'  # Root folder for scenes (e.g., scenes/0402/0402.xml)

# Use the first discovered CAV directory as reference for frames
reference_cav_data_folder = all_cav_data_paths[0] if all_cav_data_paths else None

yaml_files_in_ref_dir = sorted([f for f in os.listdir(reference_cav_data_folder) if f.endswith(".yaml")])

print(f"Found {len(yaml_files_in_ref_dir)} frames to process based on YAMLs in {reference_cav_data_folder}.")

for yaml_filename in yaml_files_in_ref_dir:
    frame_id = os.path.splitext(yaml_filename)[0]  # e.g., "0402"

    scene_folder_for_frame = os.path.join(scenes_folder, frame_id)
    frame_scene_file = os.path.join(scene_folder_for_frame, f"{frame_id}.xml")

    # Call the modified processing function for this frame and all CAVs
    print(f"\nProcessing Frame ID: {frame_id} using scene: {frame_scene_file}")

    try:
        scene = load_scene(frame_scene_file)
    except Exception as e:
        print(f"  Error loading scene {frame_scene_file}: {e}. Skipping frame.")
        continue

    scene.tx_array = PlanarArray(num_rows=1,
                                 num_cols=16,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern="dipole",
                                 polarization="V")  # Assuming fixed frequency from original

    # Configure antenna array for all receivers (as in original)
    scene.rx_array = PlanarArray(num_rows=1,
                                 num_cols=16,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern="dipole",
                                 polarization="V")

    itu_wet_ground_28 = RadioMaterial("itu_wet_ground_28",
                                      relative_permittivity=3,
                                      conductivity=2.5,
                                      scattering_coefficient=0.0,
                                      xpd_coefficient=0.0,
                                      scattering_pattern=LambertianPattern(),
                                      frequency_update_callback=None)

    scene.add(itu_wet_ground_28)

    itu_medium_dry_ground_28 = RadioMaterial("itu_medium_dry_ground_28",
                                             relative_permittivity=3,
                                             conductivity=0.4,
                                             scattering_coefficient=0.0,
                                             xpd_coefficient=0.0,
                                             scattering_pattern=LambertianPattern())

    scene.add(itu_medium_dry_ground_28)

    itu_very_dry_ground_28 = RadioMaterial("itu_very_dry_ground_28",
                                           relative_permittivity=2.5,
                                           conductivity=0.03,
                                           scattering_coefficient=0.0,
                                           xpd_coefficient=0.0,
                                           scattering_pattern=LambertianPattern())

    scene.add(itu_very_dry_ground_28)
    for rm in scene.radio_materials.values():
        rm.scattering_coefficient = 1 / np.sqrt(3)


    scene.frequency = 28e9
    scene.synthetic_array = True

    subcarrier_spacing = 120e3
    fft_size = 1024

    # Loop through CAVs (tx) and for each, iterate all other CAVs (rx)
    for i, cav_data_folder_root in enumerate(all_cav_data_paths):
        cav_id_str = cav_ids[i]
        tx_cav_yaml = os.path.join(cav_data_folder_root, f"{frame_id}.yaml")
        print(f"  Processing {cav_id_str} (YAML: {tx_cav_yaml})...")
        tx_position, tx_rot, cav_speed = load_loc_speed(tx_cav_yaml)
        tx = Transmitter(name="tx", position=tx_position, orientation=tx_rot)
        scene.add(tx)

        for j in range(num_cav):
            if j == i:
                continue
            rx_cav_yaml = os.path.join(all_cav_data_paths[j], f"{frame_id}.yaml")
            rx_position, rx_rot, cav_speed_rx = load_loc_speed(rx_cav_yaml)

            rx = Receiver(name=f'rx_{j}', position=rx_position, orientation=rx_rot)
            scene.add(rx)

            # paths = scene.compute_paths(max_depth=1, num_samples=3e5, scattering=True)
            paths = scene.compute_paths(max_depth=1, num_samples=1e6)
            a, tau = paths.cir()

            if dim_police(a.shape):
                paths = scene.compute_paths(max_depth=1, num_samples=3e5, scattering=True)
                a, tau = paths.cir()

            theta_t, phi_t = aod_transform(paths.theta_t, paths.phi_t, tx_rot)
            theta_r, phi_r = aod_transform(paths.theta_r, paths.phi_r, rx_rot)
            path_properties_cav = {
                'a': a, 'tau': paths.tau,
                'theta_t': theta_t, 'phi_t': phi_t,
                'theta_r': theta_r, 'phi_r': phi_r,
                'glob_theta_t': paths.theta_t, 'glob_phi_t': paths.phi_t,
                'glob_theta_r': paths.theta_r, 'glob_phi_r': paths.phi_r
            }

            print(f"Channel coefficient: {a.shape}")
            save_path = os.path.join(f"Nt_{16}_Nr_{16}_fc_{int(scene.frequency / 1e9)}GHz", cav_id_str)
            save_path = os.path.join(save_path, cav_ids[j])
            os.makedirs(save_path, exist_ok=True)
            H_data_storage(path_properties_cav, save_path, frame_id, cav_id_str)

            scene.remove(rx.name)

        scene.remove(tx.name)
