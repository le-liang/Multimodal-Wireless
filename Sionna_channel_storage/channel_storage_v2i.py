import os # Configure which GPU 
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

tf.get_logger().setLevel('ERROR')

try:  # detect if the notebook runs in Colab
    import google.colab
    no_preview = True # deactivate preview
except:
    if os.getenv("SIONNA_NO_PREVIEW"):
        no_preview = True
    else:
        no_preview = False

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

data_folder_cav1 = '/data/CARLA_dataset_sunny/Town03/Town03_roundabout_seed42/cav_1'
data_folder_cav2 = '/data/CARLA_dataset_sunny/Town03/Town03_roundabout_seed42/cav_2'
data_folder_cav3 = '/data/CARLA_dataset_sunny/Town03/Town03_roundabout_seed42/cav_3'
data_folder_rsu1 = '/data/CARLA_dataset_sunny/Town03/Town03_roundabout_seed42/rsu_1'
scenes_folder = '/data/scene_generator'  # Root folder for scenes (e.g., scenes/0402/0402.xml)

all_cav_data_paths = [data_folder_cav1, data_folder_cav2, data_folder_cav3]
    # Consolidated configurations for repeated runs
config = [
    {"tx_rows": 1,"tx_cols": 16, "rx_rows": 1, "rx_cols": 16, "fc": 28e9},
]

# Use data_folder_cav1 to list the frames
reference_cav_data_folder = data_folder_cav1
yaml_files_in_ref_dir = sorted([f for f in os.listdir(reference_cav_data_folder) if f.endswith(".yaml")])

print(f"Found {len(yaml_files_in_ref_dir)} frames to process.")

for yaml_filename in yaml_files_in_ref_dir:
    frame_id = os.path.splitext(yaml_filename)[0]  # e.g., "0402"

    scene_folder_for_frame = os.path.join(scenes_folder, frame_id)
    frame_scene_file = os.path.join(scene_folder_for_frame, f"{frame_id}.xml")   # find scene of the frame

    print(f"\nProcessing Frame ID: {frame_id} using scene: {frame_scene_file}")

    try:
        scene = load_scene(frame_scene_file)
    except Exception as e:
        print(f"  Error loading scene {frame_scene_file}: {e}. Skipping frame.")
        continue
    #Load materials for 28GHz
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
    # for rm in scene.radio_materials.values():
    #     rm.scattering_coefficient = 1 / np.sqrt(3)


    for cfg in config:
        scene.tx_array = PlanarArray(num_rows=cfg["tx_rows"],
                                     num_cols=cfg["tx_cols"],
                                     vertical_spacing=0.5,
                                     horizontal_spacing=0.5,
                                     pattern="dipole",
                                     polarization="V")
        scene.rx_array = PlanarArray(num_rows=cfg["rx_rows"],
                                     num_cols=cfg["rx_cols"],
                                     vertical_spacing=0.5,
                                     horizontal_spacing=0.5,
                                     pattern="dipole",
                                     polarization="V")

        yaml_file_rsu1 = os.path.join(data_folder_rsu1, f"{frame_id}.yaml")
        tx_pos, tx_rot, _ = load_loc_speed(yaml_file_rsu1)   #load RSU location
        tx = Transmitter(name="tx", position=tx_pos, orientation=tx_rot)
        scene.add(tx)
        scene.frequency = cfg["fc"]
        scene.synthetic_array = True
        # set rx positions as different CAV positions
        cav_ids = ["cav_1", "cav_2", "cav_3"]
        for i, cav_data_folder_root in enumerate(all_cav_data_paths):
            cav_id_str = cav_ids[i]
            cav_specific_yaml_file = os.path.join(cav_data_folder_root, f"{frame_id}.yaml")
            print(f"  Processing {cav_id_str} (YAML: {cav_specific_yaml_file})...")

            ue_position, ue_rot, cav_speed = load_loc_speed(cav_specific_yaml_file)

            rx_name_current = f"rx_{cav_id_str}"
            rx = Receiver(name=rx_name_current, position=ue_position, orientation=ue_rot)
            scene.add(rx)

            h_freq_cav = None
            path_properties_cav = None

            paths = scene.compute_paths(max_depth=1, num_samples=1e6)
            a, _ = paths.cir()

            current_shape = a.shape
            if dim_police(a.shape):
                # paths = scene.compute_paths(max_depth=1, num_samples=3e5, scattering=True)
                # a, tau = paths.cir()
                sys.exit(0)

            theta_t, phi_t = aod_transform(paths.theta_t, paths.phi_t, tx_rot)
            theta_r, phi_r = aod_transform(paths.theta_r, paths.phi_r, ue_rot)
            path_properties_cav = {
                'a': a, 'tau': paths.tau,
                'theta_t': theta_t, 'phi_t': phi_t,
                'theta_r': theta_r, 'phi_r': phi_r,
                'glob_theta_t': paths.theta_t, 'glob_phi_t': paths.phi_t,
                'glob_theta_r': paths.theta_r, 'glob_phi_r': paths.phi_r
            }

            print(f"Channel coefficient: {a.shape}")
            save_path = os.path.join(
                f"Nt_{cfg['tx_cols']}_Nr_{cfg['rx_cols']}_fc_{int(scene.frequency / 1e9)}GHz_3",
                cav_id_str
            )
            os.makedirs(save_path, exist_ok=True)
            H_data_storage(path_properties_cav, save_path, frame_id, cav_id_str)

            scene.remove(rx.name)

        scene.remove(tx.name)
