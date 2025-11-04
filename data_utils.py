import numpy as np
import yaml
import os
import sys
def load_loc_speed(filepath):
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    lidar_loc = data['sensors']['lidar_pose']['location']
    lidar_location = [lidar_loc.get('x', 0.0), -lidar_loc.get('y', 0.0), lidar_loc.get('z', 0.0)]
    actor_name = data.get('actor')
    if actor_name.startswith('rsu'):
        return lidar_location, None
    else:
        veh_speed = data['sensors']['vehicle_speed']['speed']
        vehicle_speed =[veh_speed.get('x', 0.0), -veh_speed.get('y', 0.0), veh_speed.get('z', 0.0)]
        return lidar_location, vehicle_speed


# Modified save_channel_data: now it also takes cav_id_str and cav_data_folder
def H_data_storage(path_properties, cav_data_folder, frame_id, cav_id_str):
    """Save channel and paths-related data to the CAV's specific data folder."""
    # Note: The original code saved into data_dir, which was the input dir.
    # This version saves into what was 'data_folder_cavX', now passed as cav_data_folder.
    # If you want a separate output directory structure, this function would need modification.
    # For now, it mirrors the original saving location behavior but per-CAV.

    if path_properties is not None:
        path_properties_numpy = {k: (v.numpy() if hasattr(v, 'numpy') else v)
                                 for k, v in path_properties.items()}
        paths_file = os.path.join(cav_data_folder, f"{frame_id}_paths.npz") # Added cav_id_str
        np.savez_compressed(paths_file, **path_properties_numpy)
        print(f"  Path attributes for {cav_id_str} (frame {frame_id}) saved to: {paths_file}")

def dim_police(current_shape):
    if hasattr(current_shape, 'as_list'):
        dims = current_shape.as_list()
    else:
        dims = list(current_shape)
    r = 0
    if any(d == 0 for d in dims):
        error_message = (
            f"Shape of 'a': {dims} with 0 dimension. No paths generated."
        )
        print(error_message, file=sys.stderr)
        r = 1
    return r