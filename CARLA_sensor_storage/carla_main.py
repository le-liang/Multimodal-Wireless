# -*- coding: utf-8 -*-
# Author: Tianhao Mao <tianhao@seu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import carla
import random
import os
import yaml
import numpy as np
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from data_utils.processor import get_location, get_transform, distt
import time
import traceback  # For printing tracebacks
from data_utils.sim_utils import SensorDataManager
from data_utils.localization import LocalizationManager

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)


actor_data_store = {}
executor = ThreadPoolExecutor(max_workers=16)
manager = SensorDataManager(executor, actor_data_store)


if __name__ == '__main__':
    sim_config = config['simulation']
    spectator_height = sim_config['spectator_height']
    seed_value = sim_config['seed']
    frame_rate = sim_config['frame_rate']
    MAX_FRAMES_TO_RUN = sim_config['duration_seconds'] * frame_rate
    queue_timeout = sim_config['queue_timeout']
    preferred_vehicle_names = sim_config['preferred_vehicles']

    scene_name = config['scenario']
    scenario_config = config['scenarios'][scene_name]
    rsu_transform = get_transform(scenario_config['rsu_transform'])
    map_name = scenario_config['map']
    spawn_center = get_location(scenario_config['spawn_center'])
    root_output_dir = config['output']['root_dir_template'].format(
        scenario_name=scene_name,
        seed=seed_value
    )

    # Create the dedicated directory for global scene description files
    scenes_output_dir = os.path.join(root_output_dir, 'scenes')
    os.makedirs(scenes_output_dir, exist_ok=True)

    actor_config = scenario_config['actors']
    num_veh = actor_config['num_vehicles']
    num_rsu = actor_config['num_rsu']
    desired_cav_ranks = actor_config['desired_cav_ranks']
    num_cav = len(desired_cav_ranks)

    is_warming_up = True
    manager.set_warming_up(True)

    print(f"Using Random Seed: {seed_value}")
    random.seed(seed_value)
    np.random.seed(seed_value)

    cav_vehicles_dict = {}
    all_spawned_vehicle_actors = []
    sensors_dict = {}
    all_actors_list = []
    sensor_list_for_callback = []

    rsu_actors_dict = {}
    rsu_sensors_dict = {}
    
    # Localization managers for each CAV
    localization_managers = {}

    client = None
    original_settings = None
    traffic_manager = None

    num_veh_to_spawn = num_veh
    num_cav_to_select = num_cav

    client = carla.Client('localhost', 2000)
    client.set_timeout(sim_config['client_timeout'])
    world = client.load_world(map_name)
    blueprint_library = world.get_blueprint_library()

    weather = carla.WeatherParameters(**sim_config['weather'])
    world.set_weather(weather)

    map_ = world.get_map()
    spawn_points = map_.get_spawn_points()

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.fixed_delta_seconds = 1 / frame_rate
    settings.synchronous_mode = True
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_random_device_seed(seed_value)

    sensor_queue = Queue()

    selected_spawn_points = []
    if not spawn_points:
        num_veh_to_spawn = 0
    else:
        spawn_point_distances = [{'transform': sp, 'distance': distt(spawn_center, sp.location)}
                                 for sp in spawn_points]
        spawn_point_distances.sort(key=lambda x: x['distance'])
        if len(spawn_point_distances) >= num_veh_to_spawn:
            selected_spawn_points = [sp_info['transform'] for sp_info in
                                     spawn_point_distances[:num_veh_to_spawn]]
        else:
            selected_spawn_points = [sp_info['transform'] for sp_info in spawn_point_distances]
            num_veh_to_spawn = len(selected_spawn_points)

    vehicle_blueprints_pref = [blueprint_library.find(name) for name in preferred_vehicle_names
                               if blueprint_library.find(name) is not None]
    vehicle_blueprints = vehicle_blueprints_pref[:num_veh_to_spawn]
    needed_bps = num_veh_to_spawn - len(vehicle_blueprints)
    if needed_bps > 0:
        available_bps_list = list(blueprint_library.filter('vehicle.*'))
        unique_available_bps = [bp for bp in available_bps_list if bp.id not in {b.id for b in vehicle_blueprints}]
        if len(unique_available_bps) >= needed_bps:
            vehicle_blueprints.extend(random.sample(unique_available_bps, k=needed_bps))
        else:
            vehicle_blueprints.extend(random.choices(available_bps_list, k=needed_bps))

    all_spawned_vehicle_actors = []
    for i in range(num_veh_to_spawn):
        if i >= len(vehicle_blueprints) or i >= len(selected_spawn_points): continue
        vehicle = world.try_spawn_actor(vehicle_blueprints[i], selected_spawn_points[i])
        if vehicle:
            all_spawned_vehicle_actors.append(vehicle);
            all_actors_list.append(vehicle)
            vehicle.set_autopilot(True, traffic_manager.get_port())
            traffic_manager.ignore_lights_percentage(vehicle, 100.0)
        else:
            print(f"Error spawning vehicle {i + 1}")

    actual_spawned_count = len(all_spawned_vehicle_actors)
    if num_cav_to_select > actual_spawned_count: num_cav_to_select = actual_spawned_count

    num_cav_to_select = len(desired_cav_ranks)
    selected_cav_actors = []
    if num_cav_to_select > 0 and actual_spawned_count > 0:
        vehicle_distances_to_center = []
        for actor in all_spawned_vehicle_actors:
            dist = distt(spawn_center, actor.get_location())
            vehicle_distances_to_center.append({'actor': actor, 'distance': dist})
        vehicle_distances_to_center.sort(key=lambda x: x['distance'])
        valid_selected_actors_by_rank = []
        for rank_index in desired_cav_ranks:
            if rank_index < len(vehicle_distances_to_center):
                selected_actor = vehicle_distances_to_center[rank_index]['actor']
                valid_selected_actors_by_rank.append(selected_actor)
            else:
                print(f"Warning: Rank {rank_index} is out of bounds. Skipping.")
        selected_cav_actors = valid_selected_actors_by_rank
        num_cav_to_select = len(selected_cav_actors)
        if num_cav_to_select > 0:
            print(f"\nFinal selected {num_cav_to_select} CAVs for data collection:")
            for i, cav_actor in enumerate(selected_cav_actors):
                cav_label = f"cav_{i + 1}"
                cav_vehicles_dict[cav_label] = cav_actor
                actor_save_dir = os.path.join(root_output_dir, cav_label)
                os.makedirs(actor_save_dir, exist_ok=True)
                actor_data_store[cav_label] = {"save_dir": actor_save_dir}
                sensors_dict[cav_label] = {"cameras": {}, "lidar": None, "imu": None, "semantic_lidar": None}

    actual_sensors_listening = 0
    cav_sensor_config = config['sensors']['cav']
    if num_cav_to_select > 0:
        for cav_label, vehicle in cav_vehicles_dict.items():
            # --- Spawning Cameras ---
            if 'camera' in cav_sensor_config:
                cam_conf = cav_sensor_config['camera']
                cam_bp = blueprint_library.find(cam_conf['blueprint'])
                for attr, val in cam_conf.get('attributes', {}).items(): cam_bp.set_attribute(attr, str(val))
                for cam_name, tf_dict in cam_conf['transforms'].items():
                    cam_tf = get_transform(tf_dict)
                    camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)
                    if camera:
                        sensors_dict[cav_label]['cameras'][cam_name] = camera
                        all_actors_list.append(camera);
                        sensor_list_for_callback.append(camera)
                        camera.listen(
                            lambda d, lbl=cav_label, st='camera', sn=cam_name: manager.sensor_callback(d, sensor_queue, lbl, st, sn))
                        actual_sensors_listening += 1

            # --- Spawning LiDAR ---
            if 'lidar' in cav_sensor_config:
                lidar_conf = cav_sensor_config['lidar']
                lidar_bp = blueprint_library.find(lidar_conf['blueprint'])
                for attr, val in lidar_conf.get('attributes', {}).items(): lidar_bp.set_attribute(attr, str(val))
                lidar_tf = get_transform(lidar_conf['transform'])
                lidar = world.spawn_actor(lidar_bp, lidar_tf, attach_to=vehicle)
                if lidar:
                    sensors_dict[cav_label]['lidar'] = lidar
                    all_actors_list.append(lidar);
                    sensor_list_for_callback.append(lidar)
                    lidar.listen(
                        lambda d, lbl=cav_label, st='lidar', sn='lidar': manager.sensor_callback(d, sensor_queue, lbl, st, sn))
                    actual_sensors_listening += 1

            # --- Spawning Semantic LiDAR ---
            if 'semantic_lidar' in cav_sensor_config:
                slidar_conf = cav_sensor_config['semantic_lidar']
                slidar_bp = blueprint_library.find(slidar_conf['blueprint'])
                for attr, val in slidar_conf.get('attributes', {}).items(): slidar_bp.set_attribute(attr, str(val))
                slidar_tf = get_transform(slidar_conf['transform'])
                semantic_lidar = world.spawn_actor(slidar_bp, slidar_tf, attach_to=vehicle)
                if semantic_lidar:
                    sensors_dict[cav_label]['semantic_lidar'] = semantic_lidar
                    all_actors_list.append(semantic_lidar)
                    sensor_list_for_callback.append(semantic_lidar)
                    semantic_lidar.listen(
                        lambda d, lbl=cav_label, st='semantic_lidar', sn='semantic_lidar': manager.sensor_callback(d, sensor_queue, lbl, st, sn))
                    actual_sensors_listening += 1
                    print(f"    Attached semantic_lidar to {cav_label}")

            # --- Spawning IMU ---
            if 'imu' in cav_sensor_config:
                imu_conf = cav_sensor_config['imu']
                imu_bp = blueprint_library.find(imu_conf['blueprint'])
                for attr, val in imu_conf.get('attributes', {}).items(): imu_bp.set_attribute(attr, str(val))
                imu_tf = get_transform(imu_conf['transform'])
                imu = world.spawn_actor(imu_bp, imu_tf, attach_to=vehicle)
                if imu:
                    sensors_dict[cav_label]['imu'] = imu
                    all_actors_list.append(imu);
                    sensor_list_for_callback.append(imu)
                    imu.listen(
                        lambda data, lbl=cav_label, st='imu', sn='imu': manager.sensor_callback(data, sensor_queue, lbl, st,
                                                                                                 sn))
                    actual_sensors_listening += 1
            
            # --- Initialize Localization Manager ---
            if 'localization' in config:
                try:
                    loc_config = config['localization'].copy()
                    loc_config['dt'] = 1.0 / frame_rate
                    localization_managers[cav_label] = LocalizationManager(vehicle, loc_config, map_)
                    all_actors_list.append(localization_managers[cav_label].gnss.sensor)
                    print(f"    Initialized localization manager for {cav_label}")
                except Exception as e:
                    print(f"    Warning: Failed to initialize localization for {cav_label}: {e}")
                    localization_managers[cav_label] = None

    if num_rsu > 0:
        print(f"\n--- Spawning {num_rsu} RSU(s) and their sensors ---")
        for i in range(num_rsu):
            rsu_label = f"rsu_{i + 1}"
            anchor_bp = blueprint_library.find('static.prop.trafficwarning')  # a small, simple static object
            rsu_anchor_actor = world.try_spawn_actor(anchor_bp, rsu_transform)

            rsu_actors_dict[rsu_label] = rsu_anchor_actor
            all_actors_list.append(rsu_anchor_actor)

            # Create directories and data structures
            actor_save_dir = os.path.join(root_output_dir, rsu_label)
            os.makedirs(actor_save_dir, exist_ok=True)
            actor_data_store[rsu_label] = {"save_dir": actor_save_dir}
            rsu_sensors_dict[rsu_label] = {"cameras": {}, "depth_cameras": {}, "lidar": None, "radar": None}

            for sensor_type_key, sensor_config in config['sensors']['rsu'].items():
                bp = blueprint_library.find(sensor_config['blueprint'])
                for attr, val in sensor_config.get('attributes', {}).items():
                    bp.set_attribute(attr, str(val))
                transform = get_transform(sensor_config['transform'])
                sensor_actor = world.spawn_actor(bp, transform, attach_to=rsu_anchor_actor)

                if sensor_actor:
                    all_actors_list.append(sensor_actor)
                    sensor_list_for_callback.append(sensor_actor)
                    actual_sensors_listening += 1

                    sensor_name_in_file = sensor_type_key.replace('_', '')
                    if sensor_type_key == 'camera':
                        sensor_name_in_file = "camera0"
                        rsu_sensors_dict[rsu_label]['cameras'][sensor_name_in_file] = sensor_actor
                    elif sensor_type_key == 'depth_camera':
                        sensor_name_in_file = "depth_camera0"
                        rsu_sensors_dict[rsu_label]['depth_cameras'][sensor_name_in_file] = sensor_actor
                    else:
                        rsu_sensors_dict[rsu_label][sensor_type_key] = sensor_actor

                    sensor_actor.listen(
                        lambda d, lbl=rsu_label, st=sensor_type_key, sn=sensor_name_in_file: manager.sensor_callback(d,sensor_queue,lbl, st,sn))
                    print(f"    Attached {sensor_type_key} to {rsu_label}")

    expected_sensor_count_per_frame = actual_sensors_listening

    world_frame = -1
    frames_processed = 0
    if spawn_center:
        spectator = world.get_spectator()
        spectator_location = carla.Location(spawn_center.x, spawn_center.y, spawn_center.z + spectator_height)
        spectator.set_transform(carla.Transform(spectator_location, carla.Rotation(pitch=-90)))

    if expected_sensor_count_per_frame > 0 or num_veh_to_spawn == 0:
        world.tick()
        time.sleep(0.5)

    num_warmup_ticks = int(frame_rate * sim_config['warmup_seconds'])
    if num_veh_to_spawn > 0:
        print(f"\n--- Performing {num_warmup_ticks} warm-up ticks ---")
        for i in range(num_warmup_ticks): world.tick()
        print("--- Warm-up complete ---")
    is_warming_up = False
    manager.set_warming_up(False)

    if expected_sensor_count_per_frame > 0:
        while frames_processed < MAX_FRAMES_TO_RUN:
            tick_start_time = time.time()
            current_frame_id = world.tick()

            print(f"\n--- Frame {current_frame_id} (Processed {frames_processed}/{MAX_FRAMES_TO_RUN}) ---")
            received_this_frame_count = 0
            frame_sensor_data_received = {}
            try:
                while received_this_frame_count < expected_sensor_count_per_frame:
                    s_data = sensor_queue.get(True, queue_timeout)
                    s_frame, s_actor_label, s_name, s_saved_submitted = s_data
                    if s_frame < current_frame_id: continue
                    if s_frame > current_frame_id: print(
                        f"  WARN: Future data F={s_frame} (Curr:{current_frame_id})"); continue
                    if s_frame == current_frame_id:
                        if s_actor_label not in frame_sensor_data_received: frame_sensor_data_received[
                            s_actor_label] = set()
                        if s_name not in frame_sensor_data_received[s_actor_label]:
                            frame_sensor_data_received[s_actor_label].add(s_name);
                            received_this_frame_count += 1

                print(f"  All {expected_sensor_count_per_frame} sensor confirmations received.")
                
                # Update localization for each CAV
                for cav_label, loc_manager in localization_managers.items():
                    if loc_manager is not None:
                        # Get IMU data for this frame
                        imu_data = manager.imu_data_latest.get(cav_label, {}).get(current_frame_id)
                        if imu_data:
                            loc_manager.set_imu_data(imu_data)
                        # Perform localization
                        loc_manager.localize()
                
                # Dumps data for individual actors (CAV, RSU)
                manager.yaml_dumper(current_frame_id, world, cav_vehicles_dict, sensors_dict,
                                     rsu_actors_dict, rsu_sensors_dict, all_spawned_vehicle_actors,
                                     localization_managers)

                # Dumps global data for the entire scene
                manager.scene_dumper(current_frame_id, all_spawned_vehicle_actors, scenes_output_dir)

                frames_processed += 1

            except Empty:
                print(f"Error: Timeout ({queue_timeout}s) waiting sensor data frame {current_frame_id}.")
                print(f"  Received {received_this_frame_count}/{expected_sensor_count_per_frame} before timeout.");
                print(f"  Data received: {frame_sensor_data_received}")
                break
            except Exception as e:
                print(f"Error in main loop frame {current_frame_id}: {e}");
                traceback.print_exc();
                break
            print(f"  Frame {current_frame_id} processed in {time.time() - tick_start_time:.3f}s.")
    else:
        print("Skipping simulation loop as no sensors are expected.")

    print('\n--- Final Cleanup ---')
    if 'world' in locals() and world and 'original_settings' in locals() and original_settings is not None:
        world.apply_settings(original_settings)
    if 'traffic_manager' in locals() and traffic_manager is not None:
        traffic_manager.set_synchronous_mode(False)
    if 'sensor_list_for_callback' in locals():
        for sensor in sensor_list_for_callback:
            if sensor and hasattr(sensor, 'is_listening') and sensor.is_listening: sensor.stop()
    if 'manager' in locals() and manager:
        manager.shutdown()
    # Destroy localization managers
    if 'localization_managers' in locals():
        for cav_label, loc_manager in localization_managers.items():
            if loc_manager is not None:
                try:
                    loc_manager.destroy()
                except Exception as e:
                    print(f"Warning: Error destroying localization manager for {cav_label}: {e}")
    if 'client' in locals() and client and 'all_actors_list' in locals() and all_actors_list:
        client.apply_batch([carla.command.DestroyActor(a) for a in all_actors_list if a is not None])

    print('Cleanup done.')
