import os
import json
import yaml
import numpy as np
import open3d as o3d


def dict_representer(dumper, data):
    return dumper.represent_mapping('tag:yaml.org,2002:map', data.items())


yaml.add_representer(dict, dict_representer)


class CustomDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(CustomDumper, self).increase_indent(flow, False)


class SensorDataManager:
    def __init__(self, executor, actor_data_store):
        self.executor = executor
        self.actor_data_store = actor_data_store
        self.imu_data_latest = {}
        self.is_warming_up = True

    def set_warming_up(self, warming_up):
        self.is_warming_up = warming_up

    def write_yaml_async(self, filepath, content):
        with open(filepath, 'w') as f:
            f.write(content)

    def lidar_dumper(self, sensor_data, filepath):
        data = np.copy(np.frombuffer(sensor_data.raw_data, dtype=np.dtype('f4')))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))
        point_cloud = data

        point_xyz = point_cloud[:, :-1]
        point_intensity = point_cloud[:, -1]
        point_intensity = np.c_[
            point_intensity,
            np.zeros_like(point_intensity),
            np.zeros_like(point_intensity)
        ]

        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(point_xyz)
        o3d_pcd.colors = o3d.utility.Vector3dVector(point_intensity)

        o3d.io.write_point_cloud(filepath,
                                 pointcloud=o3d_pcd,
                                 write_ascii=True)

    def radar_dumper(self, sensor_data, filepath):
        detections = []
        for detection in sensor_data:
            detections.append({
                'velocity': detection.velocity,
                'azimuth': detection.azimuth,
                'altitude': detection.altitude,
                'depth': detection.depth
            })

        with open(filepath, 'w') as f:
            json.dump(detections, f, indent=4)

    def sensor_callback(self, sensor_data, sensor_queue, actor_label, sensor_type, sensor_name):
        if self.is_warming_up:
            return
        frame_id = sensor_data.frame
        save_dir = self.actor_data_store.get(actor_label, {}).get("save_dir", None)
        processed_successfully = False

        if sensor_type == 'camera':
            filename = f"{frame_id:06d}_{sensor_name}.png"
            filepath = os.path.join(save_dir, filename)
            self.executor.submit(sensor_data.save_to_disk, filepath)
            processed_successfully = True
        elif sensor_type == 'depth_camera':
            filename = f"{frame_id:06d}_{sensor_name}.png"
            filepath = os.path.join(save_dir, filename)
            import carla  # local import to avoid hard dependency on module import
            self.executor.submit(sensor_data.save_to_disk, filepath, carla.ColorConverter.LogarithmicDepth)
            processed_successfully = True
        elif sensor_type == 'lidar':
            filename = f"{frame_id:06d}.pcd"
            filepath = os.path.join(save_dir, filename)
            self.executor.submit(self.lidar_dumper, sensor_data, filepath)
            processed_successfully = True
        elif sensor_type == 'semantic_lidar':
            processed_successfully = True
        elif sensor_type == 'radar':
            filename = f"{frame_id:06d}.json"
            filepath = os.path.join(save_dir, filename)
            self.executor.submit(self.radar_dumper, sensor_data, filepath)
            processed_successfully = True
        elif sensor_type == 'imu':
            if actor_label not in self.imu_data_latest:
                self.imu_data_latest[actor_label] = {}

            self.imu_data_latest[actor_label][frame_id] = {
                'accelerometer': {'x': sensor_data.accelerometer.x, 'y': sensor_data.accelerometer.y,
                                  'z': sensor_data.accelerometer.z},
                'gyroscope': {'x': sensor_data.gyroscope.x, 'y': sensor_data.gyroscope.y, 'z': sensor_data.gyroscope.z},
                'compass': sensor_data.compass,
            }
            processed_successfully = True
            frames_to_keep = sorted(self.imu_data_latest[actor_label].keys())[-3:]
            for f_id_key in list(self.imu_data_latest[actor_label].keys()):
                if f_id_key not in frames_to_keep:
                    del self.imu_data_latest[actor_label][f_id_key]
        else:
            print(f"Warning CB: Unknown sensor type '{sensor_type}' for {actor_label} {sensor_name}")

        sensor_queue.put((frame_id, actor_label, sensor_name, processed_successfully))

    def yaml_dumper(self, frame_id, world, cav_vehicles_dict, sensors_dict, rsu_actors_dict, rsu_sensors_dict,
                    all_spawned_vehicle_actors):
        for cav_label, cav_vehicle in cav_vehicles_dict.items():
            if cav_vehicle is None or not cav_vehicle.is_alive:
                continue
            save_dir = self.actor_data_store.get(cav_label, {}).get("save_dir", None)
            if not save_dir:
                continue

            vehicles_data_formatted = {}
            perception_range = 50.0
            ego_location = cav_vehicle.get_location()

            for veh in all_spawned_vehicle_actors:
                if veh.id == cav_vehicle.id or ego_location.distance(veh.get_location()) > perception_range:
                    continue

                veh_pos = veh.get_transform()
                veh_bbx = veh.bounding_box
                veh_speed_vec = veh.get_velocity()
                veh_speed = float(np.sqrt(veh_speed_vec.x ** 2 + veh_speed_vec.y ** 2 + veh_speed_vec.z ** 2))

                bp_id = veh.type_id
                color = veh.attributes.get('color', '255,255,255')

                vehicles_data_formatted[veh.id] = {
                    'bp_id': bp_id,
                    'color': color,
                    "location": [veh_pos.location.x, veh_pos.location.y, veh_pos.location.z],
                    "center": [veh_bbx.location.x, veh_bbx.location.y, veh_bbx.location.z],
                    "angle": [veh_pos.rotation.roll, veh_pos.rotation.yaw, veh_pos.rotation.pitch],
                    "extent": [veh_bbx.extent.x, veh_bbx.extent.y, veh_bbx.extent.z],
                    "speed": veh_speed
                }

            data_dict = {
                'actor': cav_label,
                'frame': frame_id,
                'sensors': {'cameras': {}},
                'vehicles': vehicles_data_formatted
            }

            vehicle_tf = cav_vehicle.get_transform()
            vel = cav_vehicle.get_velocity()
            data_dict['sensors']['vehicle_speed'] = {'speed': {'x': vel.x, 'y': vel.y, 'z': vel.z}}
            data_dict['sensors']['vehicle_pose'] = {
                'location': {'x': vehicle_tf.location.x, 'y': vehicle_tf.location.y, 'z': vehicle_tf.location.z},
                'rotation': {'pitch': vehicle_tf.rotation.pitch, 'roll': vehicle_tf.rotation.roll,
                             'yaw': vehicle_tf.rotation.yaw}
            }

            cav_sensors = sensors_dict.get(cav_label, {})
            for sensor_name, sensor_actor in cav_sensors.items():
                if sensor_name == 'cameras' or sensor_name == 'depth_cameras':
                    data_dict['sensors'][sensor_name] = {}
                    for cam_name, cam_actor in sensor_actor.items():
                        if cam_actor and cam_actor.is_alive:
                            cam_tf = cam_actor.get_transform()
                            data_dict['sensors'][sensor_name][cam_name] = {
                                'location': {'x': cam_tf.location.x, 'y': cam_tf.location.y, 'z': cam_tf.location.z},
                                'rotation': {'pitch': cam_tf.rotation.pitch, 'roll': cam_tf.rotation.roll,
                                             'yaw': cam_tf.rotation.yaw}
                            }
                elif sensor_actor and sensor_actor.is_alive:
                    sensor_tf = sensor_actor.get_transform()
                    data_dict['sensors'][f'{sensor_name}_pose'] = {
                        'location': {'x': sensor_tf.location.x, 'y': sensor_tf.location.y, 'z': sensor_tf.location.z},
                        'rotation': {'pitch': sensor_tf.rotation.pitch, 'roll': sensor_tf.rotation.roll,
                                     'yaw': sensor_tf.rotation.yaw}
                    }

            imu_measurements = self.imu_data_latest.get(cav_label, {}).get(frame_id)
            if imu_measurements:
                data_dict['sensors']['imu_measurement'] = imu_measurements
            else:
                data_dict['sensors']['imu_measurement'] = {
                    'accelerometer': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                    'gyroscope': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                    'compass': 0.0,
                    'data_missing': True
                }

            yaml_filename = f"{frame_id:06d}.yaml"
            yaml_path = os.path.join(save_dir, yaml_filename)
            yaml_content = yaml.dump(data_dict, Dumper=CustomDumper, default_flow_style=False, sort_keys=False)
            self.executor.submit(self.write_yaml_async, yaml_path, yaml_content)

        all_vehicle_data_for_rsu = []
        for veh in all_spawned_vehicle_actors:
            if not veh or not veh.is_alive:
                continue
            veh_tf = veh.get_transform()
            all_vehicle_data_for_rsu.append({
                'id': veh.id,
                'name': veh.type_id,
                'pose': {
                    'location': {'x': veh_tf.location.x, 'y': veh_tf.location.y, 'z': veh_tf.location.z},
                    'rotation': {'pitch': veh_tf.rotation.pitch, 'roll': veh_tf.rotation.roll,
                                 'yaw': veh_tf.rotation.yaw}
                }
            })

        for rsu_label, rsu_anchor in rsu_actors_dict.items():
            if rsu_anchor is None or not rsu_anchor.is_alive:
                continue
            save_dir = self.actor_data_store.get(rsu_label, {}).get("save_dir", None)
            if not save_dir:
                continue

            data_dict = {
                'actor': rsu_label,
                'frame': frame_id,
                'sensors': {'cameras': {}, 'depth_cameras': {}},
                'vehicles': all_vehicle_data_for_rsu
            }

            rsu_tf = rsu_anchor.get_transform()
            data_dict['sensors']['rsu_pose'] = {
                'location': {'x': rsu_tf.location.x, 'y': rsu_tf.location.y, 'z': rsu_tf.location.z},
                'rotation': {'pitch': rsu_tf.rotation.pitch, 'roll': rsu_tf.rotation.roll,
                             'yaw': rsu_tf.rotation.yaw}
            }

            current_rsu_sensors = rsu_sensors_dict.get(rsu_label, {})
            if 'lidar' in current_rsu_sensors and current_rsu_sensors['lidar'] and current_rsu_sensors['lidar'].is_alive:
                lidar_tf = current_rsu_sensors['lidar'].get_transform()
                data_dict['sensors']['lidar_pose'] = {
                    'location': {'x': lidar_tf.location.x, 'y': lidar_tf.location.y, 'z': lidar_tf.location.z},
                    'rotation': {'pitch': lidar_tf.rotation.pitch, 'roll': lidar_tf.rotation.roll,
                                 'yaw': lidar_tf.rotation.yaw}
                }
            if 'radar' in current_rsu_sensors and current_rsu_sensors['radar'] and current_rsu_sensors['radar'].is_alive:
                radar_tf = current_rsu_sensors['radar'].get_transform()
                data_dict['sensors']['radar_pose'] = {
                    'location': {'x': radar_tf.location.x, 'y': radar_tf.location.y, 'z': radar_tf.location.z},
                    'rotation': {'pitch': radar_tf.rotation.pitch, 'roll': radar_tf.rotation.roll,
                                 'yaw': radar_tf.rotation.yaw}
                }
            for cam_name, cam_actor in current_rsu_sensors.get('cameras', {}).items():
                if cam_actor and cam_actor.is_alive:
                    cam_tf = cam_actor.get_transform()
                    data_dict['sensors']['cameras'][cam_name] = {
                        'location': {'x': cam_tf.location.x, 'y': cam_tf.location.y, 'z': cam_tf.location.z},
                        'rotation': {'pitch': cam_tf.rotation.pitch, 'roll': cam_tf.rotation.roll,
                                     'yaw': cam_tf.rotation.yaw}
                    }
            for cam_name, cam_actor in current_rsu_sensors.get('depth_cameras', {}).items():
                if cam_actor and cam_actor.is_alive:
                    cam_tf = cam_actor.get_transform()
                    data_dict['sensors']['depth_cameras'][cam_name] = {
                        'location': {'x': cam_tf.location.x, 'y': cam_tf.location.y, 'z': cam_tf.location.z},
                        'rotation': {'pitch': cam_tf.rotation.pitch, 'roll': cam_tf.rotation.roll,
                                     'yaw': cam_tf.rotation.yaw}
                    }

            yaml_filename = f"{frame_id:06d}.yaml"
            yaml_path = os.path.join(save_dir, yaml_filename)
            yaml_content = yaml.dump(data_dict, Dumper=CustomDumper, default_flow_style=False, sort_keys=False)
            self.executor.submit(self.write_yaml_async, yaml_path, yaml_content)

    def scene_dumper(self, frame_id, all_spawned_vehicle_actors, save_dir):
        all_vehicle_data_for_yaml = []
        for veh in all_spawned_vehicle_actors:
            if not veh or not veh.is_alive:
                continue

            veh_tf = veh.get_transform()
            veh_vel = veh.get_velocity()
            speed = float(np.sqrt(veh_vel.x ** 2 + veh_vel.y ** 2 + veh_vel.z ** 2))

            all_vehicle_data_for_yaml.append({
                'id': veh.id,
                'bp_id': veh.type_id,
                'speed': speed,
                'pose': {
                    'location': {'x': veh_tf.location.x, 'y': veh_tf.location.y, 'z': veh_tf.location.z},
                    'rotation': {'pitch': veh_tf.rotation.pitch, 'roll': veh_tf.rotation.roll, 'yaw': veh_tf.rotation.yaw}
                }
            })

        data_dict = {
            'frame': frame_id,
            'vehicles': all_vehicle_data_for_yaml
        }

        yaml_filename = f"{frame_id:06d}.yaml"
        yaml_path = os.path.join(save_dir, yaml_filename)
        yaml_content = yaml.dump(data_dict, Dumper=CustomDumper, default_flow_style=False, sort_keys=False)
        self.executor.submit(self.write_yaml_async, yaml_path, yaml_content)

    def shutdown(self):
        if self.executor:
            self.executor.shutdown(wait=True)


