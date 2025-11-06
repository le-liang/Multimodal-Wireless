# -*- coding: utf-8 -*-
"""
Localization module based on teacherscript.py and provided scripts
"""
import weakref
import carla
import numpy as np
from collections import deque
from .loc_utils import geo_to_transform, KalmanFilter, LocDebugHelper


def get_speed(vehicle):
    """Get vehicle speed in m/s."""
    velocity = vehicle.get_velocity()
    return np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


class GnssSensor(object):
    """
    GNSS sensor module for GPS data collection.
    
    Parameters
    ----------
    vehicle : carla.Vehicle
        The carla.Vehicle to attach the GNSS sensor to.
    config : dict
        Configuration dictionary with noise parameters.
    """
    
    def __init__(self, vehicle, config):
        world = vehicle.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        
        # Set noise attributes
        blueprint.set_attribute('noise_alt_stddev', str(config.get('noise_alt_stddev', '0.0')))
        blueprint.set_attribute('noise_lat_stddev', str(config.get('noise_lat_stddev', '0.0')))
        blueprint.set_attribute('noise_lon_stddev', str(config.get('noise_lon_stddev', '0.0')))
        
        # Spawn the sensor
        self.sensor = world.spawn_actor(
            blueprint,
            carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0)),
            attach_to=vehicle,
            attachment_type=carla.AttachmentType.Rigid
        )
        
        # Initialize data
        self.lat, self.lon, self.alt, self.timestamp = 0.0, 0.0, 0.0, 0.0
        
        # Create weak reference to avoid circular reference
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: GnssSensor._on_gnss_event(weak_self, event)
        )
    
    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS callback that stores the current geo location."""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude
        self.alt = event.altitude
        self.timestamp = event.timestamp


class LocalizationManager(object):
    """
    Localization manager that fuses GNSS and IMU data using Kalman filtering.
    
    Parameters
    ----------
    vehicle : carla.Vehicle
        The vehicle to track.
    config_yaml : dict
        Configuration dictionary.
    carla_map : carla.Map
        The CARLA map for coordinate conversion.
    """
    
    def __init__(self, vehicle, config_yaml, carla_map):
        self.vehicle = vehicle
        self.activate = config_yaml.get('activate', True)
        self.map = carla_map
        self.geo_ref = self.map.transform_to_geolocation(carla.Location(x=0, y=0, z=0))
        
        # Position and speed
        self._ego_pos = None
        self._speed = 0
        
        # History
        self._ego_pos_history = deque(maxlen=100)
        self._timestamp_history = deque(maxlen=100)
        
        # Sensors
        self.gnss = GnssSensor(vehicle, config_yaml['gnss'])
        self.imu = None  # Will be set externally
        
        # Noise parameters
        self.heading_noise_std = config_yaml['gnss'].get('heading_direction_stddev', 0.0)
        self.speed_noise_std = config_yaml['gnss'].get('speed_stddev', 0.0)
        
        self.dt = config_yaml.get('dt', 0.01)  # Default 100fps
        
        # Kalman Filter
        self.kf = KalmanFilter(self.dt)
        
        # DebugHelper
        debug_helper_config = config_yaml.get('debug_helper', {})
        if not debug_helper_config:
            debug_helper_config = {
                'show_animation': False,
                'x_scale': 1.0,
                'y_scale': 1.0
            }
        self.debug_helper = LocDebugHelper(debug_helper_config, self.vehicle.id)
        
        # Storage for GPS noisy position
        self.gps_noisy_pos = None
        self.predicted_ego_pos = None
        self.true_ego_pos = None
    
    def set_imu_data(self, imu_data):
        """Set IMU data from external sensor."""
        self.imu = imu_data
    
    def localize(self):
        """Perform localization using GNSS and IMU fusion."""
        if not self.activate:
            self._ego_pos = self.vehicle.get_transform()
            self._speed = get_speed(self.vehicle)
            true_location = self._ego_pos.location
            true_rotation = self._ego_pos.rotation
            self.true_ego_pos = {
                'location': {'x': float(true_location.x), 'y': float(true_location.y), 'z': float(true_location.z)},
                'rotation': {'pitch': float(true_rotation.pitch), 'roll': float(true_rotation.roll), 'yaw': float(true_rotation.yaw)}
            }
            self.gps_noisy_pos = None
            self.predicted_ego_pos = None
            return
        
        speed_true = get_speed(self.vehicle)
        speed_noise = self.add_speed_noise(speed_true)
        
        # Convert GNSS coordinates to ENU (Unreal coordinate system)
        x, y, z = geo_to_transform(
            self.gnss.lat,
            self.gnss.lon,
            self.gnss.alt,
            self.geo_ref.latitude,
            self.geo_ref.longitude,
            0.0
        )
        
        # Store GPS noisy position
        self.gps_noisy_pos = {
            'location': {'x': float(x), 'y': float(y), 'z': float(z)}
        }
        
        # Get true position for ground truth
        true_location = self.vehicle.get_transform().location
        true_rotation = self.vehicle.get_transform().rotation
        self.true_ego_pos = {
            'location': {'x': float(true_location.x), 'y': float(true_location.y), 'z': float(true_location.z)},
            'rotation': {'pitch': float(true_rotation.pitch), 'roll': float(true_rotation.roll), 'yaw': float(true_rotation.yaw)}
        }
        
        # Add noise to heading direction
        heading_angle = self.add_heading_direction_noise(true_rotation.yaw)
        
        # Initialize or update Kalman filter
        if len(self._ego_pos_history) == 0:
            x_kf, y_kf, heading_angle_kf = x, y, heading_angle
            self._speed = speed_true
            self.kf.run_step_init(x, y, np.deg2rad(heading_angle), self._speed / 3.6)
        else:
            # Get IMU angular velocity
            angular_velocity_z = 0.0
            if self.imu and 'gyroscope' in self.imu:
                angular_velocity_z = self.imu['gyroscope'].get('z', 0.0)
            
            x_kf, y_kf, heading_angle_kf_rad, speed_kf = self.kf.run_step(
                x, y, np.deg2rad(heading_angle),
                speed_noise / 3.6,
                angular_velocity_z
            )
            self._speed = speed_kf * 3.6
            heading_angle_kf = np.rad2deg(heading_angle_kf_rad)
        
        # Store predicted ego position (filtered)
        self.predicted_ego_pos = {
            'location': {'x': float(x_kf), 'y': float(y_kf), 'z': float(z)},
            'rotation': {'pitch': 0, 'roll': 0, 'yaw': float(heading_angle_kf)}
        }
        
        # Add data to debug helper
        self.debug_helper.run_step(
            x, y, heading_angle, speed_noise,
            x_kf, y_kf, heading_angle_kf, self._speed,
            true_location.x, true_location.y, true_rotation.yaw, speed_true
        )
        
        # Final pose
        self._ego_pos = carla.Transform(
            carla.Location(x=x_kf, y=y_kf, z=z),
            carla.Rotation(pitch=0, yaw=heading_angle_kf, roll=0)
        )
        
        # Save history
        self._ego_pos_history.append(self._ego_pos)
        self._timestamp_history.append(self.gnss.timestamp)
    
    def add_heading_direction_noise(self, heading_direction):
        """Add synthetic noise to heading direction."""
        return heading_direction + np.random.normal(0, self.heading_noise_std)
    
    def add_speed_noise(self, speed):
        """Add gaussian white noise to the current speed."""
        return speed + np.random.normal(0, self.speed_noise_std)
    
    def get_ego_pos(self):
        """Retrieve ego vehicle position."""
        return self._ego_pos
    
    def get_ego_spd(self):
        """Retrieve ego vehicle speed."""
        return self._speed
    
    def destroy(self):
        """Destroy the sensors."""
        if self.gnss and self.gnss.sensor:
            self.gnss.sensor.destroy()
