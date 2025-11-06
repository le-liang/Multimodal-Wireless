# -*- coding: utf-8 -*-
# Author: Tianhao Mao <tianhao@seu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import carla
import math

def get_location(location_dict):
    """Converts a dictionary from YAML to a carla.Location object."""
    return carla.Location(
        x=location_dict.get('x', 0.0),
        y=location_dict.get('y', 0.0),
        z=location_dict.get('z', 0.0)
    )

def get_transform(transform_dict):
    """Converts a dictionary from YAML to a carla.Transform object."""
    location = get_location(transform_dict.get('location', {}))
    rotation_dict = transform_dict.get('rotation', {})
    rotation = carla.Rotation(
        pitch=rotation_dict.get('pitch', 0.0),
        yaw=rotation_dict.get('yaw', 0.0),
        roll=rotation_dict.get('roll', 0.0)
    )
    return carla.Transform(location, rotation)

def distt(location1, location2):
    return math.sqrt(
        (location1.x - location2.x) ** 2 + (location1.y - location2.y) ** 2 + (location1.z - location2.z) ** 2)
