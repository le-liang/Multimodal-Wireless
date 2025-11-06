# -*- coding: utf-8 -*-
# Author: Tianhao Mao <tianhao@seu.edu.cn>, OpenCOOD
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
Localization utilities combining coordinate_transform, kalman_filter, and localization_debug_helper
"""
import math
import numpy as np
import matplotlib

import matplotlib.pyplot as plt


def geo_to_transform(lat, lon, alt, lat_0, lon_0, alt_0):
    """
    Convert WG84 to ENU. The origin of the ENU should pass the geo reference.
    Note this function is a writen by reversing the
    official API transform_to_geo.

    Parameters
    ----------
    lat : float
        current latitude.

    lon : float
        current longitude.

    alt : float
        current altitude.

    lat_0 : float)
        geo_ref latitude.

    lon_0 : float
        geo_ref longitude.

    alt_0 : float
        geo_ref altitude.

    Returns
    -------
    x : float
        The transformed x coordinate.

    y : float
        The transformed y coordinate.

    z : float
        The transformed z coordinate.
    """
    EARTH_RADIUS_EQUA = 6378137.0
    scale = np.cos(np.deg2rad(lat_0))

    mx = lon * np.pi * EARTH_RADIUS_EQUA * scale / 180
    mx_0 = scale * np.deg2rad(lon_0) * EARTH_RADIUS_EQUA
    x = mx - mx_0

    my = np.log(np.tan((lat + 90) * np.pi / 360)) * EARTH_RADIUS_EQUA * scale
    my_0 = scale * EARTH_RADIUS_EQUA * \
        np.log(np.tan((90 + lat_0) * np.pi / 360))
    y = -(my - my_0)

    z = alt - alt_0

    return x, y, z


class KalmanFilter(object):
    """
    Kalman Filter implementation for gps and imu.

    Parameters
    ----------
    dt : float
        The step time for kalman filter calculation.

    Attributes
    ----------
    Q : numpy.array
        predict state covariance.

    R : numpy.array
        Observation x,y position covariance.

    time_step : float
        The step time for kalman filter calculation.

    xEst : numpy.array
        Estimated x values.

    PEst : numpy.array
        The estimated P values.
    """

    def __init__(self, dt):
        self.Q = np.diag([
            0.2,  # variance of location on x-axis
            0.2,  # variance of location on y-axis
            np.deg2rad(0.1),  # variance of yaw angle
            0.001  # variance of velocity
        ]) ** 2  # predict state covariance

        # Observation x,y position covariance
        self.R = np.diag([0.5, 0.5, 0.2]) ** 2

        self.time_step = dt

        self.xEst = np.zeros((4, 1))
        self.PEst = np.eye(4)

    def motion_model(self, x, u):
        """
        Predict current position and yaw based on
        previous result (X = F * X_prev + B * u).

        Parameters
        ----------
        x : np.array
            [x_prev, y_prev, yaw_prev, v_prev], shape: (4, 1).

        u : np.array
            [v_current, imu_yaw_rate], shape:(2, 1).

        Returns
        -------
        x : np.array
            Predicted state.
        """
        F = np.array([[1.0, 0, 0, 0],
                      [0, 1.0, 0, 0],
                      [0, 0, 1.0, 0],
                      [0, 0, 0, 0]])

        B = np.array([[self.time_step * math.cos(x[2, 0]), 0],
                      [self.time_step * math.sin(x[2, 0]), 0],
                      [0.0, self.time_step],
                      [1.0, 0.0]])

        x = F @ x + B @ u

        return x

    def observation_model(self, x):
        """
        Project the state matrix to sensor measurement matrix.

        Parameters
        __________
        x : np.array
            [x, y, yaw, v], shape: (4. 1).

        Returns
        -------
        z : np.array)
            Predicted measurement.

        """
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ])

        z = H @ x

        return z

    def run_step_init(self, x, y, heading, velocity):
        """
        Initial state filling.

        Parameters
        ----------
        x : float
            The x coordinate.

        y : float
            The y coordinate.

        heading : float
            The heading direction.

        velocity : float
            The velocity speed.

        """
        self.xEst[0] = x
        self.xEst[1] = y
        self.xEst[2] = heading
        self.xEst[3] = velocity

    def run_step(self, x, y, heading, velocity, yaw_rate_imu):
        """
        Apply KF on current measurement and previous prediction.

        Parameters
        ----------
        x : float
            x(esu) coordinate from gnss sensor at current timestamp

        y : float
            y(esu) coordinate from gnss sensor at current timestamp

        heading : float
            heading direction at current timestamp.

        velocity : float
            current speed.

        yaw_rate_imu : float
            yaw rate rad/s from IMU sensor.

        Returns
        -------
        Xest : np.array
            The corrected x, y, heading, and velocity information.
        """
        # gps observation
        z = np.array([x, y, heading]).reshape(3, 1)
        # velocity and imu yaw rate
        u = np.array([velocity, yaw_rate_imu]).reshape(2, 1)

        # state prediction
        xPred = self.motion_model(self.xEst, u)
        # sensor measurement prediction
        zPred = self.observation_model(xPred)
        y = z - zPred

        # projection matrix
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ])

        # prediction matrix
        F = np.array([[1.0, 0, 0, 0],
                      [0, 1.0, 0, 0],
                      [0, 0, 1.0, 0],
                      [0, 0, 0, 0]])

        PPred = F @ self.PEst @ F.T + self.Q
        S = np.linalg.inv(H @ PPred @ H.T + self.R)
        K = PPred @ H.T @ S

        self.xEst = xPred + K @ y
        self.PEst = K @ H @ PPred

        return self.xEst[0][0], \
            self.xEst[1][0], \
            self.xEst[2][0],\
            self.xEst[3][0]


class LocDebugHelper(object):
    """
    This class aims to help users debugging their localization algorithms.
    Users can apply this class to draw the x, y coordinate
    trajectory, yaw angle and vehicle speed from GNSS raw measurements,
    Kalman filter, and the groundtruth measurements.
    Error plotting is also enabled.

    Attributes
        show_animation : boolean
            Indicator of whether to visulize animtion.
        x_scale : float
            The scale of x coordinates.
        y_scale : float
            The scale of y coordinates.
        gnss_x : list
            The list of recorded gnss x coordinates.
        gnss_y : list
            The list of recorded gnss y coordinates.
        gnss_yaw : list
            The list of recorded gnss yaw angles.
        gnss_speed : list
            The list of recorded gnss speed values.
        filter_x : list
            The list of filtered x coordinates.
        filter_y : list
            The list of filtered y coordinates.
        filter_yaw : list
            The list of filtered yaw angles.
        filter_speed : list
            The list of filtered speed values.
        gt_x : list
            The list of ground truth x coordinates.
        gt_y : list
            The list of ground truth y coordinates.
        gt_yaw : list
            The list of ground truth yaw angles.
        gt_speed : list
            The list of ground truth speed values.
        hxEst : list
            The filtered x y coordinates.
        hTrue : list
            The true x y coordinates.
        hz : list
            The gnss detected x y coordinates.
        actor_id : int
            The list of ground truth speed values.
    """

    def __init__(self, config_yaml, actor_id):

        self.show_animation = config_yaml['show_animation']
        self.x_scale = config_yaml['x_scale']
        self.y_scale = config_yaml['y_scale']

        # off-line plotting
        self.gnss_x = []
        self.gnss_y = []
        self.gnss_yaw = []
        self.gnss_spd = []

        self.filter_x = []
        self.filter_y = []
        self.filter_yaw = []
        self.filter_spd = []

        self.gt_x = []
        self.gt_y = []
        self.gt_yaw = []
        self.gt_spd = []

        # online animation
        # filtered x y coordinates
        self.hxEst = np.zeros((2, 1))
        # gt x y coordinates
        self.hTrue = np.zeros((2, 1))
        # gnss x y coordinates
        self.hz = np.zeros((2, 1))

        self.actor_id = actor_id

    def run_step(self, gnss_x, gnss_y, gnss_yaw, gnss_spd,
                 filter_x, filter_y, filter_yaw, filter_spd,
                 gt_x, gt_y, gt_yaw, gt_spd):
        """
        Run a single step for DebugHelper to save and animate(optional)
        the localization data.

        Args:
            -gnss_x (float): GNSS detected x coordinate.
            -gnss_y (float): GNSS detected y coordinate.
            -gnss_yaw (float): GNSS detected yaw angle.
            -gnss_spd (float): GNSS detected speed value.
            -filter_x (float): Filtered x coordinates.
            -filter_y (float): Filtered y coordinates.
            -filter_yaw (float): Filtered yaw angle.
            -filter_spd (float): Filtered speed value.
            -gt_x (float): The ground truth x coordinate.
            -gt_y (float): The ground truth y coordinate.
            -gt_yaw (float): The ground truth yaw angle.
            -gt_spd (float): The ground truth speed value.

        """
        self.gnss_x.append(gnss_x)
        self.gnss_y.append(gnss_y)
        self.gnss_yaw.append(gnss_yaw)
        self.gnss_spd.append(gnss_spd / 3.6)

        self.filter_x.append(filter_x)
        self.filter_y.append(filter_y)
        self.filter_yaw.append(filter_yaw)
        self.filter_spd.append(filter_spd / 3.6)

        self.gt_x.append(gt_x)
        self.gt_y.append(gt_y)
        self.gt_yaw.append(gt_yaw)
        self.gt_spd.append(gt_spd / 3.6)

        if self.show_animation:
            # call backend setting here to solve the conflict between cv2 pyqt5
            # and pyplot qtagg
            try:
                matplotlib.use('TkAgg')
            except ImportError:
                pass
            xEst = np.array([filter_x, filter_y]).reshape(2, 1)
            zTrue = np.array([gt_x, gt_y]).reshape(2, 1)
            z = np.array([gnss_x, gnss_y]).reshape(2, 1)

            self.hxEst = np.hstack((self.hxEst, xEst))
            self.hz = np.hstack((self.hz, z))
            self.hTrue = np.hstack((self.hTrue, zTrue))

            plt.cla()
            plt.title('actor id %d localization trajectory' % self.actor_id)
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event', lambda event: [
                    plt.close() if event.key == 'escape' else None])

            plt.plot(self.hTrue[0, 1:].flatten() * self.x_scale,
                     self.hTrue[1, 1:].flatten() * self.y_scale, "-b",
                     label='groundtruth')
            plt.plot(self.hz[0, 1:] *
                     self.x_scale, self.hz[1, 1:] *
                     self.y_scale, ".g", label='gnss noise data')
            plt.plot(self.hxEst[0, 1:].flatten() * self.x_scale,
                     self.hxEst[1, 1:].flatten() * self.y_scale, "-r",
                     label='kf result')

            plt.axis("equal")
            plt.grid(True)
            plt.legend()
            plt.pause(0.001)

    def evaluate(self):
        """
        Plot the localization related data points.

        Returns:
            -figures(matplotlib.pyplot.plot): The plot of
            localization related figures.
            -perform_txt(txt file): The localization related
            datas saved as text file.

        """
        figure, axis = plt.subplots(3, 2)
        figure.set_size_inches(16, 12)
        # x, y coordinates
        axis[0, 0].plot(self.gnss_x, self.gnss_y, ".g", label='gnss')
        axis[0, 0].plot(self.gt_x, self.gt_y, ".b", label='gt')
        axis[0, 0].plot(self.filter_x, self.filter_y, ".r", label='filter')
        axis[0, 0].legend()
        axis[0, 0].set_title("x-y coordinates plotting")

        # yaw angle
        axis[0, 1].plot(np.arange(len(self.gnss_yaw)),
                        self.gnss_yaw, ".g", label='gnss')
        axis[0, 1].plot(np.arange(len(self.gt_yaw)),
                        self.gt_yaw, ".b", label='gt')
        axis[0, 1].plot(np.arange(len(self.filter_yaw)),
                        self.filter_yaw, ".r", label='filter')
        axis[0, 1].legend()
        axis[0, 1].set_title("yaw angle(degree) plotting")

        # speed
        axis[1, 0].plot(np.arange(len(self.gnss_spd)),
                        self.gnss_spd, ".g", label='gnss')
        axis[1, 0].plot(np.arange(len(self.gt_spd)),
                        self.gt_spd, ".b", label='gt')
        axis[1, 0].plot(np.arange(len(self.filter_spd)),
                        self.filter_spd, ".r", label='filter')
        axis[1, 0].legend()
        axis[1, 0].set_title("speed(m/s) plotting")

        # error curve on x
        axis[1, 1].plot(np.arange(len(self.gnss_x)), np.array(
            self.gt_x) - np.array(self.gnss_x), "-g", label='gnss')
        axis[1, 1].plot(np.arange(len(self.filter_x)), np.array(
            self.gt_x) - np.array(self.filter_x), "-r", label='filter')
        axis[1, 1].legend()
        axis[1, 1].set_title("error curve on x coordinates")

        # error curve on y
        axis[2, 0].plot(np.arange(len(self.gnss_y)), np.array(
            self.gt_y) - np.array(self.gnss_y), "-g", label='gnss')
        axis[2, 0].plot(np.arange(len(self.filter_y)), np.array(
            self.gt_y) - np.array(self.filter_y), "-r", label='filter')
        axis[2, 0].legend()
        axis[2, 0].set_title("error curve on y coordinates")

        # error curve on yaw
        axis[2, 1].plot(np.arange(len(self.gnss_yaw)), np.array(
            self.gt_yaw) - np.array(self.gnss_yaw), "-g", label='gnss')
        axis[2, 1].plot(np.arange(len(self.filter_yaw)), np.array(
            self.gt_yaw) - np.array(self.filter_yaw), "-r", label='filter')
        axis[2, 1].legend()
        axis[2, 1].set_title("error curve on yaw angle")

        figure.suptitle('localization plotting of actor id %d' % self.actor_id)

        x_error_mean = np.mean(
            np.abs(
                np.array(
                    self.gt_x) -
                np.array(
                    self.gnss_x)))
        y_error_mean = np.mean(
            np.abs(
                np.array(
                    self.gt_y) -
                np.array(
                    self.gnss_y)))
        yaw_error_mean = np.mean(
            np.abs(
                np.array(
                    self.gt_yaw) -
                np.array(
                    self.gnss_yaw)))

        perform_txt = 'mean error for GNSS raw data on x-axis: %f (meter), ' \
                      'mean error for GNSS raw data on y-axis: %f (meter),' \
                      'mean error for GNSS raw data on yaw : %f (degree) \n'\
                      % (x_error_mean,
                         y_error_mean,
                         yaw_error_mean)

        x_error_mean = np.mean(
            np.abs(
                np.array(
                    self.gt_x) -
                np.array(
                    self.filter_x)))
        y_error_mean = np.mean(
            np.abs(
                np.array(
                    self.gt_y) -
                np.array(
                    self.filter_y)))
        yaw_error_mean = np.mean(
            np.abs(
                np.array(
                    self.gt_yaw) -
                np.array(
                    self.filter_yaw)))

        perform_txt += 'mean error after data fusion on x-axis: %f (meter), ' \
                       'mean error after data fusion  on y-axis: %f (meter),' \
                       'mean error after data fusion yaw : %f (degree) \n' \
                       % (x_error_mean,
                          y_error_mean,
                          yaw_error_mean)

        return figure, perform_txt
