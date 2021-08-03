"""
    The semantic features module contains several semantic features like
    intersection of trajectories, stop and stay point detection. Moreover,
    features like distance from Point of Interests, water bodies and other
    demographic features related to the trajectory data are calculated. The
    demographic features are extracted with the help of the python osmnx
    library.

    | Authors: Yaksh J Haranwala, Salman Haidri
    | Date: August 3rd, 2021.
    | Version: 0.2 Beta

"""
import math

import osmnx as ox
import pandas as pd

from Nummobility.core.TrajectoryDF import NumPandasTraj
from Nummobility.features.spatial_features import SpatialFeatures
import Nummobility.utilities.constants as const


class SemanticFeatures:
    @staticmethod
    def nearest_bank_detection(dataframe: NumPandasTraj, dist_threshold=1000):
        """
            Take a single trajectory id and detect all the banks it passes by
            within the dist_threshold and extract their coordinates.
        """
        tags = {'amenity': ['atm', 'banks']}

        start = dataframe[const.LAT][0], dataframe[const.LONG][0]
        lst = [start]
        # First, we generate the distance of all the points of the trajectory from the
        # first point of the trajectory and then lets divide it in n equal parts.
        df = SpatialFeatures.create_distance_from_given_point_column(dataframe=dataframe,
                                                                     coordinates=start)

        len_traj = SpatialFeatures.get_distance_travelled_by_traj_id(dataframe,
                                                                     dataframe.reset_index()[const.TRAJECTORY_ID][0])

        for i in range(len(df)):
            if df[f'Distance_from_{start}'][i] > dist_threshold:
                pt = dataframe[const.LAT][i], dataframe[const.LONG][i]
                lst.append(pt)
                dist_threshold += dist_threshold

        lst.append((dataframe[const.LAT][len(df) - 1], dataframe[const.LONG][len(df) - 1]))
        return lst

    @staticmethod
    def small_dist(dataframe, start):
        pass

    @staticmethod
    def distance_from_nearby_hotels():
        pass

    @staticmethod
    def distance_from_nearby_hopitals():
        pass

    @staticmethod
    def distance_from_nearby_waterbody():
        pass

    @staticmethod
    def trajectory_crossing_paths():
        pass

    @staticmethod
    def points_inside_polygon():
        pass
