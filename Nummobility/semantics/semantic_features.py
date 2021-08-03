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
from Nummobility.utilities.DistanceCalculator import FormulaLog

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
        df = SpatialFeatures.create_distance_from_start_column(dataframe)

        for i in range(len(df)):
            if df[f'Distance_start_to_curr'][i] > dist_threshold:
                pt = dataframe[const.LAT][i], dataframe[const.LONG][i]
                lst.append(pt)
                dist_threshold += dist_threshold

        lst.append((dataframe[const.LAT][len(df) - 1], dataframe[const.LONG][len(df) - 1]))
        return lst

    @staticmethod
    def nearest_bank_alt(df: NumPandasTraj, dist=1000):
        start = df[const.LAT][0], df[const.LONG][0]
        dead_start = df[const.LAT][0], df[const.LONG][0]
        lst = list()

        for i in range(1, len(df)):
            pt = df[const.LAT][i], df[const.LONG][i]
            if FormulaLog.haversine_distance(start[0], start[1], pt[0], pt[1]) > dist:
                if FormulaLog.haversine_distance(dead_start[0], dead_start[1], pt[0], pt[1]) > dist:
                    lst.append(pt)
                    start = pt

        tags = {'amenity': ['atm', 'banks']}



        return lst

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
