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
import itertools
import math
import multiprocessing
import os
from json import JSONDecodeError
from typing import Union

#import osmnx as ox
import pandas as pd
import psutil
import numpy as np

#from geopandas import GeoDataFrame
from Nummobility.core.TrajectoryDF import NumPandasTraj
from Nummobility.features.spatial_features import SpatialFeatures
from Nummobility.utilities.DistanceCalculator import FormulaLog
from Nummobility.semantics.helpers import SemanticHelpers as helpers

import Nummobility.utilities.constants as const

NUM_CPU = math.ceil((len(os.sched_getaffinity(0)) if os.name == 'posix' else psutil.cpu_count()) * 2 / 3)


class SemanticFeatures:
    @staticmethod
    def nearest_bank_detection(dataframe: NumPandasTraj, dist_threshold: int = 1000):
        # splitting the dataframe according to trajectory id.
        ids_ = list(dataframe.reset_index()[const.TRAJECTORY_ID].value_counts().keys())
        df_chunks = [dataframe.reset_index().loc[dataframe.reset_index()[const.TRAJECTORY_ID] == ids_[i]]
                     for i in range(len(ids_))]

        # Here, create 2/3rds number of processes as there are in the system. Some CPUs are
        # kept free at all times in order to not block up the system.
        # (Note: The blocking of system is mostly prevalent in Windows and does not happen very often
        # in Linux. However, out of caution some CPUs are kept free regardless of the system.)
        multi_pool = multiprocessing.Pool(NUM_CPU)
        result = multi_pool.starmap(helpers.bank_within_dist_helper, zip(df_chunks, itertools.repeat(dist_threshold)))
        multi_pool.close()
        multi_pool.join()

        # merge the smaller pieces and then return the dataframe converted to NumPandasTraj.
        return NumPandasTraj(pd.concat(result), const.LAT, const.LONG,
                             const.DateTime, const.TRAJECTORY_ID)

    # @staticmethod
    # def bank_within_threshold(dataframe: NumPandasTraj, poi: GeoDataFrame, dist_threshold: float = 1000):
    #     """
    #         For all the points in the data, check whether there is a
    #         bank within the threshold given by the user.
    #
    #         Parameters
    #         ----------
    #             dataframe: NumPandasTraj
    #                 The dataframe containing the Trajectory Data.
    #             poi: GeoDataFrame
    #                 The GeoDataframe containing the point of interests.
    #             dist_threshold: float
    #                 The range within which banks are to be checked.
    #
    #         Returns
    #         -------
    #             NumPandasTraj:
    #                 The dataframe containing the column indicating the presence
    #                 of a bank within the given threshold.
    #     """
    #     # splitting the dataframe according to trajectory id.
    #     ids_ = list(dataframe.reset_index()[const.TRAJECTORY_ID].value_counts().keys())
    #     df_chunks = [dataframe.reset_index().loc[dataframe.reset_index()[const.TRAJECTORY_ID] == ids_[i]]
    #                  for i in range(len(ids_))]
    #
    #     # Here, create 2/3rds number of processes as there are in the system. Some CPUs are
    #     # kept free at all times in order to not block up the system.
    #     # (Note: The blocking of system is mostly prevalent in Windows and does not happen very often
    #     # in Linux. However, out of caution some CPUs are kept free regardless of the system.)
    #     multi_pool = multiprocessing.Pool(NUM_CPU)
    #     result = multi_pool.starmap(helpers.bank_within_dist_helper,
    #                                 zip(df_chunks, itertools.repeat(poi), itertools.repeat(dist_threshold)))
    #     multi_pool.close()
    #     multi_pool.join()
    #
    #     # merge the smaller pieces and then return the dataframe converted to NumPandasTraj.
    #     return NumPandasTraj(pd.concat(result), const.LAT, const.LONG,
    #                          const.DateTime, const.TRAJECTORY_ID)
    #
    # @staticmethod
    # def nearest_bank_from_point(coords: tuple, dist_threshold, tags: dict):
    #     """
    #         Given a coordinate point and a distance threshold, find
    #         the bank which is nearest to the point.
    #
    #         Parameter
    #         ---------
    #             coords: tuple
    #                 The point near which the bank is to be found.
    #             dist_threshold:
    #                 The maximum distance from the point within which
    #                 the distance is to be calculated.
    #
    #         Returns
    #         -------
    #             pandas.core.dataframe.DataFrame:
    #                 A pandas DF containing the info about the nearest bank from
    #                 the given point.
    #     """
    #     try:
    #         poi = ox.geometries_from_point(center_point=coords,
    #                                        dist=dist_threshold,
    #                                        tags=tags)
    #
    #         if len(poi) > 0:
    #             poi = poi.reset_index().loc[poi.reset_index()['element_type'] == 'node']
    #
    #             lat = list(poi['geometry'].apply(lambda p: p.y))
    #             lon = list(poi['geometry'].apply(lambda p: p.x))
    #
    #             dists = []
    #             for i in range(len(lat)):
    #                 dists.append(FormulaLog.haversine_distance(coords[0], coords[1], lat[i], lon[i]))
    #
    #             poi[f'Distance_from_{coords}'] = dists
    #
    #             return poi.loc[poi[f'Distance_from_{coords}'] ==
    #                            poi[f'Distance_from_{coords}'].min()].reset_index().drop(columns=['element_type', 'index'])
    #         else:
    #             return []
    #
    #     except JSONDecodeError:
    #         raise ValueError("The tags provided are invalid. Please check your tags and try again.")


    @staticmethod
    def check_intersect(df: NumPandasTraj, geo_layers: Union[pd.DataFrame]):
        df_lat = np.array(df.reset_index()[const.LAT].values)
        df_lon = np.array(df.reset_index()[const.LONG].values)
        gl_lat = np.array(geo_layers.reset_index()[const.LAT].values)
        gl_lon = np.array(geo_layers.reset_index()[const.LONG].values)

        # 2d array to store the displacement values
        distances = np.zeros((len(df_lat), len(gl_lat)))
        for i in range(len(df_lat)):
            for j in range(len(gl_lat)):
                # Check if the copordinate difference is +ve or -nve if +ve then find the distance else add a negative sign to it
                if df_lat[i] - gl_lat[j] > 0 or df_lon[i] - gl_lon[j] > 0:
                    distances[i][j] = (((df_lat[i] - gl_lat[j]) ** 2 + (df_lon[i] - gl_lon[j]) ** 2) ** 0.5)
                else:
                    distances[i][j] = (-(((df_lat[i] - gl_lat[j]) ** 2 + (df_lon[i] - gl_lon[j]) ** 2) ** 0.5))

        #distances = ((((df_lat[:, None] - gl_lat[None, :]) ** 2 + (df_lon[:, None] - gl_lon[None, :]) ** 2) ** 0.5) if ((df_lat[:, None] - gl_lat[None, :]) > 0 or (df_lon[:, None] - gl_lon[None, :]) > 0) else -(((df_lat[:, None] - gl_lat[None, :]) ** 2 + (df_lon[:, None] - gl_lon[None, :]) ** 2) ** 0.5))
        print(distances)

        # If any of the value in a row is negative then intersection will store true else false
        threshold = 0.0
        intersections = (distances < 0.0).any(axis=1)
        #print(intersections)
        return intersections

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
