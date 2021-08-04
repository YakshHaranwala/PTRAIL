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
import multiprocessing
import os

import osmnx as ox
import pandas as pd
import psutil

from Nummobility.core.TrajectoryDF import NumPandasTraj
from Nummobility.features.spatial_features import SpatialFeatures
from Nummobility.utilities.DistanceCalculator import FormulaLog
from Nummobility.semantics.helpers import SemanticHelpers as helpers

import Nummobility.utilities.constants as const

NUM_CPU = math.ceil((len(os.sched_getaffinity(0)) if os.name == 'posix' else psutil.cpu_count()) * 2 / 3)


class SemanticFeatures:
    @staticmethod
    def nearest_bank_detection(dataframe: NumPandasTraj, dist_threshold=1000):
        # splitting the dataframe according to trajectory ids.
            df_chunks = helpers._df_split_helper(dataframe)

            # Here, create 2/3rds number of processes as there are in the system. Some CPUs are
            # kept free at all times in order to not block up the system.
            # (Note: The blocking of system is mostly prevalent in Windows and does not happen very often
            # in Linux. However, out of caution some CPUs are kept free regardless of the system.)
            multi_pool = multiprocessing.Pool(NUM_CPU)
            result = multi_pool.map(helpers.banks_crossed, df_chunks)
            multi_pool.close()
            multi_pool.join()

            # merge the smaller pieces and then return the dataframe converted to NumPandasTraj.
            return NumPandasTraj(pd.concat(result), const.LAT, const.LONG,
                                 const.DateTime, const.TRAJECTORY_ID)

    @staticmethod
    def bank_alt(df: NumPandasTraj, dist_threshold: float = 1000):
        bbox = SpatialFeatures.get_bounding_box(df)
        tags = {'amenity': ['atm', 'banks']}

        gdf = ox.geometries_from_bbox(north=bbox[2],
                                      south=bbox[0],
                                      east=bbox[3],
                                      west=bbox[1],
                                      tags=tags)
        return gdf


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
