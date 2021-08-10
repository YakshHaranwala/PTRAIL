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
import os
from typing import Union, Text

import geopandas as gpd
import pandas as pd
import psutil
import numpy as np

from Nummobility.core.TrajectoryDF import NumPandasTraj
from Nummobility.features.spatial_features import SpatialFeatures
from Nummobility.features.helper_functions import Helpers

NUM_CPU = math.ceil((len(os.sched_getaffinity(0)) if os.name == 'posix' else psutil.cpu_count()) * 2 / 3)


class SemanticFeatures:
    @staticmethod
    def visited_location(df: NumPandasTraj,
                         geo_layers: Union[pd.DataFrame, gpd.GeoDataFrame],
                         visited_location_name: Text,
                         location_column_name: Text):
        """
            Create a column called visited_Pasture for all the pastures present in the
            dataset.

            Parameters
            ----------
                df: NumPandasTraj
                    The dataframe containing the dataset.
                geo_layers: Union[pd.DataFrame, gpd.GeoDataFrame]
                    The Dataframe containing the geographical layers near the trajectory data.
                    It is to be noted
                visited_location_name: Text
                    The location for which it is to be checked whether the objected visited it
                    or not.
                location_column_name: Text

            Returns
            -------
                NumPandasTraj:
                    The Dataframe containing a new column indicating whether the animal
                    has visited the pasture or not.

        """
        df = df.reset_index()
        geo_layers = geo_layers.loc[geo_layers[location_column_name] == visited_location_name]
        df1 = gpd.GeoDataFrame(df,
                               geometry=gpd.points_from_xy(df["lon"],
                                                           df["lat"]),
                               crs={"init": "epsg:4326"})

        df2 = geo_layers
        df2 = gpd.GeoDataFrame(geo_layers,
                               geometry=gpd.points_from_xy(geo_layers["lon"],
                                                           geo_layers["lat"]),
                               crs={"init": "epsg:4326"})

        intersection = gpd.overlay(df1, df2, how='intersection')

        print(len(intersection))

        merged = pd.merge(df, intersection, how='outer', indicator=True)['_merge']

        merged = merged.replace('both', 1)
        merged = merged.replace('left_only', 0)
        merged = merged.replace('right_only', 0)

        df[f'Visited_{visited_location_name}'] = merged
        df = df.drop(columns='geometry')
        # return merged
        return NumPandasTraj(df,
                             latitude='lat',
                             longitude='lon',
                             datetime='DateTime',
                             traj_id='traj_id')

    @staticmethod
    def distance_from_nearby_hotels():
        pass

    @staticmethod
    def distance_from_nearby_hospitals():
        pass

    @staticmethod
    def trajectory_crossing_paths():
        pass

    @staticmethod
    def points_inside_polygon():
        pass

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
