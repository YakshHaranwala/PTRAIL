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
from typing import Union, Text

import geopandas as gpd
import pandas as pd
import psutil
import numpy as np

from math import ceil
from Nummobility.core.TrajectoryDF import NumPandasTraj
from Nummobility.features.spatial_features import SpatialFeatures
from Nummobility.semantics.helpers import SemanticHelpers

NUM_CPU = ceil(psutil.cpu_count() * 2 / 3)


class SemanticFeatures:
    @staticmethod
    def visited_location(df: NumPandasTraj,
                         geo_layers: Union[pd.DataFrame, gpd.GeoDataFrame],
                         visited_location_name: Text,
                         location_column_name: Text):
        """
            Create a column called visited_Pasture for all the pastures present in the
            dataset.

            Warning
            -------
                While using this method, make sure that the geo_layers parameter dataframe
                that is being passed into the method has Latitude and Longitude columns with
                columns named as 'lat' and 'lon' respectively. If this format is not followed
                then a KeyError will be thrown.

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

        # First, filter out the geo_layers dataset to include only the points of the location
        # specified by the user.
        geo_layers = geo_layers.loc[geo_layers[location_column_name] == visited_location_name]

        # Now for the trajectory dataset and the geo layers dataset both, convert them to a
        # GeoDataFrame with the geometry of lat-lon for each point.
        df1 = gpd.GeoDataFrame(df,
                               geometry=gpd.points_from_xy(df["lon"],
                                                           df["lat"]),
                               crs={"init": "epsg:4326"})

        df2 = gpd.GeoDataFrame(geo_layers,
                               geometry=gpd.points_from_xy(geo_layers["lon"],
                                                           geo_layers["lat"]),
                               crs={"init": "epsg:4326"})

        # Now, using GeoPandas, find where the trajectory points and the geo-layers
        # point intersect.
        intersection = gpd.overlay(df1, df2, how='intersection')

        # Now, in the original dataframe, check which points have intersected
        # with the geo-layers dataset and which ones have not.
        merged = pd.merge(df, intersection, how='outer', indicator=True)['_merge']

        # Finally, replace the truth value of the points that have intersected to 1
        # and set it to 0 for the points that have not intersected.
        merged = merged.replace('both', 1)
        merged = merged.replace('left_only', 0)
        merged = merged.replace('right_only', 0)

        # Assign the resultant column to the original df and drop the unnecessary column
        # of geometry.
        df[f'Visited_{visited_location_name}'] = merged
        df = df.drop(columns='geometry')

        # return merged
        return NumPandasTraj(df,
                             latitude='lat',
                             longitude='lon',
                             datetime='DateTime',
                             traj_id='traj_id')

    @staticmethod
    def visited_waterbody(df: NumPandasTraj,
                          water_bodies: Union[pd.DataFrame, gpd.GeoDataFrame]):
        """
            Create a column with the information if the nearby water bodies were
            visited or intersected by the given trajectory.

            Parameters
            ----------
                df: NumPandasTraj
                    The dataframe containing the trajectory data.
                water_bodies: Union[pd.DataFrame, gpd.GeoDataFrame]
                    The dataframe containing the data about water bodies in the
                    area of interest.

            Return
            ------
                NumPandasTraj:
                    The dataframe containing the column indicating whether a water body/ water bodies
                    were intersected by the object or not.
        """
        df = df.reset_index()
        df1 = gpd.GeoDataFrame(df,
                               geometry=gpd.points_from_xy(df["lon"],
                                                           df["lat"]),
                               crs={"init": "epsg:4326"})

        intersection = gpd.overlay(df1, water_bodies, how="intersection")

        merged = pd.merge(df, intersection, how="outer", indicator=True)['_merge']

        merged = merged.replace('both', 1)
        merged = merged.replace('left_only', 0)
        merged = merged.replace('right_only', 0)

        df[f'Visited_Waterbodies'] = merged
        df = df.drop(columns='geometry')

        return NumPandasTraj(df,
                             latitude='lat',
                             longitude='lon',
                             datetime='DateTime',
                             traj_id='traj_id')

    @staticmethod
    def waterbody_alt(df: NumPandasTraj,
                      surrounding_data: Union[gpd.GeoDataFrame, pd.DataFrame, NumPandasTraj],
                      dist_column_label: Text):
        """
            Given a surrounding data with information about the distance to the nearest water source
            from a given coordinate, check whether the objects in the given trajectory data have
            crossed those water sources or not.

            Warning
            -------
                It is to be noted that for this method to work, the surrounding dataset NEEDS to have a
                column containing distance to the nearest water body. For more info, see the Starkey dataset
                which has the columns like 'DistCWat' and 'DistEWat'.


            Parameters
            ----------
                df: NumPandasTraj
                    The dataframe containing the trajectory data.
                surrounding_data: Union[gpd.GeoDataFrame, pd.DataFrame]
                    The surrounding data that needs to contain the information of distance
                    to the nearest water body.
                dist_column_label: Text
                    The name of the column containing the distance information.

            Returns
            -------
                NumPandasTraj:
                    The dataframe containing the new column indicating whether the object
                    at that point is nearby a water body.

            #TODO: Parallelize the shit out of this function.
        """
        df_chunks = SemanticHelpers._df_split_helper(df)
        print(len(df_chunks))
        # Here, create 2/3rds number of processes as there are in the    system. Some CPUs are
        # kept free at all times in order to not block up the system.
        # (Note: The blocking of system is mostly prevalent in Windows and does not happen very often
        # in Linux. However, out of caution some CPUs are kept free regardless of the system.)
        mp_pool = multiprocessing.Pool(NUM_CPU)
        results = mp_pool.starmap(SemanticHelpers.waterbody_visited_helper,
                                  zip(df_chunks,
                                      itertools.repeat(surrounding_data),
                                      itertools.repeat(dist_column_label)
                                      )
                                  )
        mp_pool.close()
        mp_pool.join()

        # Concatenate all the smaller dataframes and return the answer.
        results = pd.concat(results)
        return results


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
