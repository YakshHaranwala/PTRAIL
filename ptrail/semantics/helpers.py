"""
    This module contains all the helper functions for the parallel calculations in
    the semantic features.

    Warning
    -------
        These functions should not be used directly as they would result in a
        slower calculation and execution times. In some cases, these functions
        might even yield wrong results if used directly. They are meant to be used
        only as helpers. For calculation of features, use the ones in the
        features package.

    | Authors: Yaksh J Haranwala, Salman Haidri
    | Date: 4th August, 2021
    | Version: 0.2 Beta
"""
import math
import os
import warnings

import numpy as np

import ptrail.utilities.constants as const
from ptrail.core.TrajectoryDF import NumPandasTraj
from ptrail.features.helper_functions import Helpers

warnings.filterwarnings("ignore")


class SemanticHelpers:
    @staticmethod
    def visited_poi_helper(df, surrounding_data, dist_column_label, nearby_threshold):
        """
            Given a Trajectory dataframe and another dataset with the surrounding data,
            find whether the given object is nearby a point of interest or not.

            Parameters
            ----------
                df:
                    The dataframe containing the trajectory data.
                surrounding_data:
                    The dataframe containing the data of the surroundings.
                dist_column_label: Text
                    The label of the column containing the distance of the coords from
                    the nearest POI.
                nearby_threshold: int
                    The maximum distance between the POI and the current location of the object
                    within which the object is considered to be crossing/visiting the POI.

            Returns
            -------
                The original dataframe with another column added to it indicating whether
                each point is within
        """
        # A boolean list to store if that point in trajectory lies around a POI.
        POI = []
        df2 = surrounding_data.copy()

        # Loop for every point in the dataframe and create a distance column with distance from every point in the
        # surrounding data. Then use this distance column and the distance of POI column and compare each value.
        # Store the comparison in an array True if they lie within the threshold else false. And if any of the
        # value in it is true then that point in the dataframe was near a POI.
        for i in range(len(df)):
            dist_array = Helpers.distance_from_given_point_helper(df2, (df['lat'][i], df['lon'][i]))[
                f'Distance_from_{df["lat"][i], df["lon"][i]}'].to_numpy()
            poi_array = df2[dist_column_label].to_numpy()

            dist_comp = np.abs(poi_array - dist_array) <= nearby_threshold
            POI.append(np.any(dist_comp))

        # Append the boolean list containing whether each point was near the POI of interest or not
        df['Nearby_POI'] = POI
        print('Finished')

        return df

    # ------------------------------------ General Utilities ------------------------------------ #
    @staticmethod
    def _get_partition_size(size):
        """
            Takes number of ids and makes use of a formula that gives a factor to makes set of ids
            according to the number of processors available to work with.

            Parameters
            ----------
                size: int
                    The total number of trajectory IDs in the dataset.

            Returns
            -------
                int
                   The factor by which the datasets are to be split.
        """
        # Based on the Operating system, get the number of CPUs available for
        # multiprocessing.
        # Number of available CPUs.
        num = os.cpu_count()
        NUM_CPU = math.ceil((num * 2) / 3)

        # Integer divide the total number of Trajectory IDs by the number of available CPUs
        # and square the number because if too many partitions are made, then it does more
        # harm than good for the execution speed. The factor of 1 is added to avoid errors
        # when the integer division yields a 0.
        factor = ((size // NUM_CPU)) + 1

        # Return the factor if it is less than 100 otherwise return 100.
        # This factor hence is capped at 100.
        return factor if factor < 100 else 100

    @staticmethod
    def _df_split_helper(dataframe):
        """
            This is the helper function for splitting up dataframes into smaller chunks.
            This function is widely used for main functions to help split the original
            dataframe into smaller chunks based on a fixed range of IDs. This function
            splits the dataframes based on a predetermined number, stores them in a list
            and returns it.

            Note
            ----
                The dataframe is split based on the number of CPU cores available for.
                For more info, take a look at the documentation of the get_partition_size()
                function.

            Parameters
            ----------
                dataframe: NumPandasTraj
                The dataframe that is to be split.

            Returns
            -------
                list:
                    The list containing smaller dataframe chunks.
        """
        # First, create a list containing all the ids of the data and then further divide that
        # list items and split it into sub-lists of ids equal to split_factor.
        ids_ = list(dataframe.traj_id.value_counts().keys())

        # Get the ideal number of IDs by which the dataframe is to be split.
        split_factor = SemanticHelpers._get_partition_size(len(ids_))
        # split_factor = 1
        ids_ = [ids_[i: i + split_factor] for i in range(0, len(ids_), split_factor)]

        # Now split the dataframes based on set of Trajectory ids.
        # As of now, each smaller chunk is supposed to have data of 100
        # trajectory IDs max
        df_chunks = [dataframe.loc[dataframe.index.get_level_values(const.TRAJECTORY_ID).isin(ids_[i])]
                     for i in range(len(ids_))]
        return df_chunks

    # @staticmethod
    # def banks_crossed(df: NumPandasTraj, dist_threshold: float = 1000):
    #     """
    #         Find the banks crossed by the object during its trajectory.
    #
    #         Note
    #         ----
    #             By crossing, here we mean that the banks given out are within
    #             the user specified range. So basically if the user has specified the
    #             range as 1000 m, then the banks and atms given out are within 1000
    #             m of the points of trajectory.
    #
    #         Parameters
    #         ----------
    #             df: NumPandasTraj
    #                 The dataframe containing the Trajectory Data.
    #             dist_threshold: float
    #                 The radius of the circle within which the banks are to be located.
    #
    #         Returns
    #         -------
    #             pd.core.dataframe.DataFrame:
    #                 Dataframe containing the banks crossed by the object during the course
    #                 of its trajectory.
    #
    #
    #     """
    #     # The starting points of the trajectory.
    #     start = df[const.LAT][0], df[const.LONG][0]
    #     # dead_start = df[const.LAT][0], df[const.LONG][0]
    #     lst = list()
    #
    #     # Now for every point in trajectory, check whether the distance
    #     # between the current point and the start point is greater than the threshold.
    #     for i in range(1, len(df)):
    #         pt = df[const.LAT][i], df[const.LONG][i]
    #         # Now, if the distance is indeed greater than the threshold, then check whether
    #         # the distance between the very first point of the trajectory and the current
    #         # point is greater than the threshold and if so, then add that point to the
    #         # list of points from where the unique bank locations are to be found. The reasoning
    #         # behind this is that the trajectory might not be a straight line and the object might
    #         # take a turn and again come back to the point from where the trajectory started.
    #         # Hence, the double comparisons.
    #         if FormulaLog.haversine_distance(start[0], start[1], pt[0], pt[1]) > dist_threshold:
    #             lst.append(pt)
    #
    #             # Change the initial comparison point to the current point.
    #             start = pt
    #
    #     tags = {'amenity': ['atm', 'banks']}
    #
    #     print(f"Num unique points: {len(lst)}")
    #
    #     # Now, for each unique point calculated above, find the nearby banks
    #     # and atms and drop the duplicates as the areas might overlap.
    #     pdf = []
    #     for i in range(len(lst)):
    #         pt = lst[i]
    #         gdf = ox.geometries_from_point(center_point=(pt[0], pt[1]),
    #                                        tags=tags,
    #                                        dist=dist_threshold)
    #         pdf.append(gdf.drop_duplicates())
    #
    #     # Join all the smaller dataframes, add the traj_id column, drop the
    #     # element_type column.
    #     dataframe = pd.concat(pdf)
    #     dataframe['traj_id'] = df.reset_index()[const.TRAJECTORY_ID][0]
    #     # dataframe = dataframe.reset_index().drop(columns=['element_type'])
    #
    #     # Finally, set the index as traj_id and osmid and return the dataframe.
    #     return dataframe.set_index([const.TRAJECTORY_ID, 'osmid'], drop=True)
    #
    # @staticmethod
    # def nearest_bank_helper(dataframe, dist_threshold: int):
    #     """
    #         Find the bank nearest to the given point within the given threshold.
    #
    #         Parameters
    #         ----------
    #             dataframe:
    #                 The point nearest to which the bank is to be found.
    #             dist_threshold: int
    #                 The maximum range within which the banks are to be searched.
    #
    #         Returns
    #         -------
    #
    #     """
    #
    #     bank_column = []
    #     for i in range(len(dataframe)):
    #         coords = (dataframe.reset_index()[const.LAT][i], dataframe.reset_index()[const.LONG][i])
    #         tags = {'amenity': ['atm', 'banks']}
    #
    #         banks = ox.geometries_from_point(center_point=coords,
    #                                          dist=dist_threshold,
    #                                          tags=tags)
    #
    #         lat = list(banks['geometry'].apply(lambda p: p.y))
    #         lon = list(banks['geometry'].apply(lambda p: p.x))
    #
    #         dists = []
    #         for j in range(len(lat)):
    #             dists.append(FormulaLog.haversine_distance(coords[0], coords[1], lat[j], lon[j]))
    #
    #         if len(dists) > 0:
    #             val = min(dists)
    #         else:
    #             val = None
    #         bank_column.append(val)
    #         print(i)
    #
    #     dataframe['Nearest_bank'] = pd.Series(bank_column, dtype=object)
    #     print("DF over")
    #     return dataframe

    # @staticmethod
    # def bank_within_dist_helper(dataframe, poi, dist_threshold):
    #     indicator = []
    #
    #     for i in range(len(dataframe)):
    #         coords = (dataframe.reset_index()[const.LAT][i],
    #                   dataframe.reset_index()[const.LAT][i])
    #
    #         dist = SpatialFeatures.create_distance_from_given_point_column(dataframe=poi,
    #                                                                        coordinates=coords)[f'Distance_from_{coords}']
    #
    #         if min(dist) < dist_threshold:
    #             indicator.append(1)
    #         else:
    #             indicator.append(0)
    #
    #     dataframe[f'Bank_within_{dist_threshold}m'] = indicator
    #
    #     return dataframe
