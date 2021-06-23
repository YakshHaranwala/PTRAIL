"""
    The helpers class has the functionalities that interpolate a point based
    on the given data by the user. The class contains the following 4
    interpolation calculators:
        1. Linear Interpolation
        2. Cubic Interpolation
        3. Random-Walk Interpolation
        4. Kinematic Interpolation

    Besides the interpolation helpers, there are also general utilities which
    are used in splitting up dataframes for running the code in parallel.

    @Authors: Yaksh J Haranwala, Salman Haidri
    @Version: 1.0
    @Date: 16th June, 2021
"""
import os
import psutil
import pandas as pd
import numpy as np
import utilities.constants as const

from scipy.interpolate import CubicSpline
from typing import Text, Union
from core.TrajectoryDF import NumPandasTraj as NumTrajDF


class Helpers:
    # ------------------------------------ Interpolation Helpers --------------------------------------- #
    @staticmethod
    def _linear_help(dataframe: Union[pd.DataFrame, NumTrajDF], id_: Text, time_jump: float):
        """
            This method takes a dataframe and uses linear interpolation to determine coordinates
            of location on Datetime where the time difference between 2 consecutive points exceeds
            the user-specified time_jump and inserts the interpolated point those between 2 points.

            WARNING: Private Helper method
                Can't be used for dataframes with multiple trajectory ids and there might be a significant
                drop in performance.

            Parameters
            ----------
                dataframe: Union[pd.DataFrame, NumTrajDF]
                     The dataframe containing the original trajectory data.
                id_: Text
                    The Trajectory ID of the points in the dataframe.
                time_jump: float

            Returns
            -------
                pandas.core.dataframe.DataFrame
                    The dataframe containing the trajectory enhanced with interpolated
                    points.

        """
        # Now, for each unique ID in the dataframe, interpolate the points.
        # Create a Series containing new times which are calculated as follows:
        #    new_time[i] = original_time[i] + time_jump.
        new_times = dataframe.reset_index()[const.DateTime] + pd.to_timedelta(time_jump, unit='seconds')

        # Now, interpolate the latitudes using numpy based on the new times calculated above.
        ip_lat = np.interp(new_times,
                           dataframe.reset_index()[const.DateTime],
                           dataframe.reset_index()[const.LAT])

        # Now, interpolate the longitudes using numpy based on the new times calculated above.
        ip_long = np.interp(new_times,
                            dataframe.reset_index()[const.DateTime],
                            dataframe.reset_index()[const.LONG])

        # Here, store the time difference between all the consecutive points in an array.
        time_deltas = dataframe.reset_index()[const.DateTime].diff().dt.total_seconds()

        # Now, for each point in the trajectory, check whether the time difference between
        # 2 consecutive points is greater than the user-specified time_jump, and if so then
        # insert a new point that is linearly interpolated between the 2 original points.
        for j in range(len(time_deltas)):
            if time_deltas[j] > time_jump:
                dataframe.loc[new_times[j - 1]] = [id_, ip_lat[j - 1], ip_long[j - 1]]

        return dataframe

    @staticmethod
    def _cubic_help(df: Union[pd.DataFrame, NumTrajDF], id_: Text, time_jump: float):
        """
            This method takes a dataframe and uses cubic interpolation to determine coordinates
            of location on Datetime where the time difference between 2 consecutive points exceeds
            the user-specified time_jump and inserts the interpolated point those between 2 points.

            WARNING: Private Helper method
                Can't be used for dataframes with multiple trajectory ids and there might be a significant
                drop in performance.

            Parameters
            ----------
                df: Union[pd.DataFrame, NumTrajDF]
                     The dataframe containing the original trajectory data.
                id_: Text
                    The Trajectory ID of the points in the dataframe.
                time_jump: float

            Returns
            -------
                pandas.core.dataframe.DataFrame
                    The dataframe containing the trajectory enhanced with interpolated
                    points.
        """
        # Create a Series containing new times which are calculated as follows:
        #    new_time[i] = original_time[i] + time_jump.
        new_times = df.reset_index()[const.DateTime] + pd.to_timedelta(time_jump, unit='seconds')

        # Extract the Latitude, Longitude pairs for each point and store it in a
        # numpy array.
        coords = df.reset_index()[[const.LAT, const.LONG]].to_numpy()

        # Now, using Scipy's Cubic spline, create a spline object for interpolation of
        # points for the dataframes which have a length greater than 3 else CubicSpline
        # doesn't execute.
        if len(df) > 3:
            cubic_spline = CubicSpline(x=df.reset_index()[const.DateTime],
                                       y=coords,
                                       extrapolate=True, bc_type='not-a-knot')

            # Now, calculate the interpolated position of the points at all the new_times
            #    calculated above.
            ip_coords = cubic_spline(new_times)

        # Here, store the time difference between all the consecutive points in an array.
        time_deltas = df.reset_index()[const.DateTime].diff().dt.total_seconds()

        # Now, for each point in the trajectory, check whether the time difference between
        # 2 consecutive points is greater than the user-specified time_jump, and if so then
        # insert a new point that is cubic-spline interpolated between the 2 original points.
        for j in range(len(time_deltas)):
            # If the trajectory has less than 3 points, then skip the trajectory
            # from the interpolation.
            if len(df) > 3:
                if time_deltas[j] > time_jump:
                    df.loc[new_times[j - 1]] = [id_, ip_coords[j - 1][0], ip_coords[j - 1][1]]

        return df

    # -------------------------------------- General Utilities ---------------------------------- #
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
        available_cpus = len(os.sched_getaffinity(0)) if os.name == 'posix' \
            else psutil.cpu_count()  # Number of available CPUs.

        # Integer divide the total number of Trajectory IDs by the number of available CPUs
        # The factor of 1 is added to avoid errors when the integer division yields a 0.
        factor = (size // available_cpus) + 1

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
            NOTE: The dataframe is split based on the number of CPU cores available for.
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
        split_factor = Helpers._get_partition_size(len(ids_))
        ids_ = [ids_[i: i + split_factor] for i in range(0, len(ids_), split_factor)]

        # Now split the dataframes based on set of Trajectory ids.
        # As of now, each smaller chunk is supposed to have data of 100
        # trajectory IDs max
        df_chunks = [dataframe.loc[dataframe[const.TRAJECTORY_ID].isin(ids_[i])]
                     for i in range(len(ids_))]
        return df_chunks
