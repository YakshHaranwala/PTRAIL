"""
    This class interpolates dataframe positions based on Datetime.
    It provides the user with the flexibility to use linear or cubic interpolation.
    In general, the user passes the dataframe, time jum and the interpolation type,
    based on the type the proper function is mapped. And if the time difference
    exceeds the time jump, the interpolated point is added to the position with large jump
    with a time increase of time jump. This interpolated row is added to the dataframe.

    @Authors: Yaksh J Haranwala, Salman Haidri
    @Date: 21st June, 2021
    @Version: 1.0
"""
import itertools
import numpy as np
import pandas as pd
import utilities.constants as const
import multiprocessing as mlp

from preprocessing.helpers import Helpers as helper
from scipy.interpolate import CubicSpline, interp1d
from core.TrajectoryDF import NumPandasTraj as NumTrajDF
from typing import Optional, Text


class Interpolation:
    @staticmethod
    def interpolate_position(dataframe: NumTrajDF, time_jump: float, ip_type: Optional[Text] = 'linear'):
        """
            Interpolate the position of an object and create new points using one of
            the interpolation methods provided by the Library. Currently, the library
            supports the following 4 interpolation methods:
                1. Linear Interpolation
                2. Cubic-Spline Interpolation
                3. Kinematic Interpolation
                4. Random Walk Interpolation

            WARNING: THE INTERPOLATION METHODS WILL ONLY RETURN THE 4 FUNDAMENTAL LIBRARY
                     COLUMNS AND A 'Distance_prev_to_curr' COLUMN BECAUSE IT IS NOT POSSIBLE
                     TO INTERPOLATE OTHER DATA THAT MIGHT BE PRESENT IN THE DATASET APART
                     FROM LATITUDE, LONGITUDE AND DateTime. AS A RESULT, OTHER COLUMNS ARE
                     DROPPED AND LEFT TO USER TO TAKE CARE OF THAT.

            NOTE: The time-jump parameter specifies where the new points are to be
                  inserted based on the time difference between 2 consecutive points.
                  However, it does not guarantee that the dataset will be brought down
                  to having difference between 2 consecutive points equal to or
                  less than the user specified time jump.

            NOTE: The time-jump is specified in seconds. Hence, if the user-specified
                  time-jump is not sensible, then the execution of the method will take
                  a very long time.

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The dataframe containing the original dataset.
                time_jump: float
                    The maximum time difference between 2 consecutive points.
                ip_type: Optional[Text], default = linear
                    The type of interpolation that is to be used.

            Returns
            -------
                NumPandasTraj:
                    The dataframe containing the interpolated trajectory points.
        """
        # First, lets split the dataframe into smaller chunks containing
        # points of only 1 trajectory per chunk.
        df = dataframe.reset_index()
        df_chunks = helper._df_split_helper(df)

        # Create a pool of processes which has number of processes
        # equal to the number of unique dataframe partitions.
        pool = mlp.Pool(len(df_chunks))

        ip_type = ip_type.lower().strip()
        results = None
        if ip_type == 'linear':
            results = pool.starmap(Interpolation._linear_ip, (zip(df_chunks, itertools.repeat(time_jump))))
        elif ip_type == 'cubic':
            results = pool.starmap(Interpolation._cubic_ip, (zip(df_chunks, itertools.repeat(time_jump))))
        elif ip_type == 'kinematic':
            pass
        elif ip_type == 'random-walk':
            pass
        else:
            raise ValueError(f"Interpolation type: {ip_type} specified does not exist. Please check the"
                             "interpolation type specified and type again.")

        return NumTrajDF(pd.concat(results).reset_index(), const.LAT, const.LONG,
                         const.DateTime, const.TRAJECTORY_ID)


    @staticmethod
    def _linear_ip(dataframe, time_jump):
        """
            Interpolate the position of points using the Linear Interpolation method. It makes
            the use of numpy's interpolation technique for the interpolation of the points.

            WARNING: Do not use this method directly as it will run slower. Instead,
                     use the method interpolate_position() and specify the ip_type as
                     linear to perform linear interpolation much faster.

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The dataframe containing the original data.
                time_jump: float
                    The maximum time difference between 2 points. If the time difference between
                    2 consecutive points is greater than the time jump, then another point will
                    be inserted between the given 2 points.

            Returns
            -------
                pandas.core.dataframe.DataFrame:
                    The dataframe enhanced with interpolated points.
        """
        # First, reset the index, extract the Latitude, Longitude, DateTime and Trajectory ID columns
        # and set the DateTime column only as the index. Then, store all the unique Trajectory IDs in
        # a list.
        dataframe = dataframe.reset_index(drop=True)[
            [const.DateTime, const.TRAJECTORY_ID, const.LAT, const.LONG]].set_index(const.DateTime)
        ids_ = list(dataframe[const.TRAJECTORY_ID].value_counts().keys())

        # Now, for each unique ID in the dataframe, interpolate the points.
        for i in range(len(ids_)):
            df = dataframe.loc[dataframe[const.TRAJECTORY_ID] == ids_[i]]   # Extract points of only 1 traj ID.
            # Create a Series containing new times which are calculated as follows:
            #    new_time[i] = original_time[i] + time_jump.
            new_times = df.reset_index()[const.DateTime] + pd.to_timedelta(time_jump, unit='seconds')

            # Now, interpolate the latitudes using numpy based on the new times calculated above.
            ip_lat = np.interp(new_times,
                               df.reset_index()[const.DateTime],
                               df.reset_index()[const.LAT])

            # Now, interpolate the longitudes using numpy based on the new times calculated above.
            ip_long = np.interp(new_times,
                                df.reset_index()[const.DateTime],
                                df.reset_index()[const.LONG])

            # Here, store the time difference between all the consecutive points in an array.
            time_deltas = df.reset_index()[const.DateTime].diff().dt.total_seconds()
            id_ = df.reset_index()[const.TRAJECTORY_ID].iloc[0]

            # Now, for each point in the trajectory, check whether the time difference between
            # 2 consecutive points is greater than the user-specified time_jump, and if so then
            # insert a new point that is linearly interpolated between the 2 original points.
            for j in range(len(time_deltas)):
                if time_deltas[j] > time_jump:
                    dataframe.loc[new_times[j-1]] = [id_, ip_lat[j-1], ip_long[j-1]]

        return dataframe

    @staticmethod
    def _cubic_ip(dataframe, time_jump):
        """
            Method for cubic interpolation of a dataframe based on the time jump provided.
            It makes use of scipy library's CubicSpline functionality and interpolates
            the coordinates based on the Datetime of the dataframe.

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The dataframe on which interpolation is to be performed
                time_jump: float
                    The maximum time difference allowed to have between rows

            Returns
            -------
                pandas.core.dataframe.DataFrame:
                    The dataframe containing the new interpolated points.

        """
        # First, reset the index, extract the Latitude, Longitude, DateTime and Trajectory ID columns
        # and set the DateTime column only as the index. Then, store all the unique Trajectory IDs in
        # a list
        dataframe = dataframe.reset_index(drop=True)[
            [const.DateTime, const.TRAJECTORY_ID, const.LAT, const.LONG]].set_index(const.DateTime)
        ids_ = list(dataframe[const.TRAJECTORY_ID].value_counts().keys())

        # Now, for each unique ID in the dataframe, interpolate the points.
        for i in range(len(ids_)):
            df = dataframe.loc[dataframe[const.TRAJECTORY_ID] == ids_[i]]

            # If the trajectory has less than 3 points, then skip the trajectory
            # from the interpolation.
            if len(df) < 3:
                continue

            # Create a Series containing new times which are calculated as follows:
            #    new_time[i] = original_time[i] + time_jump.
            new_times = df.reset_index()[const.DateTime] + pd.to_timedelta(time_jump, unit='seconds')

            # Extract the Latitude, Longitude pairs for each point and store it in a
            # numpy array.
            coords = df.reset_index()[[const.LAT, const.LONG]].to_numpy()

            # Now, using Scipy's Cubic spline, create a spline object for interpolation of
            # points.
            cubic_spline = CubicSpline(x=df.reset_index()[const.DateTime],
                                       y=coords,
                                       extrapolate=True, bc_type='not-a-knot')

            # Now, calculate the interpolated position of the points at all the new_times
            # calculated above.
            ip_coords = cubic_spline(new_times)

            # Here, store the time difference between all the consecutive points in an array.
            time_deltas = df.reset_index()[const.DateTime].diff().dt.total_seconds()
            id_ = df.reset_index()[const.TRAJECTORY_ID].iloc[0]

            # Now, for each point in the trajectory, check whether the time difference between
            # 2 consecutive points is greater than the user-specified time_jump, and if so then
            # insert a new point that is cubic-spline interpolated between the 2 original points.
            for j in range(len(time_deltas)):
                if time_deltas[j] > time_jump:
                    dataframe.loc[new_times[j - 1]] = [id_, ip_coords[j - 1][0], ip_coords[j - 1][1]]

        return dataframe
