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

from preprocessing.helpers import InterpolationHelpers as help
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
        df_chunks = help._df_split_helper(df)

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

        # return NumTrajDF(pd.concat(results).reset_index(), const.LAT, const.LONG,
        #                  const.DateTime, const.TRAJECTORY_ID)
        return pd.concat(results)

    @staticmethod
    def _linear_ip(dataframe, time_jump):
        dataframe = dataframe.reset_index(drop=True)[[const.DateTime, const.TRAJECTORY_ID, const.LAT, const.LONG]].set_index(const.DateTime)
        ids_ = list(dataframe[const.TRAJECTORY_ID].value_counts().keys())

        final = []
        for i in range(len(ids_)):
            df = dataframe.loc[dataframe[const.TRAJECTORY_ID] == ids_[i]]
            new_times = dataframe.reset_index()[const.DateTime] + pd.to_timedelta(time_jump, unit='seconds')
            ip_lat = np.interp(new_times,
                               df.reset_index()[const.DateTime],
                               df.reset_index()[const.LAT])

            ip_long = np.interp(new_times,
                                df.reset_index()[const.DateTime],
                                df.reset_index()[const.LONG])

            time_deltas = df.reset_index()[const.DateTime].diff().dt.total_seconds()
            id_ = df.reset_index()[const.TRAJECTORY_ID].iloc[0]

            for i in range(len(time_deltas)):
                if time_deltas[i] > time_jump:
                    df.loc[new_times[i-1]] = [id_, ip_lat[i-1], ip_long[i-1]]
                final.append(df)
        return dataframe.sort_values([const.DateTime])

    @staticmethod
    def _cubic_ip(dataframe, time_jump):
        dataframe = dataframe.reset_index(drop=True)[[const.DateTime, const.TRAJECTORY_ID, const.LAT, const.LONG]].set_index(const.DateTime)
        ids_ = list(dataframe[const.TRAJECTORY_ID].value_counts().keys())

        final = []
        for i in range(len(ids_)):
            df = dataframe.loc[dataframe[const.TRAJECTORY_ID] == ids_[i]]
            if len(df) < 3:
                continue
            old_times = df.reset_index()[const.DateTime]
            new_times = df.reset_index()[const.DateTime] + pd.to_timedelta(time_jump, unit='seconds')

            coords = df.reset_index()[[const.LAT, const.LONG]].to_numpy()

            cubic_spline = CubicSpline(old_times, coords, extrapolate=True, bc_type='not-a-knot')
            ip_coords = cubic_spline(new_times)

            id_ = df.reset_index()[const.TRAJECTORY_ID].iloc[0]
            time_deltas = df.reset_index()[const.DateTime].diff().dt.total_seconds()

            for j in range(len(time_deltas)):
                if time_deltas[j] > time_jump:
                    df.loc[new_times[j-1]] = [id_, ip_coords[j-1][0], ip_coords[j-1][1]]
            final.append(df)

        return pd.concat(final).sort_values([const.DateTime])

    # @staticmethod
    # def cubic_ip(dataframe, time_jump):
    #     dataframe = dataframe.reset_index().set_index(const.DateTime)
    #
    #     old_times = dataframe.reset_index()[const.DateTime]
    #     new_times = dataframe.reset_index()[const.DateTime] + pd.to_timedelta(time_jump, unit='seconds')
    #
    #     lats = interp1d(old_times, dataframe.reset_index()[const.LAT], kind='cubic', fill_value='extrapolate')
    #     lons = interp1d(old_times, dataframe.reset_index()[const.LONG], kind='cubic', fill_value='extrapolate')
    #
    #     ip_lat = lats(new_times)
    #     ip_lon = lons(new_times)
    #
    #     id_ = dataframe.reset_index()[const.TRAJECTORY_ID].iloc[0]
    #     time_deltas = dataframe.reset_index()[const.DateTime].diff().dt.total_seconds()
    #
    #     for i in range(len(time_deltas)):
    #         if time_deltas[i] > time_jump:
    #             dataframe.loc[new_times[i - 1]] = [id_, ip_lat[i - 1], ip_lon[i - 1]]
    #
    #     return dataframe.sort_values([const.DateTime])