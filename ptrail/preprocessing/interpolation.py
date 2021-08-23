"""
    This class interpolates dataframe positions based on Datetime.
    It provides the user with the flexibility to use linear or cubic interpolation.
    In general, the user passes the dataframe, time jum and the interpolation type,
    based on the type the proper function is mapped. And if the time difference
    exceeds the time jump, the interpolated point is added to the position with large jump
    with a time increase of time jump. This interpolated row is added to the dataframe.

    | Authors: Yaksh J Haranwala, Salman Haidri

"""
import itertools
import multiprocessing as mlp
import os
from math import ceil
from typing import Optional, Text, Union

import pandas
import pandas as pd

from ptrail.core.TrajectoryDF import PTRAILDataFrame as NumTrajDF
from ptrail.preprocessing.helpers import Helpers as helper
from ptrail.utilities import constants as const

num = os.cpu_count()
NUM_CPU = ceil((num * 2) / 3)


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

            Warning
            -------
                The Interpolation methods will only return the 4 mandatory library columns
                because it is not possible to interpolate other data that may or may not be
                present in the dataset apart from latitude, longitude and datetime. As a
                result, other columns are dropped.

            Note
            ----
                The time-jump parameter specifies where the new points are to be
                inserted based on the time difference between 2 consecutive points.
                However, it does not guarantee that the dataset will be brought down
                to having difference between 2 consecutive points equal to or
                less than the user specified time jump.

            Note
            ----
                The time-jump is specified in seconds. Hence, if the user-specified
                time-jump is not sensible, then the execution of the method will take
                a very long time.

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The dataframe containing the original dataset.
                time_jump: float
                    The maximum time difference between 2 consecutive points.
                ip_type: Optional[Text], default = linear
                    The type of interpolation that is to be used.

            Returns
            -------
                PTRAILDataFrame:
                    The dataframe containing the interpolated trajectory points.
        """
        # First, lets split the dataframe into smaller chunks containing
        # points of only n trajectory per chunk. n is
        df = dataframe.reset_index()
        df_chunks = helper._df_split_helper(df)

        # Create a pool of processes which has number of processes
        # equal to the number of unique dataframe partitions.
        processes = [None] * len(df_chunks)
        manager = mlp.Manager()
        return_list = manager.list()

        ip_type = ip_type.lower().strip()
        if ip_type == 'linear':
            for i in range(len(processes)):
                processes[i] = mlp.Process(target=Interpolation._linear_ip,
                                           args=(df_chunks[i], time_jump, return_list))
                processes[i].start()

            for j in range(len(processes)):
                processes[j].join()

        elif ip_type == 'cubic':
            for i in range(len(processes)):
                processes[i] = mlp.Process(target=Interpolation._cubic_ip,
                                           args=(df_chunks[i], time_jump, return_list))
                processes[i].start()

            for j in range(len(processes)):
                processes[j].join()
        elif ip_type == 'kinematic':
            for i in range(len(processes)):
                processes[i] = mlp.Process(target=Interpolation._kinematic_ip,
                                           args=(df_chunks[i], time_jump, return_list))
                processes[i].start()

            for j in range(len(processes)):
                processes[j].join()
        elif ip_type == 'random-walk':
            for i in range(len(processes)):
                processes[i] = mlp.Process(target=Interpolation._random_walk_ip,
                                           args=(df_chunks[i], time_jump, return_list))
                processes[i].start()

            for j in range(len(processes)):
                processes[j].join()
        else:
            raise ValueError(f"Interpolation type: {ip_type} specified does not exist. Please check the"
                             "interpolation type specified and type again.")

        return NumTrajDF(pd.concat(return_list).reset_index(),
                         const.LAT, const.LONG, const.DateTime, const.TRAJECTORY_ID)

    @staticmethod
    def _linear_ip(dataframe: Union[pd.DataFrame, NumTrajDF], time_jump: float, return_list: list):
        """
            Interpolate the position of points using the Linear Interpolation method. It makes
            the use of numpy's interpolation technique for the interpolation of the points.

            WARNING: Do not use this method directly as it will run slower. Instead,
                     use the method interpolate_position() and specify the ip_type as
                     linear to perform linear interpolation much faster.

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The dataframe containing the original data.
                time_jump: float
                    The maximum time difference between 2 points. If the time difference between
                    2 consecutive points is greater than the time jump, then another point will
                    be inserted between the given 2 points.
                return_list: list
                    The list used by the Multiprocessing manager to get the return values

            Returns
            -------
                pandas.core.dataframe.DataFrame:
                    The dataframe enhanced with interpolated points.
        """
        # First, reset the index, extract the Latitude, Longitude, DateTime and Trajectory ID columns
        # and set the DateTime column only as the index. Then, store all the unique Trajectory IDs in
        # a list.
        dataframe = dataframe.reset_index()[
            [const.DateTime, const.TRAJECTORY_ID, const.LAT, const.LONG]].set_index(const.DateTime)

        # Split the smaller dataframe further into smaller chunks containing only 1
        # Trajectory ID per index.
        ids_ = list(dataframe[const.TRAJECTORY_ID].value_counts().keys())
        df_chunks = [dataframe.loc[dataframe[const.TRAJECTORY_ID] == ids_[i]] for i in range(len(ids_))]

        # Here, create as many processes at once as there are number of CPUs available in
        # the system - 1. One CPU is kept free at all times in order to not block up
        # the system. (Note: The blocking of system is mostly prevalent in Windows and does
        # not happen very often in Linux. However, out of caution 1 CPU is kept free regardless
        # of the system.)
        small_pool = mlp.Pool(NUM_CPU)
        final = small_pool.starmap(helper.linear_help,
                                   zip(df_chunks, ids_, itertools.repeat(time_jump)))
        small_pool.close()
        small_pool.join()

        # Append the smaller dataframe to process manager list so that result
        # can be finally merged into a larger dataframe.
        return_list.append(pd.concat(final))

    @staticmethod
    def _cubic_ip(dataframe: Union[pd.DataFrame, NumTrajDF], time_jump: float, return_list: list):
        try:
            """
                Method for cubic interpolation of a dataframe based on the time jump provided.
                It makes use of scipy library's CubicSpline functionality and interpolates
                the coordinates based on the Datetime of the dataframe.
    
                WARNING: Do not use this method directly as it will run slower. Instead,
                         use the method interpolate_position() and specify the ip_type as
                         cubic to perform cubic interpolation much faster.
    
                Parameters
                ----------
                    dataframe: Union[pd.DataFrame, NumTrajDF]
                        The dataframe on which interpolation is to be performed
                    time_jump: float
                        The maximum time difference allowed to have between rows
                    return_list: list
                        The list used by the Multiprocessing manager to get the return values
    
                Returns
                -------
                    pandas.core.dataframe.DataFrame:
                        The dataframe containing the new interpolated points.
    
            """
            # First, reset the index, extract the Latitude, Longitude, DateTime and Trajectory ID columns
            # and set the DateTime column only as the index. Then, store all the unique Trajectory IDs in
            # a list.
            dataframe = dataframe.reset_index()[
                [const.DateTime, const.TRAJECTORY_ID, const.LAT, const.LONG]].set_index(const.DateTime)

            # Split the smaller dataframe further into smaller chunks containing only 1
            # Trajectory ID per index.
            ids_ = list(dataframe[const.TRAJECTORY_ID].value_counts().keys())
            df_chunks = [dataframe.loc[dataframe[const.TRAJECTORY_ID] == ids_[i]] for i in range(len(ids_))]

            # Here, create as many processes at once as there are number of CPUs available in
            # the system - 1. One CPU is kept free at all times in order to not block up
            # the system. (Note: The blocking of system is mostly prevalent in Windows and does
            # not happen very often in Linux. However, out of caution 1 CPU is kept free regardless
            # of the system.).
            small_pool = mlp.Pool(NUM_CPU)
            final = small_pool.starmap(helper.cubic_help,
                                       zip(df_chunks, ids_, itertools.repeat(time_jump)))
            small_pool.close()
            small_pool.join()

            # Append the smaller dataframe to process manager list so that result
            # can be finally merged into a larger dataframe.
            return_list.append(pd.concat(final))

        except ValueError:
            raise ValueError

    @staticmethod
    def _kinematic_ip(dataframe: Union[pd.DataFrame, NumTrajDF], time_jump, return_list):
        """
             Method for Kinematic interpolation of a dataframe based on the time jump provided.
             It interpolates the coordinates based on the Datetime of the dataframe.

             WARNING: Do not use this method directly as it will run slower. Instead,
                     use the method interpolate_position() and specify the ip_type as
                     kinematic to perform kinematic interpolation much faster.

             Parameters
             ----------
                 dataframe: Union[pd.DataFrame, NumTrajDF]
                     The dataframe on which interpolation is to be performed
                 time_jump: float
                     The maximum time difference allowed to have between rows
                 return_list: list
                     The list used by the Multiprocessing manager to get the return values

             Returns
             -------
                 pandas.core.dataframe.DataFrame:
                     The dataframe containing the new interpolated points.

         """
        # First, reset the index, extract the Latitude, Longitude, DateTime and Trajectory ID columns
        # and set the DateTime column only as the index. Then, store all the unique Trajectory IDs in
        # a list.
        dataframe = dataframe.reset_index()[
            [const.DateTime, const.TRAJECTORY_ID, const.LAT, const.LONG]].set_index(const.DateTime)

        # Split the smaller dataframe further into smaller chunks containing only 1
        # Trajectory ID per index.
        ids_ = list(dataframe[const.TRAJECTORY_ID].value_counts().keys())
        df_chunks = [dataframe.loc[dataframe[const.TRAJECTORY_ID] == ids_[i]] for i in range(len(ids_))]

        # Here, create as many processes at once as there are number of CPUs available in
        # the system - 1. One CPU is kept free at all times in order to not block up
        # the system. (Note: The blocking of system is mostly prevalent in Windows and does
        # not happen very often in Linux. However, out of caution 1 CPU is kept free regardless
        # of the system.).
        small_pool = mlp.Pool(NUM_CPU)
        final = small_pool.starmap(helper.kinematic_help,
                                   zip(df_chunks, ids_, itertools.repeat(time_jump)))
        small_pool.close()
        small_pool.join()

        # Append the smaller dataframe to process manager list so that result
        # can be finally merged into a larger dataframe.
        return_list.append(pd.concat(final))

    @staticmethod
    def _random_walk_ip(dataframe: Union[pd.DataFrame, NumTrajDF], time_jump, return_list):
        """
             Method for Random walk interpolation of a dataframe based on the time jump provided.
             It interpolates the coordinates based on the Datetime of the dataframe.

             WARNING: Do not use this method directly as it will run slower. Instead,
                     use the method interpolate_position() and specify the ip_type as
                     random-walk to perform random walk interpolation much faster.


             Parameters
             ----------
                 dataframe: Union[pd.DataFrame, NumTrajDF]
                     The dataframe on which interpolation is to be performed
                 time_jump: float
                     The maximum time difference allowed to have between rows
                 return_list: list
                     The list used by the Multiprocessing manager to get the return values

             Returns
             -------
                 pandas.core.dataframe.DataFrame:
                     The dataframe containing the new interpolated points.

         """
        # First, reset the index, extract the Latitude, Longitude, DateTime and Trajectory ID columns
        # and set the DateTime column only as the index. Then, store all the unique Trajectory IDs in
        # a list.
        dataframe = dataframe.reset_index()[
            [const.DateTime, const.TRAJECTORY_ID, const.LAT, const.LONG]].set_index(const.DateTime)

        # Split the smaller dataframe further into smaller chunks containing only 1
        # Trajectory ID per index.
        ids_ = list(dataframe[const.TRAJECTORY_ID].value_counts().keys())
        df_chunks = [dataframe.loc[dataframe[const.TRAJECTORY_ID] == ids_[i]] for i in range(len(ids_))]

        # Here, create as many processes at once as there are number of CPUs available in
        # the system - 1. One CPU is kept free at all times in order to not block up
        # the system. (Note: The blocking of system is mostly prevalent in Windows and does
        # not happen very often in Linux. However, out of caution 1 CPU is kept free regardless
        # of the system.).
        small_pool = mlp.Pool(NUM_CPU)
        final = small_pool.starmap(helper.random_walk_help,
                                   zip(df_chunks, ids_, itertools.repeat(time_jump)))
        small_pool.close()
        small_pool.join()

        # Append the smaller dataframe to process manager list so that result
        # can be finally merged into a larger dataframe.
        return_list.append(pd.concat(final))

    @staticmethod
    def interpolate_at_time(dataframe: Union[NumTrajDF, pandas.DataFrame], time: Text, ip_type: Text = 'linear'):
        pass
