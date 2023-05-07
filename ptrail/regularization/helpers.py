"""
    Warning
    -------
        | 1. None of the methods in this module should be used directly while performing operations on dataset.
        | 2. These methods are helpers for the regularization methods in the regularization.py
             module and hence run linearly and not in parallel which will result in slower execution time.

    Besides the interpolation helpers, there are also general utilities which
    are used in splitting up dataframes for running the code in parallel.

    | Author: Yaksh J. Haranwala
"""
import os
import warnings
from typing import Text, Union
from math import ceil

import numpy as np
import pandas as pd
import ptrail.utilities.constants as const

from ptrail.core.TrajectoryDF import PTRAILDataFrame
from ptrail.utilities.KinematicUtils import FormulaLog
from ptrail.utilities.conversions import Conversions
from warnings import simplefilter

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
warnings.simplefilter("ignore", RuntimeWarning)


class Helpers:
    @staticmethod
    def calculate_metric_bw_points(traj, traj_time, metric):
        """
            Given a trajectory and the metric, calculate the metric
            for all the points in the trajectory.

            Parameters
            ----------
                traj:
                    The trajectory object for which the metric is to be
                    calculated.
                traj_time: np.array
                    Array with seconds of each point.
                metric: function object
                    The metric that is to be calculated.

            Returns
            -------
                d_max: float
                    The maximum distance.
                idx: int
                    The index that provides the maximum distance.
                mean_dist: float
                    The average of distances.

        """
        # Initiate the values to be returned.
        d_max = 0
        idx = 0
        distances = np.array([])
        traj_len = len(traj['lat'])

        # Get the first and the last points.
        start = traj['lat'][0], traj['lon'][0], traj_time[0]
        end = traj['lat'][-1], traj['lon'][-1], traj_time[-1]

        # Find the max distance and index and return it.
        for i in range(1, traj_len - 1):
            # Get the mid-point at index i.
            middle = traj['lat'][i], traj['lon'][i], traj_time[i]

            # Calculate the distance and append it to the distances array.
            dist = metric(start, middle, end)
            distances = np.append(distances, dist)
            if dist > d_max:
                d_max = dist
                idx = i

        # Return the required values.
        return d_max, idx, distances.mean()

    @staticmethod
    def compress_individual(trajectory, dim_set, traj_time, calc_func, epsilon):
        """
            Compress the given trajectory using the give metric.

            Parameters
            ----------
                trajectory:
                    The trajectory object which is to be compressed.
                dim_set: list
                    The attributes (columns) of the trajectory.
                traj_time: np.array
                    The array with the time in seconds of each point.
                calc_func: function object
                    The measure to be used to compress the trajectory.
                epsilon: float
                    Compression threshold.

            Returns
            -------
                trajectory:
                    The compressed trajectory object.
        """
        new_trajectory = {}
        for dim in dim_set:
            new_trajectory[dim] = np.array([])
        traj_len = len(trajectory['lat'])

        # Get the time in seconds
        d_max, idx, _ = Helpers.calculate_metric_bw_points(trajectory, traj_time, calc_func)
        trajectory['DateTime'] = trajectory['DateTime'].astype(str)

        # print(f"\tepsilon: {epsilon}, d_max: {d_max}, index: {idx}, traj_len: {len(trajectory)}")
        if d_max > epsilon:
            traj1 = {}
            traj2 = {}
            for dim in dim_set:
                traj1[dim] = trajectory[dim][:idx]
                traj2[dim] = trajectory[dim][idx:]

            # Compress the parts created above
            recResults1 = traj1
            if len(traj1) > 2:
                recResults1 = Helpers.compress_individual(traj1, dim_set, traj_time[:idx], calc_func, epsilon)

            recResults2 = traj2
            if len(traj2) > 2:
                recResults2 = Helpers.compress_individual(traj2, dim_set, traj_time[idx:], calc_func, epsilon)

            for dim in dim_set:
                new_trajectory[dim] = np.append(new_trajectory[dim], recResults1[dim])
                new_trajectory[dim] = np.append(new_trajectory[dim], recResults2[dim])
        else:
            trajectory['DateTime'] = trajectory['DateTime'].astype(str)
            for dim in dim_set:
                new_trajectory[dim] = np.append(new_trajectory[dim], trajectory[dim][0])
                if traj_len > 1:
                    new_trajectory[dim] = np.append(new_trajectory[dim], trajectory[dim][-1])

        return new_trajectory

    @staticmethod
    def _compress_chunk(dataframe: Union[pd.DataFrame, PTRAILDataFrame],
                        metric: Text = 'TR', alpha: float = 1, verbose: bool = False):
        """
            Given the dataframe, compress each trajectory in it based on the user-provided
            metric.

            Note
            ----
                Even though this method itself is sufficient to compress trajectories on its
                own, do not use it directly as the compression module has a method called
                `compress_trajectories` that uses code parallelization to speed up the process.

            Parameters
            ----------
                dataframe: Union[pd.DataFrame, PTRAILDataFrame]
                    The dataframe containing trajectories that are to be compressed.
                metric: Text
                    The metric that is to be used in compression.
                alpha: float
                    The compression threshold to be used.
                verbose: bool
                    Flag of whether the method needs to print status of running or not.

            Returns
            -------
                PTRAILDataFrame:
                    The original dataset after compression, converted into a PTRAILDataFrame.
        """
        # All the available metric that we have.
        METRICS = {
            'TR': FormulaLog.synchronous_euclidean_distance,
            'DP': FormulaLog.perpendicular_distance,
            'SP': FormulaLog.absolute_speed_value,
        }

        # Get the metric that needs to be used as specified by the user.
        calc_func = METRICS[metric]

        # Get the list of trajectory Ids.
        traj_ids = dataframe.traj_id.unique()
        compression_rate = np.array([])

        # Start the compression procedure.
        if verbose:
            print(f"Compressing with {metric} and factor {alpha}.")

        result = []
        for i in range(len(traj_ids)):
            if verbose:
                print(f"\tCompressing {i + 1} of {len(traj_ids)}")

            # Get the current trajectory.
            # curr_traj = dataframe.loc[dataframe['traj_id'] == traj_ids[i]]
            curr_traj = Conversions.pandas_to_dict(
                dataframe.reset_index().loc[dataframe.reset_index()['traj_id'] == traj_ids[i]])
            curr_traj = curr_traj[list(curr_traj.keys())[0]]

            # Get the time in seconds.
            traj_time = curr_traj['DateTime'].astype('datetime64[ns]')
            traj_time = np.hstack((0, np.diff(traj_time).cumsum().astype('float')))
            traj_time /= traj_time.max()

            # Now do the actual compression procedure.
            compressed = curr_traj
            # try:
            max_epsilon, idx, epsilon = Helpers.calculate_metric_bw_points(curr_traj, traj_time, calc_func)
            dim_set = curr_traj.keys()
            compressed = Helpers.compress_individual(curr_traj, dim_set, traj_time,
                                                     calc_func, epsilon * alpha)
            # except:
            #     print(f"\t\tIt was not possible to compress the trajectory {traj_ids[i]} of length {len(curr_traj)}")

            # Calculate compression rate.
            compressed['DateTime'] = compressed['DateTime'].astype('datetime64[ns]')
            result.append(Conversions.dict_to_pandas({traj_ids[i]: compressed}))

            if verbose:
                print(f"\tlength before: {len(curr_traj['lat'])}, length now: {len(compressed['lat'])},"
                      f" reduction of: {1 - len(compressed) / len(curr_traj)}")

            compression_rate = np.append(compression_rate, 1 - (len(compressed) / len(curr_traj)))

        return pd.concat(result, axis=0, ignore_index=True)

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
        num = os.cpu_count()
        num = int(num)
        NUM_CPU = ceil((num * 2) / 3)

        # Integer divide the total number of Trajectory IDs by the number of available CPUs
        # and square the number because if too many partitions are made, then it does more
        # harm than good for the execution speed. The factor of 1 is added to avoid errors
        # when the integer division yields a 0.
        factor = ((size // NUM_CPU) ** 2) + 1

        # Return the factor if it is less than 100 otherwise return 100.
        # This factor hence is capped at 100.
        return factor if factor < size//10 else size//10

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
                dataframe: PTRAILDataFrame
                The dataframe that is to be split.

            Returns
            -------
                list:
                    The list containing smaller dataframe chunks.
        """
        # First, create a list containing all the ids of the dataset and then further divide that
        # list items and split it into sub-lists of ids equal to split_factor.
        ids_ = list(dataframe.reset_index().traj_id.value_counts().keys())

        # Get the ideal number of IDs by which the dataframe is to be split.
        split_factor = Helpers._get_partition_size(len(ids_))
        ids_ = [ids_[i: i + split_factor] for i in range(0, len(ids_), split_factor)]

        # Now split the dataframes based on set of Trajectory ids.
        # As of now, each smaller chunk is supposed to have dataset of 100
        # trajectory IDs max
        df_chunks = [dataframe.reset_index().loc[dataframe.reset_index()[const.TRAJECTORY_ID].isin(ids_[i])]
                     for i in range(len(ids_))]
        return df_chunks
