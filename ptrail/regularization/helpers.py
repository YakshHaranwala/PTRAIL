"""
    Warning
    -------
        | 1. None of the methods in this module should be used directly while performing operations on data.
        | 2. These methods are helpers for the regularization methods in the regularization.py
             module and hence run linearly and not in parallel which will result in slower execution time.

    Besides the interpolation helpers, there are also general utilities which
    are used in splitting up dataframes for running the code in parallel.

    | Author: Yaksh J. Haranwala
"""
from typing import Text, Union

import numpy as np
import pandas as pd

from ptrail.core.TrajectoryDF import PTRAILDataFrame


class Helpers:
    @staticmethod
    def calculate_metric_bw_points(traj: Union[pd.DataFrame, PTRAILDataFrame], metric: function):
        """
            Given a trajectory and the metric, calculate the metric
            for all the points in the trajectory.

            Parameters
            ----------
                traj:
                    The trajectory object for which the metric is to be
                    calculated.
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

        # Get the first and the last points.
        start = traj['lat'][0], traj['lon'][0], traj['DateTime'][0]
        end = traj['lat'][-1], traj['lon'][-1], traj['DateTime'][-1]

        # Find the max distance and index and return it.
        for i in range(1, len(traj) - 1):
            # Get the mid-point at index i.
            middle = traj['lat'][i], traj['lon'][i], traj['DateTime'][i]

            # Calculate the distance and append it to the distances array.
            dist = metric(start, middle, end)
            distances = np.append(distances, dist)
            if dist > dist_max:
                d_max = d_max
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

        # Get the time in seconds
        d_max, idx, _ = Helpers.calculate_metric_bw_points(trajectory, calc_func)
        trajectory['DateTime'] = trajectory['DateTime'].astype(str)

        print(f"\tepsilon: {epsilon}, d_max: {d_max}, index: {idx}, traj_len: {len(trajectory)}")
        if d_max > epsilon:
            traj1 = {}
            traj2 = {}

            for dim in dim_set:
                traj1[dim] = trajectory[dim][0:idx]
                traj2[dim] = trajectory[dim][idx:]

                # Compress the parts created above
                recResults1 = traj1
                if len(traj1['lat']) > 2:
                    recResults1 = Helpers.compress_individual(traj1, dim_set, traj_time[0:idx], calc_func, epsilon)

                recResults2 = traj2
                if len(traj2['lat']) > 2:
                    recResults2 = Helpers.compress_individual(traj2, dim_set, traj_time[idx:], calc_func, epsilon)

                for dim in dim_set:
                    new_trajectory[dim] = np.append(new_trajectory[dim], recResults1[dim])
                    new_trajectory[dim] = np.append(new_trajectory[dim], recResults2[dim])

        else:
            trajectory['DateTime'] = trajectory['DateTime'].astype(str)
            for dim in dim_set:
                new_trajectory[dim] = np.append(new_trajectory[dim], trajectory[dim][0])
                if len(trajectory) > 1:
                    new_trajectory[dim] = np.append(new_trajectory[dim], trajectory[dim][-1])

        return new_trajectory
