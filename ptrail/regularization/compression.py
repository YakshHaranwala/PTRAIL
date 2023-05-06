"""
    This compression module contains various techniques used
    for compression of trajectories.
    It is adapted from: https://github.com/marthadais/TrajectoriesCompressionAnalysis/blob/main/src/compression.py
    Credits to the Dr. Dais.

    | Author: Yaksh J Haranwala
"""
from typing import Union, Text
import numpy as np
import time

import pandas as pd

from ptrail.utilities.KinematicUtils import FormulaLog
from ptrail.core.TrajectoryDF import PTRAILDataFrame
from ptrail.regularization.helpers import Helpers
from ptrail.utilities.conversions import Conversions

# TODO: Parallelize the code and make it fit the standard PTRAIL methodology.


class Compression:
    @staticmethod
    def compress_trajectories(dataframe: Union[pd.DataFrame, PTRAILDataFrame],
                              metric: Text = 'TR', alpha: float = 1, verbose: bool = False):
        """
            Given the dataframe, compress each trajectory in it based on the user-provided
            metric.

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
                np.array:
                    Array containing the compression rate for each of the trajectory compressed.

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
        dataframe = dataframe.reset_index()
        traj_ids = dataframe['traj_id'].unique()
        compression_rate = np.array([])

        # Start the compression procedure.
        if verbose:
            print(f"Compressing with {metric} and factor {alpha}.")

        result = []
        for i in range(len(traj_ids)):
            if verbose:
                print(f"\tCompressing {i+1} of {len(traj_ids)}")

            # Get the current trajectory.
            curr_traj = dataframe.loc[dataframe['traj_id'] == traj_ids[i]]

            # Get the time in seconds.
            traj_time = curr_traj['DateTime'].astype('datetime64[ns]')
            traj_time = np.hstack((0, np.diff(traj_time).cumsum().astype('float')))
            traj_time /= traj_time.max()

            # Now do the actual compression procedure.
            # try:
            max_epsilon, idx, epsilon = Helpers.calculate_metric_bw_points(curr_traj, traj_time, calc_func)
            dim_set = curr_traj.columns
            compressed = Helpers.compress_individual(curr_traj, dim_set, traj_time,
                                                     calc_func, epsilon*alpha)
            # except:
            #     print(f"\t\tIt was not possible to compress the trajectory {traj_ids[i]} of length {len(curr_traj)}")

            # Calculate compression rate.
            compressed['DateTime'] = compressed['DateTime'].astype('datetime64[ns]')
            result.append(compressed)

            if verbose:
                print(f"\tlength before: {len(curr_traj)}, length now: {len(compressed)},"
                      f" reduction of: {1 - len(compressed)/len(curr_traj)}")

            compression_rate = np.append(compression_rate, 1 - (len(compressed) / len(curr_traj)))

        return PTRAILDataFrame(data_set=pd.concat(result, axis=0, ignore_index=True), latitude='lat', longitude='lon',
                               datetime='DateTime', traj_id='traj_id'), compression_rate
