"""
    This compression module contains various techniques used
    for compression of trajectories.
    It is adapted from: https://github.com/marthadais/TrajectoriesCompressionAnalysis/blob/main/src/compression.py
    Credits to the Dr. Dais.

    | Author: Yaksh J Haranwala
"""
import itertools
import multiprocessing
import os
from math import ceil
from typing import Union, Text

import pandas as pd

import ptrail.utilities.constants as const
from ptrail.core.TrajectoryDF import PTRAILDataFrame
from ptrail.regularization.helpers import Helpers as helpers

NUM_CPU = ceil((os.cpu_count() * 9) / 10)


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
        # splitting the dataframe according to trajectory ids
        df_chunks = helpers._df_split_helper(dataframe)

        # Create a pool and perform the multiprocessing task.
        pool = multiprocessing.Pool(NUM_CPU)
        args = zip(df_chunks, itertools.repeat(metric), itertools.repeat(alpha), itertools.repeat(verbose))
        result = pool.starmap(helpers._compress_chunk, args)
        pool.close()
        pool.join()

        # Now lets join all the smaller partitions and then add the Distance to the
        # specific point column.
        final_df = pd.concat(result)

        # return the answer dataframe converted to PTRAILDataFrame.
        return PTRAILDataFrame(data_set=final_df,
                               latitude=const.LAT,
                               longitude=const.LONG,
                               datetime=const.DateTime,
                               traj_id=const.TRAJECTORY_ID)
