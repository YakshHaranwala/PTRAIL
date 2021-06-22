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

import utilities.constants as const


class Helpers:
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
        # and square the number because if too many partitions are made, then it does more
        # harm than good for the execution speed. The factor of 1 is added to avoid errors
        # when the integer division yields a 0.
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
