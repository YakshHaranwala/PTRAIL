"""
    The statistics module has several functionalities that calculate kinematic
    statistics of the trajectory, split trajectories, pivot dataframes etc.
    The main purpose of this module is to get the dataframe ready for Machine
    Learning tasks such as clustering, calssification etc.

    | Author: Yaksh J Haranwala

"""
import itertools
from typing import Optional

import pandas as pd

from ptrail.core.TrajectoryDF import PTRAILDataFrame
from ptrail.features.kinematic_features import KinematicFeatures
from ptrail.preprocessing.helpers import Helpers as helpers
import ptrail.utilities.constants as const
from math import ceil

import multiprocessing
import os

num = os.cpu_count()
NUM_CPU = ceil((num * 2) / 3)


class Statistics:
    @staticmethod
    def segment_traj_by_days(df: PTRAILDataFrame, num_days):
        """
            Given a dataframe containing trajectory data, segment all
            the trajectories by each week.

            Parameters
            ----------
                df: PTRAILDataFrame
                    The dataframe containing trajectory data.
                num_days: int
                    The number of days that each segment is supposed to have.

            Returns
            -------
                pandas.core.dataframe.DataFrame:
                    The dataframe containing segmented trajectories
                    with a new column added called segment_id
        """
        # splitting the dataframe according to trajectory ids
        df_chunks = helpers._df_split_helper(dataframe=df.reset_index())

        # Here, create 2/3rds number of processes as there are in the system. Some CPUs are
        # kept free at all times in order to not block up the system.
        # (Note: The blocking of system is mostly prevalent in Windows and does not happen very often
        # in Linux. However, out of caution some CPUs are kept free regardless of the system.)
        pool = multiprocessing.Pool(NUM_CPU)
        results = pool.starmap(helpers.split_traj_helper, zip(df_chunks, itertools.repeat(num_days)))
        pool.close()
        pool.join()

        to_return = pd.concat(results).reset_index().set_index(['traj_id', 'seg_id', 'DateTime'])

        return to_return.drop(columns=['index'])

    @staticmethod
    def generate_kinematic_stats(dataframe: PTRAILDataFrame, target_col_name: str, segmented: Optional[bool] = False):
        """
            Generate the statistics of kinematic features for each unique trajectory in
            the dataframe.

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The dataframe containing the trajectory data.
                target_col_name: str
                    This is the 'y' value that is used for ML tasks, this is
                    asked to append the species back at the end.
                segmented: Optional[bool]
                    Indicate whether the trajectory has segments or not.

            Returns
            -------
                pandas.core.dataframe.DataFrame:
                    A pandas dataframe containing stats for all kinematic features for
                    each unique trajectory in the dataframe.
        """
        # Generate kinematic features on the entire dataframe.
        ptdf = KinematicFeatures.generate_kinematic_features(dataframe)

        # Then, lets break down the entire dataframe into pieces containing data of
        # 1 trajectory in each piece.
        ids_ = list(dataframe.reset_index().traj_id.value_counts().keys())
        df_chunks = []

        if segmented:
            segments = ptdf.reset_index()['seg_id'].unique()
            for i in range(len(ids_)):
                for seg in segments:
                    df = ptdf.reset_index().loc[(ptdf.reset_index()['traj_id'] == ids_[i])
                                                & (ptdf.reset_index()['seg_id'] == seg)]
                    if len(df) > 0:
                        df_chunks.append(df)
                    else:
                        continue
        else:
            for i in range(len(ids_)):
                small_df = ptdf.reset_index().loc[ptdf.reset_index()[const.TRAJECTORY_ID] == ids_[i]]
                df_chunks.append(small_df)

        # Here, create 2/3rds number of processes as there are in the system. Some CPUs are
        # kept free at all times in order to not block up the system.
        # (Note: The blocking of system is mostly prevalent in Windows and does not happen very often
        # in Linux. However, out of caution some CPUs are kept free regardless of the system.)
        mp_pool = multiprocessing.Pool(NUM_CPU)
        results = mp_pool.starmap(helpers.stats_helper, zip(df_chunks,
                                                            itertools.repeat(target_col_name),
                                                            itertools.repeat(segmented)))
        mp_pool.close()
        mp_pool.join()

        return pd.concat(results)

    @staticmethod
    def pivot_stats_df(dataframe, target_col_name: str, segmented: Optional[bool] = False):
        """
            Given a dataframe with stats present in it, melt the dataframe to make it
            ready for ML tasks. This is specifically for melting the type of dataframe
            generated by the generate_kinematic_stats() function of the kinematic_features
            module.

            Check the kinematic_features module for further details about the dataframe
            expected.

            Parameters
            ----------
                dataframe: pd.core.dataframe.DataFrame
                    The dataframe containing stats.
                target_col_name: str
                    This is the 'y' value that is used for ML tasks, this is
                    asked to append the species back at the end.
                segmented: Optional[bool]
                    Indicate whether the trajectory has segments or not.

            Returns
            -------
                pd.core.dataframe.DataFrame:
                    The dataframe above which is pivoted and has rows converted to columns.
        """
        # Get all the unique trajectory IDs.
        ids_ = list(dataframe.index.get_level_values('traj_id').unique())

        final_chunks = []
        if not segmented:
            for val in ids_:
                # separated the data for each trajectory id.
                small = dataframe.loc[dataframe.index.get_level_values('traj_id') == val].reset_index().set_index(
                    'traj_id')

                # Get the target value out and drop the target column.
                target = small[target_col_name].iloc[0]
                small = small.drop(columns=['Species'])

                # Pivot the table now and adjust the column names.
                pivoted = small.reset_index().pivot_table(index='traj_id', columns='Columns')
                pivoted.columns = pivoted.columns.map('_'.join).str.strip('|')

                # Assign the target column again.
                pivoted[target_col_name] = target
                final_chunks.append(pivoted)
        else:
            for val in ids_:
                segments = dataframe.index.get_level_values('seg_id').unique()
                for seg in segments:
                    small = dataframe.loc[(dataframe.index.get_level_values('traj_id') == val) &
                                          (dataframe.index.get_level_values('seg_id') == seg)].reset_index().set_index(['traj_id', 'seg_id'])

                    if len(small) <= 0:
                        continue

                    # Get the target value out and drop the target column.
                    target = small[target_col_name].iloc[0]
                    small = small.drop(columns=['Species'])

                    # Pivot the table now and adjust the column names.
                    pivoted = small.reset_index().pivot_table(index=['traj_id', 'seg_id'], columns='Columns')
                    pivoted.columns = pivoted.columns.map('_'.join).str.strip('|')

                    # Assign the target column again.
                    pivoted[target_col_name] = target
                    final_chunks.append(pivoted)

        # Concatenate the smaller chunks and reorder the columns.
        to_return = pd.concat(final_chunks)

        # Store the correct order of the columns to a variable and add the name
        # of the target column to the end of it.
        cols = const.ORDERED_COLS
        cols.append(target_col_name)

        # Reorder the final DF and return it.
        return to_return[cols]
