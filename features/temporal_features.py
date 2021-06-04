"""
    The datetime_based  module contains all the features of the library
    that calculates several features based on the DateTime provided in
    the data. It is to be noted that most of the functions in this module
    calculate the features and then add the results to an entirely new
    column with a new column header. It is to be also noted that a lot of
    these features are inspired from the PyMove library and we are
    crediting the PyMove creators with them.

    @authors Yaksh J Haranwala, Salman Haidri
    @date 22 May, 2021
    @version 1.0
    @credits PyMove creators
"""
import multiprocessing
from typing import Optional, Text

import pandas as pd

from core.TrajectoryDF import NumPandasTraj
from features.helper_functions import Helpers
from utilities import constants as const


class TemporalFeatures:
    @staticmethod
    def create_date_column(dataframe: NumPandasTraj):
        """
            Create a Date column in the dataframe given.

            Parameters
            ----------
                dataframe: core.TrajectoryDF.NumPandasTraj
                    The NumPandasTraj on which the creation of date column is to be done.

            Returns
            -------
                core.TrajectoryDF.NumPandasTraj
                    The dataframe containing the resultant column.
        """
        # Split the entire data set into chunks of 75000 rows each
        # so that we can work on each separate part in parallel.
        split_list = []
        for i in range(0, len(dataframe), 75000):
            split_list.append(dataframe.reset_index(drop=False).iloc[i:i + 75000])

        method_pool = multiprocessing.Pool(len(split_list))  # Create a pool of processes.

        # Now run the date helper method in parallel with the dataframes in split_list
        # and the new dataframes with date columns are stored in the variable result which is of type Mapper.
        result = method_pool.map(Helpers._date_helper, split_list)

        ans = pd.concat(result)  # Merge all the smaller chunks together.

        # Now depending on the value of inplace, return the required data structure.
        return NumPandasTraj(ans, const.LAT, const.LONG, const.DateTime, const.TRAJECTORY_ID)

    @staticmethod
    def create_time_column(dataframe: NumPandasTraj):
        """
            From the DateTime column already present in the data, extract only the time
            and then add another column containing just the time.
            Parameters
            ----------
                dataframe: NumPandasTraj
                    The DaskTrajectoryDF on which the creation of the time column is to be done.

            Returns
            -------
                core.TrajectoryDF.NumPandasTraj
                    The dataframe containing the resultant column.

        """
        df_split_list = []  # A list for storing the split dataframes.

        # Now, we are going to split the dataframes into chunks of 75000 rows each.
        # This is done in order to create processes later and then feed each process
        # a separate dataframe and calculate the results in parallel.
        for i in range(0, len(dataframe), 75000):
            df_split_list.append(dataframe.reset_index(drop=False).iloc[i:i + 75000])

        # Now, create a pool of processes which has a number of processes
        # equal to the number of smaller chunks of the original dataframe.
        # Then, calculate the result on each separate part in parallel and store it in
        # the result variable which is of type Mapper.
        pool = multiprocessing.Pool(len(df_split_list))
        result = pool.map(Helpers._time_helper, df_split_list)

        time_containing_df = pd.concat(result)  # Now join all the smaller pieces together.

        # Now check whether the user wants the result applied to the original dataframe or
        # wants a separate new dataframe. If the user wants a separate new dataframe, then
        # a pandas dataframe is returned instead of NumTrajectoryDF.
        return NumPandasTraj(time_containing_df, const.LAT, const.LONG, const.DateTime, const.TRAJECTORY_ID)

    @staticmethod
    def create_day_of_week_column(dataframe: NumPandasTraj):
        """
            Create a column called Day_Of_Week which contains the day of the week
            on which the trajectory point is recorded. This is calculated on the basis
            of timestamp recorded in the data.

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The dataframe containing the entire data on which the operation is to be performed

            Returns
            -------
                core.TrajectoryDF.NumPandasTraj
                    The dataframe containing the resultant column.
        """
        chunk_list = []  # A list for storing the split dataframes.

        # Now lets split the entire dataframe into chunks of 75000 row each so that
        # we can run the calculations on each smaller chunk in parallel.
        for i in range(0, len(dataframe), 75000):
            chunk_list.append(dataframe.reset_index(drop=False).loc[i:i + 75000])

        # Now lets create a pool of processes which will then create processes equal to
        # the number of smaller chunks of the original dataframe and will calculate
        # the result on each of the smaller chunk.
        pool_of_processes = multiprocessing.Pool(len(chunk_list))
        results = pool_of_processes.map(Helpers._day_of_week_helper, chunk_list)

        final_df = pd.concat(results)
        return NumPandasTraj(final_df, const.LAT, const.LONG, const.DateTime, const.TRAJECTORY_ID)

    @staticmethod
    def create_weekend_indicator_column(dataframe: NumPandasTraj):
        """
            Create a column called Weekend which indicates whether the point data is collected
            on either a Saturday or a Sunday.

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The dataframe on which the operation is to be performed.

            Returns
            -------
                core.TrajectoryDF.NumPandasTraj
                    The dataframe containing the resultant column if inplace.

        """
        parts = []

        # Now lets split the entire dataframe into chunks of 75000 row each so that
        # we can run the calculations on each smaller chunk in parallel.
        for i in range(0, len(dataframe), 75000):
            parts.append(dataframe.reset_index(drop=False).loc[i:i + 75000])

        # Now lets create a pool of processes and run the weekend calculator function
        # on all the smaller parts of the original dataframe and store their results.
        mp_pool = multiprocessing.Pool(len(parts))
        results = mp_pool.map(Helpers._weekend_helper, parts)

        # Now, lets merge all the smaller parts together and then return the results based on
        # the value of the inplace parameter.
        final_df = pd.concat(results)
        return NumPandasTraj(final_df, const.LAT, const.LONG, const.DateTime, const.TRAJECTORY_ID)

    @staticmethod
    def create_time_of_day_column(dataframe: NumPandasTraj):
        """
            Create a Time_Of_Day column in the dataframe which indicates at what time of the
            day was the point data captured.
            Note: The divisions of the day based on the time are provided in the utilities.constants module.

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The dataframe on which the calculation is to be done.

            Returns
            -------
                core.TrajectoryDF.NumPandasTraj
                    The dataframe containing the resultant column.

        """
        divisions = []

        # Now lets split the entire dataframe into chunks of 75000 row each so that
        # we can run the calculations on each smaller chunk in parallel.
        for i in range(0, len(dataframe), 75000):
            divisions.append(dataframe.reset_index(drop=False).loc[i:i + 75000])

        # Now lets create a pool of processes and run the weekend calculator function
        # on all the smaller parts of the original dataframe and store their results.
        multi_pool = multiprocessing.Pool(len(divisions))
        results = multi_pool.map(Helpers._time_of_day_helper, divisions)

        # Now, lets merge all the smaller parts together and then return the results based on
        # the value of the inplace parameter.
        final_df = pd.concat(results)
        return NumPandasTraj(final_df, const.LAT, const.LONG, const.DateTime, const.TRAJECTORY_ID)

    @staticmethod
    def get_traj_duration(dataframe: NumPandasTraj, traj_id: Optional[Text] = None):
        """
            Accessor method for the duration of a trajectory specified by the user.
            Note: If no trajectory ID is given by the user, then the duration of the entire
                  data set is given i.e., the difference between the max time in the dataset
                  and the min time in the dataset.

            Parameters
            ----------
                dataframe: core.TrajectoryDF.NumPandasTraj
                    The dataframe containing the resultant column if inplace is True.
                traj_id: Optional[Text]
                    The trajectory id for which the duration is required.

            Returns
            -------
                pandas.TimeDelta
                    The trajectory duration.
        """
        if traj_id is None:
            return dataframe.reset_index()['DateTime'].max() - dataframe.reset_index()['DateTime'].min()
        else:
            small = dataframe.reset_index().loc[
                dataframe.reset_index()[const.TRAJECTORY_ID] == traj_id, [const.DateTime]]
            if len(small) == 0:
                return f"No {traj_id} exists in the given data. Please try again."
            else:
                return small.max() - small.min()
