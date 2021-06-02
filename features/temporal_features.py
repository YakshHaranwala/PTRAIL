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
import numpy
import numpy as np
import pandas as pd
import multiprocessing

from core.TrajectoryDF import NumPandasTraj
from utilities import constants as const


class TemporalFeatures:
    @staticmethod
    def date_helper(dataframe):
        """
            This function is a helper method for the create_date_column(). The create_date_helper()
            methods delegates the actual task of creating the date to date_helper() function. What
            this function does is that it extracts the date from the DateTime column present in the
            DF and then adds a column to the DF itself, containing the date.

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The DaskTrajectoryDF on which the creation of the date column is to be done.

            Returns
            -------
                pandas.core.dataframe.DataFrame
                    The dataframe containing the date column.

        """
        date_format = "%Y-%m-%d"  # Format of the date.

        # Reset the index of the DF in order to access the date time column and then generate
        # an iterable list of the items inside the column.
        gen = dataframe.reset_index(drop=False)['DateTime'].iteritems()
        gen = list(gen)

        # Now, we extract the Date from all the time values.
        for i in range(len(gen)):
            gen[i] = gen[i][1].strftime(date_format)

        dataframe['Date'] = gen  # Assign the date column to the dataframe.
        return dataframe    # Return the dataframe with the date column inside it.

    @staticmethod
    def time_helper(dataframe):
        """
            This function is a helper method for the create_time_column(). The create_time_helper()
            methods delegates the actual task of creating the time to time_helper() function. What
            this function does is that it extracts the time from the DateTime column present in the
            DF and then adds a column to the DF itself, containing the time.

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The DaskTrajectoryDF on which the creation of the time column is to be done.

            Returns
            -------
                pandas.core.dataframe.DataFrame
                    The dataframe containing the resultant time column.

        """
        time_format = "%H:%M:%S"

        # Reset the index of the DF in order to access the date time column and then generate
        # an iterable list of the items inside the column.
        datetime = dataframe.reset_index(drop=False)['DateTime'].iteritems()
        datetime = list(datetime)

        # Now lets extract the time from the DateTime column.
        for i in range(len(datetime)):
            datetime[i] = datetime[i][1].strftime(time_format)

        dataframe['Time'] = datetime
        return dataframe

    @staticmethod
    def day_of_week_helper(dataframe):
        """
            This function is the helper function of the create_day_of_week() function. The day_of_week()
            function delegates the actual task of calculating the day of the week based on the datetime
            on this function. This function does the calculation and creates a column called Day_Of_week
            and places it in the dataframe and returns in.

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The dataframe on which calculation is to be performed.

            Returns
            -------
                pandas.core.dataframe
                    The dataframe containing the resultant Day_Of_Week column.
        """
        datetime = dataframe.reset_index(drop=False)['DateTime'].iteritems()
        datetime = list(datetime)

        for i in range(len(datetime)):
            datetime[i] = datetime[i][1].day_name()

        dataframe['Day_Of_Week'] = datetime

        return dataframe

    @staticmethod
    def create_date_column(dataframe: NumPandasTraj, inplace=False):
        """
            Create a Date column in the dataframe given.

            Parameters
            ----------
                dataframe: core.TrajectoryDF.NumPandasTraj
                    The NumPandasTraj on which the creation of date column is to be done
                inplace: bool
                    Whether to apply the results to the original dataframe or not.

            Returns
            -------
                core.TrajectoryDF.NumPandasTraj
                    When the inplace parameter is True.
                pandas.core.dataframe.DataFrame
                    When the inplace parameter is False.
        """
        # Split the entire data set into chunks of 33000 rows each
        # so that we can work on each separate part in parallel.
        split_list = []
        for i in range(0, len(dataframe), 33000):
            split_list.append(dataframe.reset_index(drop=False).iloc[i:i + 33000])

        method_pool = multiprocessing.Pool(len(split_list))     # Create a pool of processes.

        # Now run the date helper method in parallel with the dataframes in split_list
        # and the new dataframes with date columns are stored in the variable result which is of type Mapper.
        result = method_pool.map(TemporalFeatures.date_helper, split_list)

        ans = pd.concat(result)     # Merge all the smaller chunks together.

        # Now depending on the value of inplace, return the required data structure.
        if inplace:
            return NumPandasTraj(ans, const.LAT, const.LONG, const.DateTime, const.TRAJECTORY_ID)
        else:
            return ans.set_index([const.DateTime, const.TRAJECTORY_ID])

    @staticmethod
    def create_time_column(dataframe: NumPandasTraj, inplace=False):
        """
            From the DateTime column already present in the data, extract only the time
            and then add another column containing just the time.
            Parameters
            ----------
                dataframe: NumPandasTraj
                    The DaskTrajectoryDF on which the creation of the time column is to be done.
                inplace: bool
                    Whether to apply changes to the given dataframe or just return a new pandas DF.
            Returns
            -------
                core.TrajectoryDF.NumPandasTraj / pandas.core.dataframe.DataFrame
                    The dataframe containing the time column.
                    Pandas DF is returned if the value of inplace parameter is False.
        """
        df_split_list = []  # A list for storing the split dataframes.

        # Now, we are going to split the dataframes into chunks of 33000 rows each.
        # This is done in order to create processes later and then feed each process
        # a separate dataframe and calculate the results in parallel.
        for i in range(0, len(dataframe), 33000):
            df_split_list.append(dataframe.reset_index(drop=False).iloc[i:i + 33000])

        # Now, create a pool of processes which has a number of processes
        # equal to the number of smaller chunks of the original dataframe.
        # Then, calculate the result on each separate part in parallel and store it in
        # the result variable which is of type Mapper.
        pool = multiprocessing.Pool()
        result = pool.map(TemporalFeatures.time_helper, df_split_list)

        time_containing_df = pd.concat(result)  # Now join all the smaller pieces together.

        # Now check whether the user wants the result applied to the original dataframe or
        # wants a separate new dataframe. If the user wants a separate new dataframe, then
        # a pandas dataframe is returned instead of NumTrajectoryDF.
        if inplace:
            return NumPandasTraj(time_containing_df, const.LAT, const.LONG, const.DateTime, const.TRAJECTORY_ID)
        else:
            return time_containing_df.set_index([const.DateTime, const.TRAJECTORY_ID])

    @staticmethod
    def create_day_of_week_column(dataframe: NumPandasTraj, inplace=False):
        """
            Create a column called Day_Of_Week which contains the day of the week
            on which the trajectory point is recorded. This is calculated on the basis
            of timestamp recorded in the data.

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The dataframe containing the entire data on which the operation is to be performed
                inplace: bool
                    Indication of whether the results are to be applied to original DF or a new
                    one is to be returned with the calculated results.

            Returns
            -------
                core.TrajectoryDF.NumPandasTraj
                    The dataframe containing the resultant column if inplace is True.
                pandas.core.dataframe.DataFrame
                    The dataframe containing the resultant column if inplace is False
        """
        chunk_list = []     # A list for storing the split dataframes.

        # Now lets split the entire dataframe into chunks of 33000 row each so that
        # we can run the calculations on each smaller chunk in parallel.
        for i in range(0, len(dataframe), 33000):
            chunk_list.append(dataframe.reset_index(drop=False).loc[i:i + 33000])

        # Now lets create a pool of processes which will then create processes equal to
        # the number of smaller chunks of the original dataframe and will calculate
        # the result on each of the smaller chunk.
        pool_of_processes = multiprocessing.Pool()
        results = pool_of_processes.map(TemporalFeatures.day_of_week_helper, chunk_list)

        final_df = pd.concat(results)
        if inplace:
            return NumPandasTraj(final_df,  const.LAT, const.LONG, const.DateTime, const.TRAJECTORY_ID)
        else:
            return final_df.set_index([const.DateTime, const.TRAJECTORY_ID], inplace=True, drop=True)
