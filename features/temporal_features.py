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
    def date_helper(dataframe: NumPandasTraj, inplace=False):
        """
            From the DateTime column already present in the data, extract only the date
            and then add another column containing just the date.

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The DaskTrajectoryDF on which the creation of the date column is to be done.
                inplace: bool
                    Whether to apply changes to the given dataframe or just return a new pandas DF.

            Returns
            -------
                core.TrajectoryDF.NumPandasTraj
                    The dataframe containing the date column.
                pandas.core.dataframe.DataFrame
                    returned if the value of inplace parameter is False.

        """
        time = "%Y-%m-%d"  # Format of the date.
        # print(type(dataframe))
        if inplace:
            data = dataframe
        else:
            data = dataframe.copy()

        gen = data.reset_index(drop=False)['DateTime'].iteritems()
        gen = list(gen)

        # Now, we extract the Date from all the time values.
        for i in range(len(gen)):
            gen[i] = gen[i][1].strftime(time)

        data['Date'] = gen  # Assign the date column to the dataframe.
        return data

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
        # Depending on the value of inplace parameter, select the dataset to work on.
        if inplace:
            data = dataframe
        else:
            data = dataframe.copy()

        # Split the entire data set into chunks of 33000 rows each
        # so that we can work on each separate part in parallel.
        split_list = []
        for i in range(0, len(dataframe), 33000):
            split_list.append(data.reset_index(drop=False).iloc[i:i + 33000])

        method_pool = multiprocessing.Pool(len(split_list))     # Create a pool of processes.

        # Now run the date helper method in parallel with the dataframes in split_list
        # and the new dataframes with date columns are stored in the variable result which is of type Mapper.
        result = method_pool.map(TemporalFeatures.date_helper, split_list)

        ans = pd.concat(result)     # Merge all the smaller chunks together.

        # Now depending on the value of inplace, return the required data structure.
        if inplace:
            return NumPandasTraj(ans, const.LAT, const.LONG, const.DateTime, const.TRAJECTORY_ID)
        else:
            return ans.set_index([const.DateTime, const.TRAJECTORY_ID], inplace=False)
