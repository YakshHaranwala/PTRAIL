"""
    | 1. The temporal_features module contains all the features of the library
         that calculates several features based on the DateTime provided in
         the data.
    | 2. It is to be noted that most of the functions in this module
         calculate the features and then add the results to an entirely new
         column with a new column header.
    | 3. It is to be also noted that a lot of these features are inspired from
         the PyMove library and we are crediting the PyMove creators with them.

    | Authors: Yaksh J Haranwala, Salman Haidri

    References
    ----------
        Arina De Jesus Amador Monteiro Sanches. “Uma Arquitetura E Imple-menta ̧c ̃ao Do M ́odulo De
        Pr ́e-processamento Para Biblioteca Pymove”.Bachelor’s thesis. Universidade Federal Do Cear ́a, 2019
"""
import itertools
import multiprocessing
from typing import Optional, Text

import numpy as np
import pandas as pd

from ptrail.core.TrajectoryDF import PTRAILDataFrame
from ptrail.features.helper_functions import Helpers as helpers
from ptrail.utilities import constants as const


class TemporalFeatures:
    @staticmethod
    def create_date_column(dataframe: PTRAILDataFrame):
        """
            From the DateTime column already present in the data, extract only the date
            and then add another column containing just the date.

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The PTRAILDataFrame Dataframe on which the creation of the time column is to be done.

            Returns
            -------
                PTRAILDataFrame:
                    The dataframe containing the resultant Date column.

        """
        df = dataframe.reset_index()

        # From the DateTime value extract the dates and store them in Date column
        df['Date'] = df[const.DateTime].dt.date

        # Return the dataframe by converting it to PTRAILDataFrame
        return PTRAILDataFrame(df.reset_index(drop=True),
                               const.LAT, const.LONG, const.DateTime, const.TRAJECTORY_ID)

    @staticmethod
    def create_time_column(dataframe: PTRAILDataFrame):
        """
            From the DateTime column already present in the data, extract only the time
            and then add another column containing just the time.

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The PTRAILDataFrame Dataframe on which the creation of the time column is to be done.

            Returns
            -------
                PTRAILDataFrame
                    The dataframe containing the resultant Time column.

        """
        dataframe = dataframe.reset_index()

        # From the DateTime column extract the time and store them in the Time column
        dataframe['Time'] = dataframe[const.DateTime].dt.time

        # Return the dataframe by converting it into PTRAILDataFrame type
        return PTRAILDataFrame(dataframe, const.LAT, const.LONG, const.DateTime, const.TRAJECTORY_ID)

    @staticmethod
    def create_day_of_week_column(dataframe: PTRAILDataFrame):
        """
            Create a column called Day_Of_Week which contains the day of the week
            on which the trajectory point is recorded. This is calculated on the basis
            of timestamp recorded in the data.

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The dataframe containing the entire data on which the operation is to be performed

            Returns
            -------
                PTRAILDataFrame
                    The dataframe containing the resultant Day_of_week column.
        """
        dataframe = dataframe.reset_index()

        # From the DateTime column extract the time and store them in the Time column
        dataframe['Day_Of_Week'] = dataframe[const.DateTime].dt.day_name()

        # Return the dataframe by converting it into PTRAILDataFrame type
        return PTRAILDataFrame(dataframe, const.LAT, const.LONG, const.DateTime, const.TRAJECTORY_ID)

    @staticmethod
    def create_weekend_indicator_column(dataframe: PTRAILDataFrame):
        """
            Create a column called Weekend which indicates whether the point data is collected
            on either a Saturday or a Sunday.

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The dataframe on which the operation is to be performed.

            Returns
            -------
                PTRAILDataFrame
                    The dataframe containing the resultant Weekend column.

            References
            ----------
                Arina De Jesus Amador Monteiro Sanches. 'Uma Arquitetura E Imple-menta ̧c ̃ao
                Do M ́odulo De Pr ́e-processamento Para Biblioteca Pymove'.Bachelor’s thesis.
                Universidade Federal Do Cear ́a, 2019.

        """
        # Check if Day_of_Week column is present in the dataframe
        if 'Day_Of_Week' in dataframe.columns:
            dataframe = dataframe.reset_index()
            # store the weekend days in a series
            fd = np.logical_or(dataframe['Day_Of_Week'] == const.WEEKEND[0],
                               dataframe['Day_Of_Week'] == const.WEEKEND[1])
            # store the index of the weekends
            index_fd = dataframe[fd].index
            # initialize the Weekend column with False and then update all the indexes which
            # indicates that it's a weekend
            dataframe['Weekend'] = False
            dataframe.at[index_fd, 'Weekend'] = True
            # Return the dataframe by converting it into PTRAILDataFrame
        else:
            # If the dataframe does not contain the Day_Of_Week column already,
            # then call the method and create the column.
            dataframe = TemporalFeatures.create_day_of_week_column(dataframe)
            dataframe = dataframe.reset_index()
            # store the weekend days in a series
            fd = np.logical_or(dataframe['Day_Of_Week'] == const.WEEKEND[0],
                               dataframe['Day_Of_Week'] == const.WEEKEND[1])
            # store the index of the weekends
            index_fd = dataframe[fd].index
            # initialize the Weekend column with False and then update all the indexes which
            # indicates that it's a weekend
            dataframe['Weekend'] = False
            dataframe.at[index_fd, 'Weekend'] = True
            # Return the dataframe by converting it into PTRAILDataFrame

        return PTRAILDataFrame(dataframe, const.LAT, const.LONG, const.DateTime, const.TRAJECTORY_ID)

    @staticmethod
    def create_time_of_day_column(dataframe: PTRAILDataFrame):
        """
            Create a Time_Of_Day column in the dataframe using parallelization which indicates at what time of the
            day was the point data captured.
            Note: The divisions of the day based on the time are provided in the utilities.constants module.

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The dataframe on which the calculation is to be done.

            Returns
            -------
                PTRAILDataFrame
                    The dataframe containing the resultant Time_Of_Day column.

            References
            ----------
                Arina De Jesus Amador Monteiro Sanches. 'Uma Arquitetura E Imple-menta ̧c ̃ao
                Do M ́odulo De Pr ́e-processamento Para Biblioteca Pymove'.Bachelor’s thesis.
                Universidade Federal Do Cear ́a, 2019.

        """
        dataframe = dataframe.reset_index()
        # Extract the hours from the Datetime column and then create a list of conditions for the
        # different periods of time
        hours = dataframe[const.DateTime].dt.hour
        conditions = [
            (hours >= 0) & (hours <= 4),
            (hours > 4) & (hours <= 8),
            (hours > 8) & (hours <= 12),
            (hours > 12) & (hours <= 16),
            (hours > 16) & (hours <= 20),
            (hours > 20) & (hours <= 24)
        ]
        # Map the conditions to the different periods of  the day
        dataframe['Time_Of_Day'] = np.select(conditions, const.TIME_OF_DAY)
        return PTRAILDataFrame(dataframe, const.LAT, const.LONG, const.DateTime, const.TRAJECTORY_ID)

    @staticmethod
    def get_traj_duration(dataframe: PTRAILDataFrame, traj_id: Optional[Text] = None):
        """
            Accessor method for the duration of a trajectory specified by the user.

            Note
            ----
                If no trajectory ID is given by the user, then the duration of each
                unique trajectory is calculated.

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The dataframe containing the resultant column if inplace is True.
                traj_id: Optional[Text]
                    The trajectory id for which the duration is required.

            Returns
            -------
                pandas.TimeDelta:
                    The trajectory duration.
                pandas.core.dataframe.DataFrame:
                    The dataframe containing the duration of all trajectories in the dataset.
        """
        dataframe = dataframe.reset_index()
        if traj_id is None:
            ids_ = dataframe[const.TRAJECTORY_ID].value_counts(ascending=True).keys().to_list()

            split_factor = helpers._get_partition_size(len(ids_))
            ids_ = [ids_[i: i + split_factor] for i in range(0, len(ids_), split_factor)]

            mp_pool = multiprocessing.Pool(len(ids_))
            results = mp_pool.starmap(helpers.traj_duration_helper, zip(itertools.repeat(dataframe), ids_))
            mp_pool.close()
            mp_pool.join()

            results = pd.concat(results).sort_values(const.TRAJECTORY_ID)
            return results
        else:
            small = dataframe.loc[dataframe[const.TRAJECTORY_ID] == traj_id, [const.DateTime]]
            if len(small) == 0:
                return f"No {traj_id} exists in the given data. Please try again."
            else:
                return small.max() - small.min()

    @staticmethod
    def get_start_time(dataframe: PTRAILDataFrame, traj_id: Optional[Text] = None):
        """
            Get the starting time of the trajectory.

            Note
            ----
                If the trajectory ID is not specified by the user, then by default,
                the starting times of all the trajectory IDs in the data are
                returned.

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The dataframe on which the operations are to be performed.
                traj_id: Optional[Text]
                    The trajectory for which the start time is required.

            Returns
            -------
                pandas.DateTime:
                    The start time of a single trajectory.
                pandas.core.dataframe.DataFrame
                    Pandas dataframe containing the start time of all the trajectories
                    present in the data when the user hasn't asked for a particular
                    trajectory's start time.
        """
        dataframe = dataframe.reset_index()
        if traj_id is None:
            # First, create a list containing all the ids of the data and then further divide that
            # list items and split it into sub-lists of 100 ids each if there are more than 100 ids.
            ids_ = dataframe[const.TRAJECTORY_ID].value_counts(ascending=True).keys().to_list()

            # Get the ideal number of IDs by which the dataframe is to be split.
            split_factor = helpers._get_partition_size(len(ids_))
            ids_ = [ids_[i: i + split_factor] for i in range(0, len(ids_), split_factor)]

            # Now, create a multiprocessing pool and then run processes in parallel
            # which calculate the start times for a smaller set of IDs only.
            mp_pool = multiprocessing.Pool(len(ids_))
            results = mp_pool.starmap(helpers.start_time_helper, zip(itertools.repeat(dataframe), ids_))
            mp_pool.close()
            mp_pool.join()

            # Concatenate all the smaller dataframes and return the answer.
            results = pd.concat(results).sort_values(const.TRAJECTORY_ID)
            return results
        else:
            filt = dataframe.loc[dataframe[const.TRAJECTORY_ID] == traj_id]
            filt_two = filt.loc[filt[const.DateTime] == filt[const.DateTime].min()]
            return filt_two[const.DateTime].iloc[0]

    @staticmethod
    def get_end_time(dataframe: PTRAILDataFrame, traj_id: Optional[Text] = None):
        """
            Get the ending time of the trajectory.

            Note
            ----
                If the trajectory ID is not specified by the user, then by default,
                the ending times of all the trajectory IDs in the data are
                returned.

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The dataframe on which the operations are to be performed.
                traj_id: Optional[Text]
                    The trajectory for which the end time is required.

            Returns
            -------
                pandas.DateTime:
                    The end time of a single trajectory.
                pandas.core.dataframe.DataFrame
                    Pandas dataframe containing the end time of all the trajectories
                    present in the data when the user hasn't asked for a particular
                    trajectory's end time.
        """
        dataframe = dataframe.reset_index()
        if traj_id is None:
            # First, create a list containing all the ids of the data and then further divide that
            # list items and split it into sub-lists of 100 ids each if there are more than 100 ids.
            ids_ = dataframe[const.TRAJECTORY_ID].value_counts(ascending=True).keys().to_list()

            # Get the ideal number of IDs by which the dataframe is to be split.
            split_factor = helpers._get_partition_size(len(ids_))
            ids_ = [ids_[i: i + split_factor] for i in range(0, len(ids_), split_factor)]

            # Now, create a multiprocessing pool and then run processes in parallel
            # which calculate the end times for a smaller set of IDs only.
            mp_pool = multiprocessing.Pool(len(ids_))
            results = mp_pool.starmap(helpers.end_time_helper, zip(itertools.repeat(dataframe), ids_))
            mp_pool.close()
            mp_pool.join()

            # Concatenate all the smaller dataframes and return the answer.
            results = pd.concat(results).sort_values(const.TRAJECTORY_ID)
            return results
        else:
            filt = dataframe.loc[dataframe[const.TRAJECTORY_ID] == traj_id]
            filt_two = filt.loc[filt[const.DateTime] == filt[const.DateTime].max()]
            return filt_two[const.DateTime].iloc[0]

    @staticmethod
    def generate_temporal_features(dataframe: PTRAILDataFrame):
        """
            Generate all the temporal features with a single call of this function.
            
            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The dataframe on which the features are to be generated.
            
            Returns
            -------
                PTRAILDataFrame:
                    The dataframe enriched with Temporal Features.
        """
        to_return = TemporalFeatures.create_date_column(dataframe)
        to_return = TemporalFeatures.create_time_column(to_return)
        to_return = TemporalFeatures.create_day_of_week_column(to_return)
        to_return = TemporalFeatures.create_weekend_indicator_column(to_return)
        to_return = TemporalFeatures.create_time_of_day_column(to_return)

        return to_return