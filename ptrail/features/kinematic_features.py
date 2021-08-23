"""
    The spatial_features module contains several functions of the library
    that calculates kinematic features based on the coordinates of points
    provided in the data. This module mostly extracts and modifies data
    collected from some existing dataframe and appends these information
    to them. Inspiration of lots of functions in this module is taken from
    the PyMove library.


    | Authors: Yaksh J Haranwala, Salman Haidri

    References
    ----------
            Arina De Jesus Amador Monteiro Sanches. “Uma Arquitetura E Imple-menta ̧c ̃ao Do M ́odulo De
            Pr ́e-processamento Para Biblioteca Pymove”.Bachelor’s thesis. Universidade Federal Do Cear ́a, 2019
"""
import itertools
import multiprocessing
import os
from math import ceil
from typing import Optional, Text

import numpy as np
import pandas as pd

from ptrail.core.TrajectoryDF import PTRAILDataFrame
from ptrail.features.helper_functions import Helpers as helpers
from ptrail.utilities import constants as const
from ptrail.utilities.DistanceCalculator import FormulaLog as calc
from ptrail.utilities.exceptions import *

num = os.cpu_count()
NUM_CPU = ceil((num * 2) / 3)


class KinematicFeatures:
    @staticmethod
    def get_bounding_box(dataframe: PTRAILDataFrame):
        """
            Return the bounding box of the Trajectory data. Essentially, the bounding box is of
            the following format:
                (min Latitude, min Longitude, max Latitude, max Longitude).

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The dataframe containing the trajectory data.

            Returns
            -------
                tuple:
                    The bounding box of the trajectory
        """
        return (
            dataframe[const.LAT].min(),
            dataframe[const.LONG].min(),
            dataframe[const.LAT].max(),
            dataframe[const.LONG].max(),
        )

    @staticmethod
    def get_start_location(dataframe: PTRAILDataFrame, traj_id=None):
        """
            Get the starting location of an object's trajectory in the data.

            Note
            ----
                If the user does not give in any traj_id, then the library,
                by default gives out the start locations of all the unique
                trajectory ids present in the data.

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The PTRAILDataFrame storing the trajectory data.
                traj_id:
                    The ID of the object whose start location is to be found.

            Returns
            -------
                tuple:
                    The (lat, longitude) tuple containing the start location.
                pandas.core.dataframe.DataFrame:
                    The dataframe containing start locations of all trajectory IDs.
        """
        # If traj_id is None, find the start times of all the unique trajectories present in the data.
        # Else first filter out a dataframe containing the given traj_id and then return the start
        # location of that point.
        dataframe = dataframe.copy().reset_index()
        if traj_id is None:
            ids_ = dataframe[const.TRAJECTORY_ID].value_counts(ascending=True).keys().to_list()
            # Get the ideal number of IDs by which the dataframe is to be split.
            split_factor = helpers._get_partition_size(len(ids_))
            ids_ = [ids_[i: i + split_factor] for i in range(0, len(ids_), split_factor)]

            # Here, create 2/3rds number of processes as there are in the system. Some CPUs are
            # kept free at all times in order to not block up the system.
            # (Note: The blocking of system is mostly prevalent in Windows and does not happen very often
            # in Linux. However, out of caution some CPUs are kept free regardless of the system.)
            mp_pool = multiprocessing.Pool(NUM_CPU)
            results = mp_pool.starmap(helpers.start_location_helper, zip(itertools.repeat(dataframe), ids_))
            mp_pool.close()
            mp_pool.join()

            # Concatenate all the smaller dataframes and return the answer.
            results = pd.concat(results)
            return results

        else:
            filt = (dataframe.loc[dataframe[const.TRAJECTORY_ID] == traj_id, [const.DateTime, const.LAT, const.LONG]])
            start_loc = (filt.loc[filt[const.DateTime] == filt[const.DateTime].min(),
                                  [const.LAT, const.LONG]]).reset_index()
            if len(start_loc) == 0:
                return f"Trajectory ID: {traj_id} does not exist in the dataset. Please try again!"
            else:
                return start_loc[const.LAT][0], start_loc[const.LONG][0]

    @staticmethod
    def get_end_location(dataframe: PTRAILDataFrame, traj_id: Optional[Text] = None):
        """
            Get the ending location of an object's trajectory in the data.

            Note
            ----
                If the user does not give in any traj_id, then the library,
                by default gives out the end locations of all the unique
                trajectory ids present in the data.

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The PTRAILDataFrame storing the trajectory data.
                traj_id
                    The ID of the trajectory whose end location is to be found.

            Returns
            -------
                tuple:
                    The (lat, longitude) tuple containing the end location.
                pandas.core.dataframe.DataFrame:
                    The dataframe containing start locations of all trajectory IDs.
        """
        # If traj_id is None, find the end times of all the unique trajectories present in the data.
        # Else first filter out a dataframe containing the given traj_id and then return the end
        # location of that point.
        dataframe = dataframe.copy().reset_index()
        if traj_id is None:
            ids_ = dataframe[const.TRAJECTORY_ID].value_counts(ascending=True).keys().to_list()
            # Get the ideal number of IDs by which the dataframe is to be split.
            split_factor = helpers._get_partition_size(len(ids_))
            ids_ = [ids_[i: i + split_factor] for i in range(0, len(ids_), split_factor)]

            # Here, create 2/3rds number of processes as there are in the system. Some CPUs are
            # kept free at all times in order to not block up the system.
            # (Note: The blocking of system is mostly prevalent in Windows and does not happen very often
            # in Linux. However, out of caution some CPUs are kept free regardless of the system.)
            mp_pool = multiprocessing.Pool(NUM_CPU)
            results = mp_pool.starmap(helpers.end_location_helper, zip(itertools.repeat(dataframe), ids_))
            mp_pool.close()
            mp_pool.join()

            # Concatenate all the smaller dataframes and return the answer.
            results = pd.concat(results)
            return results
        else:
            filt = (dataframe.loc[dataframe[const.TRAJECTORY_ID] == traj_id, [const.DateTime, const.LAT, const.LONG]])
            start_loc = (filt.loc[filt[const.DateTime] == filt[const.DateTime].max(),
                                  [const.LAT, const.LONG]]).reset_index()
            if len(start_loc) == 0:
                return f"Trajectory ID: {traj_id} does not exist in the dataset. Please try again!"
            else:
                return start_loc[const.LAT][0], start_loc[const.LONG][0]

    @staticmethod
    def create_distance_between_consecutive_column(dataframe: PTRAILDataFrame):
        """
            Create a column called Dist_prev_to_curr containing distance between 2 consecutive points.
            The distance calculated is the Great-Circle (Haversine) distance.

            Note
            ----
                When the trajectory ID changes in the data, then the distance calculation again starts
                from the first point of the new trajectory ID and the distance-value of the first point
                of the new Trajectory ID will be set to 0.

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The data where speed is to be calculated.

            Returns
            -------
                PTRAILDataFrame:
                    The dataframe containing the resultant Distance_prev_to_curr column.
        """
        # Case-1: The number of unique Trajectory IDs is less than 100.
        if dataframe.traj_id.nunique() < const.MIN_IDS:
            result = helpers.distance_between_consecutive_helper(dataframe)
            return PTRAILDataFrame(result, const.LAT, const.LONG,
                                   const.DateTime, const.TRAJECTORY_ID)
        # Case-2: The number of unique Trajectory IDs is significant.
        else:
            # splitting the dataframe according to trajectory ids.
            df_chunks = helpers._df_split_helper(dataframe)

            # Here, create 2/3rds number of processes as there are in the system. Some CPUs are
            # kept free at all times in order to not block up the system.
            # (Note: The blocking of system is mostly prevalent in Windows and does not happen very often
            # in Linux. However, out of caution some CPUs are kept free regardless of the system.)
            multi_pool = multiprocessing.Pool(NUM_CPU)
            result = multi_pool.map(helpers.distance_between_consecutive_helper, df_chunks)
            multi_pool.close()
            multi_pool.join()

            # merge the smaller pieces and then return the dataframe converted to PTRAILDataFrame.
            return PTRAILDataFrame(pd.concat(result), const.LAT, const.LONG,
                                   const.DateTime, const.TRAJECTORY_ID)

    @staticmethod
    def create_distance_from_start_column(dataframe: PTRAILDataFrame):
        """
            Create a column containing distance between the start location and the rest of the
            points using Haversine formula. The distance calculated is the Great-Circle distance.

            Note
            ----
                When the trajectory ID changes in the data, then the distance calculation again
                starts from the first point of the new trajectory ID and the first distance of the
                new trajectory ID will be set to 0.

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The data where speed is to be calculated.

            Returns
            -------
                PTRAILDataFrame:
                    The dataframe containing the resultant Distance_start_to_curr column.
        """
        # Case-1: The number of unique Trajectory IDs is less than 100.
        if dataframe.traj_id.nunique() < const.MIN_IDS:
            result = helpers.distance_from_start_helper(dataframe)
            return PTRAILDataFrame(result, const.LAT, const.LONG,
                                   const.DateTime, const.TRAJECTORY_ID)

        # Case-2: The number of unique Trajectory IDs is significant.
        else:
            # splitting the dataframe according to trajectory ids.
            df_chunks = helpers._df_split_helper(dataframe)

            # Here, create 2/3rds number of processes as there are in the system. Some CPUs are
            # kept free at all times in order to not block up the system.
            # (Note: The blocking of system is mostly prevalent in Windows and does not happen very often
            # in Linux. However, out of caution some CPUs are kept free regardless of the system.)
            multi_pool = multiprocessing.Pool(NUM_CPU)
            result = multi_pool.map(helpers.distance_from_start_helper, df_chunks)
            multi_pool.close()
            multi_pool.join()

            # merge the smaller pieces and then return the dataframe converted to PTRAILDataFrame.
            return PTRAILDataFrame(pd.concat(result), const.LAT, const.LONG,
                                   const.DateTime, const.TRAJECTORY_ID)

    @staticmethod
    def get_distance_travelled_by_date_and_traj_id(dataframe: PTRAILDataFrame, date, traj_id=None):
        """
            Given a date and trajectory ID, calculate the total distance
            covered in the trajectory on that particular date.

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The dataframe in which teh actual data is stored.
                date: Text
                    The Date on which the distance covered is to be calculated.
                traj_id: Text
                    The trajectory ID for which the distance covered is to be calculated.

            Returns
            -------
                float:
                    The total distance covered on that date by that trajectory ID.
        """
        # First, reset the index of the dataframe.
        # Then, filter the dataframe based on Date and Trajectory ID if given by user.
        dataframe = dataframe.reset_index()
        filt = dataframe.loc[dataframe[const.DateTime].dt.date == pd.to_datetime(date)]
        small = filt.loc[filt[const.TRAJECTORY_ID] == traj_id] if traj_id is not None else filt

        # First, lets fetch the latitude and longitude columns from the dataset and store it
        # in a numpy array.
        traj_ids = np.array(small.reset_index()[const.TRAJECTORY_ID])
        latitudes = np.array(small[const.LAT])
        longitudes = np.array(small[const.LONG])
        distances = np.zeros(len(traj_ids))

        # Now, lets calculate the Great-Circle (Haversine) distance between the 2 points and store
        # each of the values in the distance numpy array.
        for i in range(len(latitudes) - 1):
            distances[i + 1] = calc.haversine_distance(latitudes[i], longitudes[i], latitudes[i + 1], longitudes[i + 1])

        return np.sum(distances)  # Sum all the distances and return the total path length.

    @staticmethod
    def create_point_within_range_column(dataframe: PTRAILDataFrame, coordinates: tuple,
                                         dist_range: float):
        """
            Check how many points are within the range of the given coordinate by first making a column
            containing the distance between the given coordinate and rest of the points in dataframe by calling
            create_distance_from_point() and then comparing each point using the condition if it's within the
            range and appending the values in a column and attaching it to the dataframe.

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The dataframe on which the point within range calculation is to be done.
                coordinates: tuple
                    The coordinates from which the distance is to be calculated.
                dist_range: float
                    The range within which the resultant distance from the coordinates should lie.

            Returns
            -------
                PTRAILDataFrame:
                    The dataframe containing the resultant Within_x_m_from_(x,y) column.
        """
        # splitting the dataframe according to trajectory ids
        df_chunks = helpers._df_split_helper(dataframe)

        # Here, create 2/3rds number of processes as there are in the system. Some CPUs are
        # kept free at all times in order to not block up the system.
        # (Note: The blocking of system is mostly prevalent in Windows and does not happen very often
        # in Linux. However, out of caution some CPUs are kept free regardless of the system.)
        pool = multiprocessing.Pool(NUM_CPU)
        args = zip(df_chunks, itertools.repeat(coordinates), itertools.repeat(dist_range))
        result = pool.starmap(helpers.point_within_range_helper, args)
        pool.close()
        pool.join()

        # Now lets join all the smaller partitions and return the resultant dataframe
        result = pd.concat(result)
        return PTRAILDataFrame(result.reset_index(), const.LAT, const.LONG, const.DateTime, const.TRAJECTORY_ID)

    @staticmethod
    def create_distance_from_given_point_column(dataframe: PTRAILDataFrame, coordinates: tuple):
        """
            Given a point, this function calculates the distance between that point and all the
            points present in the dataframe and adds that column into the dataframe.

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The dataframe on which calculation is to be done.
                coordinates: tuple
                    The coordinates from which the distance is to be calculated.

            Returns
            -------
                PTRAILDataFrame:
                    The dataframe containing the resultant Distance_from_(x, y) column.
        """
        # dataframe = dataframe.reset_index()
        # splitting the dataframe according to trajectory ids
        df_chunks = helpers._df_split_helper(dataframe)

        # Here, create 2/3rds number of processes as there are in the system. Some CPUs are
        # kept free at all times in order to not block up the system.
        # (Note: The blocking of system is mostly prevalent in Windows and does not happen very often
        # in Linux. However, out of caution some CPUs are kept free regardless of the system.)
        pool = multiprocessing.Pool(NUM_CPU)
        answer = pool.starmap(helpers.distance_from_given_point_helper, zip(df_chunks, itertools.repeat(coordinates)))
        pool.close()
        pool.join()

        # Now lets join all the smaller partitions and then add the Distance to the
        # specific point column.
        answer = pd.concat(answer)

        # return the answer dataframe converted to PTRAILDataFrame.
        return PTRAILDataFrame(answer.reset_index(), const.LAT, const.LONG, const.DateTime, const.TRAJECTORY_ID)

    @staticmethod
    def create_speed_from_prev_column(dataframe: PTRAILDataFrame):
        """
            Create a column containing speed of the object from the previous point
            to the current point.

            Note
            ----
                When the trajectory ID changes in the data, then the speed calculation again
                starts from the first point of the new trajectory ID and the speed of the
                first point of the new trajectory ID will be set to 0.

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The dataframe on which the calculation of speed is to be done.

            Returns
            -------
                PTRAILDataFrame:
                    The dataframe containing the resultant Speed_prev_to_curr column.
        """
        # Here, we are using try and catch blocks to check whether the DataFrame has the
        # Distance_prev_to_curr column.
        try:
            # If the Distance_prev_to_curr column is already present in the dataframe,
            # then extract it, calculate the time differences between the consecutive
            # rows in the dataframe and then calculate distances/time_deltas in order to
            # calculate the speed.
            # WARNING!!!! Use dt.total_seconds() as dt.seconds gives false values and as it
            #             does not account for time difference when it is negative.
            distances = dataframe.reset_index()['Distance_prev_to_curr']
            time_deltas = dataframe.reset_index()[const.DateTime].diff().dt.total_seconds()

            # Assign the new column and return the NumPandasTrajDF.
            dataframe['Speed_prev_to_curr'] = (distances / time_deltas).to_numpy()
            return dataframe

        except KeyError:
            # If the Distance_prev_to_curr column is not present in the Dataframe and a KeyError
            # is thrown, then catch it and the overridden behaviour is as follows:
            #   1. Calculate the distance by calling the create_distance_between_consecutive_column() function.
            #   2. Calculate the time deltas.
            #   3. Divide the 2 values to calculate the speed.
            # WARNING!!!! Use dt.total_seconds() as dt.seconds gives false values and as it
            #             does not account for time difference when it is negative.
            dataframe = KinematicFeatures.create_distance_between_consecutive_column(dataframe)
            distances = dataframe.reset_index()['Distance_prev_to_curr']
            time_deltas = dataframe.reset_index()[const.DateTime].diff().dt.total_seconds()

            # Assign the column and return the NumPandasTrajDF.
            dataframe['Speed_prev_to_curr'] = (distances / time_deltas).to_numpy(dtype=np.float64)
            return dataframe

    @staticmethod
    def create_acceleration_from_prev_column(dataframe: PTRAILDataFrame):
        """
            Create a column containing acceleration of the object from the previous to the current
            point.

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The dataframe on which the calculation of acceleration is to be done.

            Returns
            -------
                PTRAILDataFrame:
                    The dataframe containing the resultant Acceleration_prev_to_curr column.
        """
        # Try catch is used to check if speed column is present or not
        try:
            # When Speed column is present extract the data from there and then take calculate the time delta
            # And use that to calculate acceleration by dividing speed by time delta and then add the column to
            # the dataframe
            # WARNING!!!! Use dt.total_seconds() as dt.seconds gives false values and as it
            #             does not account for time difference when it is negative.
            speed_deltas = dataframe.reset_index()['Speed_prev_to_curr'].diff()
            time_deltas = dataframe.reset_index()[const.DateTime].diff().dt.total_seconds()

            dataframe['Acceleration_prev_to_curr'] = (speed_deltas / time_deltas).to_numpy()
            return dataframe

        except KeyError:
            # When Speed column is not present then first call create_speed_from_prev_column() function to make
            # the speed column and then follow the steps mentioned above
            # WARNING!!!! Use dt.total_seconds() as dt.seconds gives false values and as it
            #             does not account for time difference when it is negative.
            dataframe = KinematicFeatures.create_speed_from_prev_column(dataframe)
            speed_deltas = dataframe.reset_index()['Speed_prev_to_curr'].diff()
            time_deltas = dataframe.reset_index()[const.DateTime].diff().dt.total_seconds()

            dataframe['Acceleration_prev_to_curr'] = (speed_deltas / time_deltas).to_numpy()
            return dataframe

    @staticmethod
    def create_jerk_from_prev_column(dataframe: PTRAILDataFrame):
        """
            Create a column containing jerk of the object from previous to the current
            point.

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The dataframe on which the calculation of jerk is to be done.

            Returns
            -------
                PTRAILDataFrame:
                    The dataframe containing the resultant jerk_prev_to_curr column.
        """
        # Try catch is used to check if acceleration column is present or not
        try:
            # When acceleration column is present extract the data from there and then take calculate the time delta
            # And use that to calculate acceleration by dividing speed_delta by time delta and then add the column to
            # the dataframe
            # WARNING!!!! Use dt.total_seconds() as dt.seconds gives false values and as it
            #             does not account for time difference when it is negative.
            acceleration_deltas = dataframe.reset_index()['Acceleration_prev_to_curr'].diff()
            time_deltas = dataframe.reset_index()[const.DateTime].diff().dt.total_seconds()

            dataframe['jerk_prev_to_curr'] = (acceleration_deltas / time_deltas).to_numpy()
            return dataframe

        except KeyError:
            # When Speed column is not present then first call create_speed_from_prev_column() function to make
            # the speed column and then follow the steps mentioned above
            # WARNING!!!! Use dt.total_seconds() as dt.seconds gives false values and as it
            #             does not account for time difference when it is negative.
            dataframe = KinematicFeatures.create_acceleration_from_prev_column(dataframe)
            acceleration_deltas = dataframe.reset_index()['Acceleration_prev_to_curr'].diff()
            time_deltas = dataframe.reset_index()[const.DateTime].diff().dt.total_seconds()

            dataframe['jerk_prev_to_curr'] = (acceleration_deltas / time_deltas).to_numpy()
            return dataframe

    @staticmethod
    def create_bearing_column(dataframe: PTRAILDataFrame):
        """
            Create a column containing bearing between 2 consecutive points. Bearing is also
            referred as "Forward Azimuth" sometimes. Bearing/Forward Azimuth is defined as
            follows:
                Bearing is the horizontal angle between the direction of an object and another
                object, or between the object and the True North.

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The dataframe on which the bearing is to be calculated.

            Returns
            -------
                PTRAILDataFrame:
                        The dataframe containing the resultant Bearing_from_prev column.
        """
        # if cpu_count <= 1:
        #     cpu_count = 1
        # elif cpu_count >= NUM_CPU:
        #     cpu_count = NUM_CPU - 1
        # else:
        #     cpu_count = cpu_count

        # Case-1: The number of unique Trajectory IDs is less than x.
        if dataframe.traj_id.nunique() < const.MIN_IDS:
            result = helpers.bearing_helper(dataframe)
            return PTRAILDataFrame(result, const.LAT, const.LONG,
                                   const.DateTime, const.TRAJECTORY_ID)

        # Case-2: The number unique Trajectory IDs is significant.
        else:
            # splitting the dataframe according to trajectory ids.
            df_chunks = helpers._df_split_helper(dataframe)

            # Here, create 2/3rds number of processes as there are in the system. Some CPUs are
            # kept free at all times in order to not block up the system.
            # (Note: The blocking of system is mostly prevalent in Windows and does not happen very often
            # in Linux. However, out of caution some CPUs are kept free regardless of the system.)
            multi_pool = multiprocessing.Pool(NUM_CPU)
            result = multi_pool.map(helpers.bearing_helper, df_chunks)
            multi_pool.close()
            multi_pool.join()

            # merge the smaller pieces and then return the dataframe converted to PTRAILDataFrame.
            return PTRAILDataFrame(pd.concat(result), const.LAT, const.LONG,
                                   const.DateTime, const.TRAJECTORY_ID)

    @staticmethod
    def create_bearing_rate_column(dataframe: PTRAILDataFrame):
        """
            Calculates the bearing rate of the consecutive points. And adding that column into
            the dataframe

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The dataframe on which the bearing rate is to be calculated

            Returns
            -------
                PTRAILDataFrame:
                    The dataframe containing the resultant Bearing_rate_from_prev column.
        """
        # Try catch to check for Bearing column
        try:
            # If Bearing from previous column is present, extract that and then calculate time_deltas
            # Using these calculate Bearing_rate_from_prev by dividing bearing_deltas with time_deltas
            # And then adding the column to the dataframe
            # WARNING!!!! Use dt.total_seconds() as dt.seconds gives false values and as it
            #             does not account for time difference when it is negative.
            bearing_deltas = dataframe.reset_index()['Bearing_between_consecutive'].diff()
            time_deltas = dataframe.reset_index()[const.DateTime].diff().dt.total_seconds()

            dataframe['Bearing_rate_from_prev'] = (bearing_deltas / time_deltas).to_numpy()
            return dataframe
        except KeyError:
            # Similar to the step above but just makes the Bearing column first
            # WARNING!!!! Use dt.total_seconds() as dt.seconds gives false values and as it
            #             does not account for time difference when it is negative.
            dataframe = KinematicFeatures.create_bearing_column(dataframe)
            bearing_deltas = dataframe.reset_index()['Bearing_between_consecutive'].diff()
            time_deltas = dataframe.reset_index()[const.DateTime].diff().dt.total_seconds()

            dataframe['Bearing_rate_from_prev'] = (bearing_deltas / time_deltas).to_numpy()
            return dataframe

    @staticmethod
    def create_rate_of_bearing_rate_column(dataframe: PTRAILDataFrame):
        """
            Calculates the rate of bearing rate of the consecutive points.
            And then adding that column into the dataframe.

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The dataframe on which the rate of bearing rate is to be calculated

            Returns
            -------
                PTRAILDataFrame:
                    The dataframe containing the resultant Rate_of_bearing_rate_from_prev column
        """
        # Try catch to check for Bearing Rate column
        try:
            # If Bearing from previous column is present, extract that and then calculate time_deltas
            # Using these calculate Bearing_rate_from_prev by dividing bearing_deltas with time_deltas
            # And then adding the column to the dataframe
            # WARNING!!!! Use dt.total_seconds() as dt.seconds gives false values and as it
            #             does not account for time difference when it is negative.
            bearing_rate_deltas = dataframe.reset_index()['Bearing_rate_from_prev'].diff()
            time_deltas = dataframe.reset_index()[const.DateTime].diff().dt.total_seconds()

            dataframe['Rate_of_bearing_rate_from_prev'] = (bearing_rate_deltas / time_deltas).to_numpy()
            return dataframe
        except KeyError:
            # Similar to the step above but just makes the Bearing column first
            # WARNING!!!! Use dt.total_seconds() as dt.seconds gives false values and as it
            #             does not account for time difference when it is negative.
            dataframe = KinematicFeatures.create_bearing_rate_column(dataframe)
            bearing_rate_deltas = dataframe.reset_index()['Bearing_between_consecutive'].diff()
            time_deltas = dataframe.reset_index()[const.DateTime].diff().dt.total_seconds()

            dataframe['Rate_of_bearing_rate_from_prev'] = (bearing_rate_deltas / time_deltas).to_numpy()
            return dataframe

    @staticmethod
    def get_distance_travelled_by_traj_id(dataframe: PTRAILDataFrame, traj_id: Text):
        """
            Given a trajectory ID, calculate the total distance covered by the trajectory.
            NOTE: The distance calculated is in metres.

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The dataframe containing the entire dataset.
                traj_id: Text
                    The trajectory ID for which the distance covered is to be calculated.

            Returns
            -------
                float:
                    The distance covered by the trajectory

            Raises
            ------
                MissingTrajIDException:
                    The Trajectory ID given by the user is not present in the dataset.
        """
        # First, filter the dataframe and create a smaller dataframe containing only the
        # trajectory points of the specified trajectory ID.
        dataframe = dataframe.reset_index()
        filtered_df = dataframe.loc[dataframe[const.TRAJECTORY_ID] == traj_id]

        # Now, check if the smaller dataframe actually has anything because if the user-given
        # trajectory ID is not present in the dataset, then an empty dataframe is returned and
        # if so raise an exception.
        if len(filtered_df) > 0:
            # First, calculate the distance by calling the distance_between_consecutive_column() function
            # and convert it into a numpy array and then sum the array using nansum() to make sure that
            # NaN values are considered as zeros.
            distances = KinematicFeatures.create_distance_between_consecutive_column(filtered_df)[
                'Distance_prev_to_curr'].to_numpy()
            return np.nansum(distances)
        else:
            raise MissingTrajIDException(f"The Trajectory ID '{traj_id}' is not present in the data."
                                         f"Please check the Trajectory ID and try again.")

    @staticmethod
    def get_number_of_locations(dataframe: PTRAILDataFrame, traj_id: Text = None):
        """
            Get the number of unique coordinates in the dataframe specific to a trajectory ID.

            Note
            ----
                If no Trajectory ID is specified, then the number of unique locations in the
                visited by each trajectory in the dataset is calculated.

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The dataframe of which the number of locations are to be computed
                traj_id: Text
                    The trajectory id for which the number of unique locations are to be found

            Returns
            -------
                int:
                    The number of unique locations in the dataframe/trajectory id.
                pandas.core.dataframe.DataFrame:
                    The dataframe containing start locations of all trajectory IDs.
        """
        dataframe = dataframe.reset_index()
        if traj_id is None:
            ids_ = dataframe[const.TRAJECTORY_ID].value_counts(ascending=True).keys().to_list()
            # Get the ideal number of IDs by which the dataframe is to be split.
            split_factor = helpers._get_partition_size(len(ids_))
            ids_ = [ids_[i: i + split_factor] for i in range(0, len(ids_), split_factor)]

            # Here, create 2/3rds number of processes as there are in the system. Some CPUs are
            # kept free at all times in order to not block up the system.
            # (Note: The blocking of system is mostly prevalent in Windows and does not happen very often
            # in Linux. However, out of caution some CPUs are kept free regardless of the system.)
            mp_pool = multiprocessing.Pool(NUM_CPU)
            results = mp_pool.starmap(helpers.number_of_location_helper, zip(itertools.repeat(dataframe), ids_))
            mp_pool.close()
            mp_pool.join()

            # Concatenate all the smaller dataframes and return the answer.
            results = pd.concat(results)
            return results

        else:
            filtered_df = dataframe.loc[dataframe[const.TRAJECTORY_ID] == traj_id]
            return filtered_df.groupby([const.LAT, const.LONG]).ngroups
