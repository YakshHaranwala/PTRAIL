"""
    The spatial_features  module contains several functions of the library
    that calculates many features based on the DateTime provided in
    the data. This module mostly extracts and modifies data collected from
    some existing dataframe and appends these information to them. It is to
    be also noted that a lot of these features are inspired from the PyMove
    library and we are crediting the PyMove creators with them.

    @authors Yaksh J Haranwala, Salman Haidri
    @date 2nd June, 2021
    @version 1.0
    @credits PyMove creators
"""
import itertools
import multiprocessing
from typing import Optional, Text

import numpy as np
import pandas as pd

from core.TrajectoryDF import NumPandasTraj
from features.helper_functions import Helpers as helpers
from utilities import constants as const
from utilities.DistanceCalculator import DistanceFormulaLog as calc
from utilities.exceptions import *


class SpatialFeatures:
    @staticmethod
    def get_bounding_box(dataframe: NumPandasTraj):
        """
            Return the bounding box of the Trajectory data. Essentially, the bounding box is of
            the following format:
                (min Latitude, min Longitude, max Latitude, max Longitude).

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The dataframe containing the trajectory data.

            Returns
            -------
                tuple
                    The bounding box of the trajectory
        """
        return (
            dataframe[const.LAT].min(),
            dataframe[const.LONG].min(),
            dataframe[const.LAT].max(),
            dataframe[const.LONG].max(),
        )

    @staticmethod
    def get_start_location(dataframe: NumPandasTraj, traj_id=None):
        """
            Get the starting location of an object's trajectory in the data.

            NOTE: If the user does not give in any traj_id, then the library,
                  by default gives out the start locations of all the unique trajectory ids
                  present in the data.

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The DaskTrajectoryDF storing the trajectory data.
                traj_id
                    The ID of the object whose start location is to be found.

            Returns
            -------
                tuple
                    The (lat, longitude) tuple containing the start location.
                pandas.core.dataframe.DataFrame
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

            # Now, create a multiprocessing pool and then run processes in parallel
            # which calculate the start locations for a smaller set of IDs only.
            mp_pool = multiprocessing.Pool(len(ids_))
            results = mp_pool.starmap(helpers._start_location_helper, zip(itertools.repeat(dataframe), ids_))

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
    def get_end_location(dataframe: NumPandasTraj, traj_id: Optional[Text] = None):
        """
            Get the ending location of an object's trajectory in the data.
            Note: If the user does not provide a trajectory id, then the last
            Parameters
            ----------
                dataframe: DaskTrajectoryDF
                    The DaskTrajectoryDF storing the trajectory data.
                traj_id
                    The ID of the trajectory whose end location is to be found.
            Returns
            -------
                tuple
                    The (lat, longitude) tuple containing the end location.
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

            # Now, create a multiprocessing pool and then run processes in parallel
            # which calculate the end locations for a smaller set of IDs only.
            mp_pool = multiprocessing.Pool(len(ids_))
            results = mp_pool.starmap(helpers._end_location_helper, zip(itertools.repeat(dataframe), ids_))

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
    def create_distance_between_consecutive_column(dataframe: NumPandasTraj):
        """
            Create a column called Dist_prev_to_curr containing distance between 2 consecutive points.
            The distance calculated is the Great-Circle distance.
            NOTE: When the trajectory ID changes in the data, then the distance calculation again starts
                  from the first point of the new trajectory ID and the first point of the new trajectory
                  ID will be set to 0.

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The data where speed is to be calculated.

            Returns
            -------
                core.TrajectoryDF.NumPandasTraj
                    The dataframe containing the resultant column.
        """
        # splitting the dataframe according to trajectory ids
        df_chunks = helpers._df_split_helper(dataframe)
        # Now, lets create a pool of processes which contains processes equal to the number
        # of smaller chunks and then run them in parallel so that we can calculate
        # the distance for each smaller chunk and then merge all of them together.
        multi_pool = multiprocessing.Pool(len(df_chunks))
        result = multi_pool.map(helpers._consecutive_distance_alt, df_chunks)

        # Now lets, merge the smaller pieces and then return the dataframe
        result = pd.concat(result)
        return result

    @staticmethod
    def create_distance_from_start_column(dataframe: NumPandasTraj):
        """
            Create a column containing distance between the start location and the rest of the
            points using Haversine formula. The distance calculated is the Great-Circle distance.
            NOTE: When the trajectory ID changes in the data, then the distance calculation again
                  starts from the first point of the new trajectory ID and the first point of the
                  new trajectory ID will be set to 0.

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The data where speed is to be calculated.

            Returns
            -------
                core.TrajectoryDF.NumPandasTraj
                    The dataframe containing the resultant column.
        """
        #dataframe = dataframe.reset_index()
        # splitting the dataframe according to trajectory ids
        df_chunks = helpers._df_split_helper(dataframe)

        # Now, lets create a multiprocessing pool of processes and then create as many
        # number of processes as there are number of partitions and run each process in parallel.
        pool = multiprocessing.Pool(len(df_chunks))
        answer = pool.map(helpers._start_distance_helper, df_chunks)

        answer = pd.concat(answer)
        return answer
        #return NumPandasTraj(answer, const.LAT, const.LONG, const.DateTime, const.TRAJECTORY_ID)

    @staticmethod
    def get_distance_travelled_by_date_and_traj_id(dataframe: NumPandasTraj, date, traj_id=None):
        """
            Given a date and trajectory ID, this function calculates the total distance
            covered in the trajectory on that particular date and returns it.

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The dataframe in which teh actual data is stored.
                date: Text
                    The Date on which the distance covered is to be calculated.
                traj_id: Text
                    The trajectory ID for which the distance covered is to be calculated.

            Returns
            -------
                float
                    The total distance covered on that date by that trajectory ID.
        """
        # First, reset the index of the dataframe.
        # Then, filter the dataframe based on Date and Trajectory ID if given by user.
        data = dataframe.reset_index()
        filt = data.loc[data[const.DateTime].dt.date == pd.to_datetime(date)]
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
    def create_point_within_range_column(dataframe: NumPandasTraj, coordinates: tuple,
                                         dist_range: float):
        """
            Checks how many points are within the range of the given coordinate. By first making a column
            containing the distance between the given coordinate and rest of the points in dataframe by calling
            create_distance_from_point(). And then comparing each point using the condition if it's within the
            range and appending the values in a column and attaching it to the dataframe.

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The dataframe on which the point within range calculation is to be done.
                coordinates: tuple
                    The coordinates from which the distance is to be calculated.
                dist_range: float
                    The range within which the resultant distance from the coordinates should lie.

            Returns
            -------
                core.TrajectoryDF.NumPandasTraj
                    The dataframe containing the resultant column.

        """
        dataframe = dataframe.reset_index()
        # splitting the dataframe according to trajectory ids
        df_chunks = helpers._df_split_helper(dataframe)

        # Now, lets create a multiprocessing pool of processes and then create as many
        # number of processes as there are number of partitions and run each process in parallel.
        pool = multiprocessing.Pool(len(df_chunks))
        args = zip(df_chunks, itertools.repeat(coordinates), itertools.repeat(dist_range))
        result = pool.starmap(helpers._point_within_range_helper, args)

        # Now lets join all the smaller partitions and return the resultant dataframe
        result = pd.concat(result)
        return NumPandasTraj(result, const.LAT, const.LONG, const.DateTime, const.TRAJECTORY_ID)

    @staticmethod
    def create_distance_from_given_point_column(dataframe: NumPandasTraj, coordinates: tuple):
        """
            Given a point, this function calculates the distance between that point and all the
            points present in the dataframe and adds that column into the dataframe.

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The dataframe on which calculation is to be done.
                coordinates: tuple
                    The coordinates from which the distance is to be calculated.

            Returns
            -------
                core.TrajectoryDF.NumPandasTraj
                    The dataframe containing the resultant column.
        """
        dataframe = dataframe.reset_index()
        # splitting the dataframe according to trajectory ids
        df_chunks = helpers._df_split_helper(dataframe)

        # Now, lets create a multiprocessing pool of processes and then create as many
        # number of processes as there are number of partitions and run each process in parallel.
        pool = multiprocessing.Pool(len(df_chunks))
        answer = pool.starmap(helpers._given_point_distance_helper, zip(df_chunks, itertools.repeat(coordinates)))

        # Now lets join all the smaller partitions and then add the Distance to the
        # specific point column.
        answer = pd.concat(answer)

        # return the answer dataframe converted to NumPandasTraj.
        return NumPandasTraj(answer, const.LAT, const.LONG, const.DateTime, const.TRAJECTORY_ID)

    @staticmethod
    def create_speed_from_prev_column(dataframe: NumPandasTraj):
        """
            Create a column containing speed of the object from the start to the current
            point.

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The dataframe on which the calculation of speed is to be done.

            Returns
            -------
                core.TrajectoryDF.NumPandasTraj
                    The dataframe containing the resultant column.
        """
        # Here, we are using try and catch blocks to check whether the DataFrame has the
        # Distance_prev_to_curr column.
        try:
            # If the Distance_prev_to_curr column is already present in the dataframe,
            # then extract it, calculate the time differences between the consecutive
            # rows in the dataframe and then calculate distances/time_deltas in order to
            # calculate the speed.
            distances = dataframe.reset_index()['Distance_prev_to_curr']
            time_deltas = dataframe.reset_index()[const.DateTime].diff().dt.seconds

            # Assign the new column and return the NumPandasTrajDF.
            dataframe['Speed_prev_to_curr'] = (distances / time_deltas).to_numpy()
            return dataframe

        except KeyError:
            # If the Distance_prev_to_curr column is not present in the Dataframe and a KeyError
            # is thrown, then catch it and the overridden behaviour is as follows:
            #   1. Calculate the distance by calling the create_distance_between_consecutive_column() function.
            #   2. Calculate the time deltas.
            #   3. Divide the 2 values to calculate the speed.
            dataframe = SpatialFeatures.create_distance_between_consecutive_column(dataframe)
            distances = dataframe.reset_index()['Distance_prev_to_curr']
            time_deltas = dataframe.reset_index()[const.DateTime].diff().dt.seconds

            # Assign the column and return the NumPandasTrajDF.
            dataframe['Speed_prev_to_curr'] = (distances / time_deltas).to_numpy()
            return dataframe

    @staticmethod
    def create_acceleration_from_prev_column(dataframe: NumPandasTraj):
        """
            Create a column containing acceleration of the object from the start to the current
            point.

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The dataframe on which the calculation of acceleration is to be done.

            Returns
            -------
                core.TrajectoryDF.NumPandasTraj
                    The dataframe containing the resultant column.
        """
        # Try catch is used to check if speed column is present or not
        try:
            # When Speed column is present extract the data from there and then take calculate the time delta
            # And use that to calculate acceleration by dividing speed by time delta and then add the column to
            # the dataframe
            speed_deltas = dataframe.reset_index()['Speed_prev_to_curr'].diff()
            time_deltas = dataframe.reset_index()[const.DateTime].diff().dt.seconds

            dataframe['Acceleration_prev_to_curr'] = (speed_deltas / time_deltas).to_numpy()
            return dataframe

        except KeyError:
            # When Speed column is not present then first call create_speed_from_prev_column() function to make
            # the speed column and then follow the steps mentioned above
            dataframe = SpatialFeatures.create_speed_from_prev_column(dataframe)
            speed_deltas = dataframe.reset_index()['Speed_prev_to_curr'].diff()
            time_deltas = dataframe.reset_index()[const.DateTime].diff().dt.seconds

            dataframe['Acceleration_prev_to_curr'] = (speed_deltas / time_deltas).to_numpy()
            return dataframe

    @staticmethod
    def create_jerk_from_prev_column(dataframe: NumPandasTraj):
        """
            Create a column containing jerk of the object from the start to the current
            point.

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The dataframe on which the calculation of jerk is to be done.

            Returns
            -------
                core.TrajectoryDF.NumPandasTraj
                    The dataframe containing the resultant column.
        """
        # Try catch is used to check if acceleration column is present or not
        try:
            # When acceleration column is present extract the data from there and then take calculate the time delta
            # And use that to calculate acceleration by dividing speed_delta by time delta and then add the column to
            # the dataframe
            acceleration_deltas = dataframe.reset_index()['Acceleration_prev_to_curr'].diff()
            time_deltas = dataframe.reset_index()[const.DateTime].diff().dt.seconds

            dataframe['jerk_prev_to_curr'] = (acceleration_deltas / time_deltas).to_numpy()
            return dataframe

        except KeyError:
            # When Speed column is not present then first call create_speed_from_prev_column() function to make
            # the speed column and then follow the steps mentioned above
            dataframe = SpatialFeatures.create_acceleration_from_prev_column(dataframe)
            acceleration_deltas = dataframe.reset_index()['Acceleration_prev_to_curr'].diff()
            time_deltas = dataframe.reset_index()[const.DateTime].diff().dt.seconds

            dataframe['jerk_prev_to_curr'] = (acceleration_deltas / time_deltas).to_numpy()
            return dataframe

    @staticmethod
    def create_bearing_column(dataframe: NumPandasTraj):
        """
            Create a column containing bearing between 2 consecutive points. Bearing is also
            referred as "Forward Azimuth" sometimes. Bearing/Forward Azimuth is defined as
            follows:
                Bearing is the horizontal angle between the direction of an object and another
                object, or between the object and the True North.

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The dataframe on which the bearing is to be calculated.

            Returns
            -------
                NumPandasTraj
                    The dataframe containing the resultant column.
        """
        dataframe = dataframe.reset_index()
        # splitting the dataframe according to trajectory ids
        df_chunks = helpers._df_split_helper(dataframe)

        # Now lets create a Pool of processes which has number of processes equal
        # to the number of smaller pieces of data and then lets run them all in
        # parallel.
        multi_pool = multiprocessing.Pool(len(df_chunks))
        result = multi_pool.map(helpers._bearing_helper, df_chunks)

        # Now, lets concat the results and then return the dataframe.
        result = pd.concat(result)
        return NumPandasTraj(result, const.LAT, const.LONG, const.DateTime, const.TRAJECTORY_ID)

    @staticmethod
    def create_bearing_rate_column(dataframe: NumPandasTraj):
        """
            Calculates the bearing rate of the consecutive points. And adding that column into
            the dataframe

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The dataframe on which the bearing rate is to be calculated

            Returns
            -------
                NumPandasTraj
                    The dataframe containing the Bearing rate column
        """
        # Try catch to check for Bearing column
        try:
            # If Bearing from previous column is present, extract that and then calculate time_deltas
            # Using these calculate Bearing_rate_from_prev by dividing bearing_deltas with time_deltas
            # And then adding the column to the dataframe
            bearing_deltas = dataframe.reset_index()['Bearing_between_consecutive'].diff()
            time_deltas = dataframe.reset_index()[const.DateTime].diff().dt.seconds

            dataframe['Bearing_rate_from_prev'] = (bearing_deltas / time_deltas).to_numpy()
            return dataframe
        except KeyError:
            # Similar to the step above but just makes the Bearing column first
            dataframe = SpatialFeatures.create_bearing_column(dataframe)
            bearing_deltas = dataframe.reset_index()['Bearing_between_consecutive'].diff()
            time_deltas = dataframe.reset_index()[const.DateTime].diff().dt.seconds

            dataframe['Bearing_rate_from_prev'] = (bearing_deltas / time_deltas).to_numpy()
            return dataframe

    @staticmethod
    def create_rate_of_bearing_rate_column(dataframe: NumPandasTraj):
        """
            Calculates the rate of bearing rate of the consecutive points.
            Add adding that column into the dataframe

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The dataframe on which the rate of bearing rate is to be calculated

            Returns
            -------
                NumPandasTraj
                    The dataframe containing the rate of Bearing rate column
        """
        # Try catch to check for Bearing Rate column
        try:
            # If Bearing from previous column is present, extract that and then calculate time_deltas
            # Using these calculate Bearing_rate_from_prev by dividing bearing_deltas with time_deltas
            # And then adding the column to the dataframe
            bearing_rate_deltas = dataframe.reset_index()['Bearing_rate_from_prev'].diff()
            time_deltas = dataframe.reset_index()[const.DateTime].diff().dt.seconds

            dataframe['Rate_of_bearing_rate_from_prev'] = (bearing_rate_deltas / time_deltas).to_numpy()
            return dataframe
        except KeyError:
            # Similar to the step above but just makes the Bearing column first
            dataframe = SpatialFeatures.create_bearing_column(dataframe)
            bearing_rate_deltas = dataframe.reset_index()['Bearing_between_consecutive'].diff()
            time_deltas = dataframe.reset_index()[const.DateTime].diff().dt.seconds

            dataframe['Rate_of_bearing_rate_from_prev'] = (bearing_rate_deltas / time_deltas).to_numpy()
            return dataframe

    @staticmethod
    def get_distance_travelled_by_traj_id(dataframe: NumPandasTraj, traj_id: Text):
        """
            Given a trajectory ID, calculate the total distance covered by the trajectory.
            NOTE: The distance calculated is in metres.

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The dataframe containing the entire dataset.
                traj_id: Text
                    The trajectory ID for which the distance covered is to be calculated.

            Returns
            -------
                float
                    The distance covered by the trajectory

            Raises
            ------
                MissingTrajIDException
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
            distances = SpatialFeatures.create_distance_between_consecutive_column(filtered_df)[
                'Distance_prev_to_curr'].to_numpy()
            return np.nansum(distances)
        else:
            raise MissingTrajIDException(f"The Trajectory ID '{traj_id}' is not present in the data."
                                         f"Please check the Trajectory ID and try again.")

    @staticmethod
    def get_number_of_locations(dataframe: NumPandasTraj, traj_id: Text = None):
        """
            Get the number of unique coordinates in the dataframe specific to a trajectory ID.
            NOTE: If no Trajectory ID is specified, then the number of unique locations in the
                  entire dataset is calculated.

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The dataframe of which the number of locations are to be computed
                traj_id: Text
                    The trajectory id for which the number of unique locations are to be found

            Returns
            -------
                integer
                    The number of unique locations in the dataframe/trajectory id.
        """
        dataframe = dataframe.reset_index()
        if traj_id is None:
            ids_ = dataframe[const.TRAJECTORY_ID].value_counts(ascending=True).keys().to_list()
            # Get the ideal number of IDs by which the dataframe is to be split.
            split_factor = helpers._get_partition_size(len(ids_))
            ids_ = [ids_[i: i + split_factor] for i in range(0, len(ids_), split_factor)]

            # Now, create a multiprocessing pool and then run processes in parallel
            # which calculate the end times for a smaller set of IDs only.
            mp_pool = multiprocessing.Pool(len(ids_))
            results = mp_pool.starmap(helpers._number_of_location_helper, zip(itertools.repeat(dataframe), ids_))

            # Concatenate all the smaller dataframes and return the answer.
            results = pd.concat(results)
            return results

        else:
            filtered_df = dataframe.loc[dataframe[const.TRAJECTORY_ID] == traj_id]
            return filtered_df.groupby([const.LAT, const.LONG]).ngroups

    @staticmethod
    def get_radius_of_gyration(dataframe: NumPandasTraj):
        pass
