"""
    This module contains all the helper functions for the parallel calculations in
    the spatial and temporal features classes.

    Warning
    -------
        These functions should not be used directly as they would result in a
        slower calculation and execution times. In some cases, these functions
        might even yield wrong results if used directly. They are meant to be used
        only as helpers. For calculation of features, use the ones in the
        features package.

    | Authors: Yaksh J Haranwala, Salman Haidri
"""
import os
from math import ceil

import numpy as np
import pandas as pd

from ptrail.utilities import constants as const
from ptrail.utilities.DistanceCalculator import FormulaLog as calc

pd.options.mode.chained_assignment = None


class Helpers:
    # ------------------------------------ Temporal Helpers --------------------------------------#
    @staticmethod
    def traj_duration_helper(dataframe, ids_):
        """
            Calculate the duration of the trajectory i.e. subtract the max time of
            the trajectory by the min time of the trajectory.

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The dataframe containing all the original data.
                ids_: list
                    A list containing all the Trajectory IDs present in the dataset.

            Returns
            -------
                pandas.core.dataframe.DataFrame
                    The resultant dataframe containing all the trajectory durations.
        """
        durations = []  # A list for storing results.

        # Iterate over each ID and calculate the time duration of each unique ID.
        for i in range(len(ids_)):
            # Filter out only the points of the ID in question.
            small = dataframe.loc[dataframe[const.TRAJECTORY_ID] == ids_[i], [const.DateTime]]

            # Calculate the duration of the trajectory in question nd append
            # a row containing [traj_duration, traj_id] to the results list.
            durations.append([(small.max() - small.min())[0], ids_[i]])

        # Convert the list containing results to a pandas dataframe, reset the index
        # and then rename the columns.
        result = pd.DataFrame(durations).reset_index(drop=True).rename(columns={0: "Traj_Duration",
                                                                                1: const.TRAJECTORY_ID})
        # Set the index to traj_id and return it.
        return result.set_index(const.TRAJECTORY_ID)

    @staticmethod
    def start_time_helper(dataframe, ids_):
        """
            This function is the helper function of the get_start_time(). The get_start_time() function
            delegates the task of calculating the start_time of the trajectories in the dataframe because the
            original functions runs multiple instances of this function in parallel. This function finds the start
            time of the specified trajectory IDs the DF and then another returns dataframe containing
            start latitude, start longitude, DateTime and trajectory ID for each trajectory

            Parameter
            ---------
                dataframe: PTRAILDataFrame
                    The dataframe containing the original data.
                ids_: list
                    List of trajectory ids for which the start times are to be calculated

            Returns
            -------
                pandas.core.dataframe.Dataframe
                    New dataframe containing Trajectory ID as index and start time of all
                    trajectories.
        """
        results = []

        # Loops over the length of trajectory ids. Filter the dataframe according to each of the ids
        # and then further filter that dataframe according to the earliest(minimum) time.
        # And then append the data of that earliest time into a list.
        for i in range(len(ids_)):
            filt = (dataframe.loc[dataframe[const.TRAJECTORY_ID] == ids_[i],
                                  [const.DateTime, const.LAT, const.LONG]])
            start_time = (filt.loc[filt[const.DateTime] == filt[const.DateTime].min()]).reset_index()
            results.append([start_time[const.DateTime][0], ids_[i]])

        # Make a new dataframe containing Latitude Longitude and Trajectory id
        df = pd.DataFrame(results).reset_index(drop=True).rename(columns={0: const.DateTime,
                                                                          1: const.TRAJECTORY_ID})
        # Return the dataframe by setting Trajectory id as index
        return df.set_index(const.TRAJECTORY_ID)

    @staticmethod
    def end_time_helper(dataframe, ids_):
        """
            This function is the helper function of the get_end_time(). The get_end_time() function
            delegates the task of calculating the end_time of the trajectories in the dataframe because the
            original functions runs multiple instances of this function in parallel. This function finds the end
            time of the specified trajectory IDs the DF and then another returns dataframe containing
            end latitude, end longitude, DateTime and trajectory ID for each trajectory

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The dataframe containing the original data.
                ids_: list
                    List of trajectory ids for which the end times are to be calculated

            Returns
            -------
                pandas.core.dataframe.Dataframe
                    New dataframe containing Trajectory ID as index end time of all trajectories.
        """
        results = []

        # Loops over the length of trajectory ids. Filter the dataframe according to each of the ids
        # and then further filter that dataframe according to the latest(maximum) time.
        # And then append the data of that latest time into a list.
        for i in range(len(ids_)):
            filt = (dataframe.loc[dataframe[const.TRAJECTORY_ID] == ids_[i],
                                  [const.DateTime, const.LAT, const.LONG]])
            start_time = (filt.loc[filt[const.DateTime] == filt[const.DateTime].max()]).reset_index()
            results.append([start_time[const.DateTime][0], ids_[i]])

        # Make a new dataframe containing Latitude Longitude and Trajectory id
        df = pd.DataFrame(results).reset_index(drop=True).rename(columns={0: const.DateTime,
                                                                          1: const.TRAJECTORY_ID})
        # Return the dataframe by setting Trajectory id as index
        return df.set_index(const.TRAJECTORY_ID)


    # -------------------------------------- Spatial Helpers ----------------------------------------------- #
    @staticmethod
    def distance_between_consecutive_helper(dataframe):
        """
            This function is the helper function of the create_distance_between_consecutive_column() function.
            The create_distance_between_consecutive_column() function delegates the actual task of calculating
            the distance between 2 consecutive points. This function does the calculation and creates a column
            called Distance_prev_to_curr and places it in the dataframe and returns it.

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The dataframe on which calculation is to be performed.

            Returns
            -------
                core.TrajectoryDF.PTRAILDataFrame
                    The dataframe containing the resultant Distance_prev_to_curr column.

            References
            ----------
                Arina De Jesus Amador Monteiro Sanches. 'Uma Arquitetura E Imple-menta ̧c ̃ao
                Do M ́odulo De Pr ́e-processamento Para Biblioteca Pymove'.Bachelor’s thesis.
                Universidade Federal Do Cear ́a, 2019.
        """
        # Reset the index and set it to trajectory ID in order to iterate
        # over the dataframe based on trajectory ID.
        dataframe = dataframe.reset_index().set_index(const.TRAJECTORY_ID)
        ids_ = dataframe.index.unique()  # Find out all the unique IDs in the dataframe.

        # For each unique ID, calculate the haversine distance and then assign it to
        # a new column and add that column to the dataframe.
        for val in ids_:
            curr_lat = dataframe.at[val, 'lat']
            curr_lon = dataframe.at[val, 'lon']
            size_id = curr_lat.size

            # Check whether the IDs are changing in the dataset and if they are, then assign
            # Nan as the value to the first point of the new Trajectory ID.
            if size_id <= 1:
                dataframe.at[val, 'Distance'] = np.nan
            else:
                prev_lat = curr_lat.shift(1)
                prev_lon = curr_lon.shift(1)
                dataframe.at[val, 'Distance'] = \
                    calc.haversine_distance(prev_lat, prev_lon, curr_lat, curr_lon)

        return dataframe.reset_index()  # Reset the index and return the dataframe.

    @staticmethod
    def distance_from_start_helper(dataframe):
        """
            This function is the helper function of the create_distance_from_start_column() function.
            The create_distance_from_start_column() function delegates the actual task of calculating
            the distance between 2 the start point of the trajectory to the current point.This function
            does the calculation and creates a column called Distance_start_to_curr and places it in the
            dataframe and returns it.

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The dataframe on which calculation is to be performed.

            Returns
            -------
                pandas.core.dataframe
                    The dataframe containing the resultant Distance_start_to_curr column.
        """
        dataframe = dataframe.reset_index().set_index(const.TRAJECTORY_ID)
        ids_ = dataframe.index.unique()  # Find out all the unique IDs in the dataframe.

        # For each unique ID, calculate the haversine distance and then assign it to
        # a new column and add that column to the dataframe.
        for val in ids_:
            curr_lat = dataframe.at[val, const.LAT]
            curr_lon = dataframe.at[val, const.LONG]
            size_id = curr_lat.size

            # Here, the purpose of the if else statement is as follows:
            #   In the above curr_lat variable, when the current latitude is extracted, sometimes
            #   if the trajectory has only a single point, then it only returns a single float value
            #   which naturally cannot be indexed. This if else statement checks if the value returned
            #   by the curr_lat extraction is a float and if so, then dont try to index it and instead
            #   take the value itself. The values are also shifted by 1 to avoid the distance calculation
            #   yielding the value 0 and instead give out NaN.
            start_lat = pd.Series(np.full(size_id,
                                          curr_lat[0] if type(curr_lat) is not np.float64 else curr_lat)).shift(1)
            start_lon = pd.Series(np.full(size_id,
                                          curr_lon[0] if type(curr_lat) is not np.float64 else curr_lat)).shift(1)

            # Check whether the IDs are changing in the dataset and if they are, then assign
            # Nan as the value to the first point of the new Trajectory ID.
            if size_id <= 1:
                dataframe.at[val, 'Distance_from_start'] = np.nan

            else:
                dataframe.at[val, 'Distance_from_start'] = \
                    calc.haversine_distance(start_lat, start_lon, curr_lat, curr_lon)

        return dataframe.reset_index()  # Reset the index and return the dataframe.

    @staticmethod
    def distance_from_given_point_helper(dataframe, coordinates):
        """
            This function is the helper function of the create_distance_from_point() function. The
            create_distance_from_point() function delegates the actual task of calculating distance
            between the given point to all the points in the dataframe to this function. This function
            calculates the distance and creates another column called 'Distance_to_specified_point'.

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The dataframe on which calculation is to be done.
                coordinates: tuple
                    The coordinates from which the distance is to be calculated.

            Returns
            -------
                pandas.core.dataframe.DataFrame
                    The dataframe containing the resultant Distance_from_(x, y) column.
        """
        # First, lets fetch the latitude and longitude columns from the dataset and store it
        # in a numpy array.
        latitudes = np.array(dataframe[const.LAT])
        longitudes = np.array(dataframe[const.LONG])
        distances = np.zeros(len(latitudes))

        # Now, lets calculate the Great-Circle (Haversine) distance between the 2 points and store
        # each of the values in the distance numpy array.
        for i in range(len(latitudes)):
            distances[i] = calc.haversine_distance(coordinates[0], coordinates[1],
                                                   latitudes[i], longitudes[i])

        dataframe[f'Distance_from_{coordinates}'] = distances
        return dataframe

    @staticmethod
    def point_within_range_helper(dataframe, coordinates, dist_range):
        """
            This is the helper function for create_point_within_range() function. The
            create_point_within_range_column()

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The dataframe on which the operation is to be performed.
                coordinates: tuple
                    The coordinates from which the distance is to be checked.
                dist_range:
                    The range within which the distance from the coordinates should lie.

            Returns
            -------
                pandas.core.dataframe.DataFrame
                    The dataframe containing the resultant Within_X_m_from_(x,y) column.
        """
        # First, lets fetch the latitude and longitude columns from the dataset and store it
        # in a numpy array.
        latitudes = np.array(dataframe[const.LAT])
        longitudes = np.array(dataframe[const.LONG])
        distances = []

        # Now, lets calculate the Great-Circle (Haversine) distance between the 2 points and then check
        # whether the distance is within the user specified range and store each of the values in the \
        # distance numpy array.
        for i in range(len(latitudes)):
            distances.append(calc.haversine_distance(coordinates[0], coordinates[1],
                                                     latitudes[i], longitudes[i]) <= dist_range)

        # Now, assign the column containing the results calculated above and
        # return the dataframe.
        dataframe[f'Within_{dist_range}_m'] = distances
        return dataframe

    @staticmethod
    def bearing_helper(dataframe):
        """
            This function is the helper function of the create_bearing_column(). The create_bearing_column()
            delegates the task of calculation of bearing between 2 points to this function because the original
            functions runs multiple instances of this function in parallel. This function does the calculation
            of bearing between 2 consecutive points in the entire DF and then creates a column in the dataframe
            and returns it.

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The dataframe on which the calculation is to be done.

            Returns
            -------
                PTRAILDataFrame:
                    The dataframe containing the Bearing column.
        """
        # Reset the index and set it to trajectory ID in order to iterate
        # over the dataframe based on trajectory ID.
        dataframe = dataframe.reset_index().set_index(const.TRAJECTORY_ID)
        ids_ = dataframe.index.unique()  # Find out all the unique IDs in the dataframe.

        # For each unique ID, calculate the haversine distance and then assign it to
        # a new column and add that column to the dataframe.
        for val in ids_:
            curr_lat = dataframe.at[val, 'lat']
            curr_lon = dataframe.at[val, 'lon']
            size_id = curr_lat.size

            # Check whether the IDs are changing in the dataset and if they are, then assign
            # Nan as the value to the first point of the new Trajectory ID.
            if size_id <= 1:
                dataframe.at[val, 'Bearing'] = np.nan
            else:
                prev_lat = curr_lat.shift(1)
                prev_lon = curr_lon.shift(1)
                dataframe.at[val, 'Bearing'] = \
                    calc.bearing_calculation(prev_lat, prev_lon, curr_lat, curr_lon)

        return dataframe.reset_index()

    @staticmethod
    def start_location_helper(dataframe, ids_):
        """
            This function is the helper function of the get_start_location(). The get_start_location() function
            delegates the task of calculating the start location of the trajectories in the dataframe because the
            original functions runs multiple instances of this function in parallel. This function finds the start
            location of the specified trajectory IDs the DF and then another returns dataframe containing start
            latitude, start longitude and trajectory ID for each trajectory


            Parameter
            ---------
                dataframe: PTRAILDataFrame
                    The dataframe of which the locations are to be found.dataframe
                ids_: list
                    List of trajectory ids for which the start locations are to be calculated

            Returns
            -------
                pandas.core.dataframe.Dataframe
                    New dataframe containing Trajectory as index and latitude and longitude
        """
        results = []

        # Loops over the length of trajectory ids. Filter the dataframe according to each of the ids
        # and then further filter that dataframe according to the earliest(minimum) time.
        # And then append the start location of that earliest time into a list
        for i in range(len(ids_)):
            filt = (dataframe.loc[dataframe[const.TRAJECTORY_ID] == ids_[i],
                                  [const.DateTime, const.LAT, const.LONG]])
            start_loc = (filt.loc[filt[const.DateTime] == filt[const.DateTime].min(),
                                  [const.LAT, const.LONG]]).reset_index()
            results.append([start_loc[const.LAT][0], start_loc[const.LONG][0], ids_[i]])

        # Make a new dataframe containing Latitude Longitude and Trajectory id
        df = pd.DataFrame(results).reset_index(drop=True).rename(columns={0: const.LAT,
                                                                          1: const.LONG,
                                                                          2: const.TRAJECTORY_ID})

        # Return the dataframe by setting Trajectory id as index
        return df.set_index(const.TRAJECTORY_ID)

    @staticmethod
    def end_location_helper(dataframe, ids_):
        """
            This function is the helper function of the get_end_location(). The get_end_location() function
            delegates the task of calculating the end location of the trajectories in the dataframe because the
            original functions runs multiple instances of this function in parallel. This function finds the end
            location of the specified trajectory IDs the DF and then another returns dataframe containing
            end latitude, end longitude and trajectory ID for each trajectory

            Parameter
            ---------
                dataframe: PTRAILDataFrame
                    The dataframe of which the locations are to be found.dataframe
                ids_: list
                    List of trajectory ids for which the end locations are to be calculated

            Returns
            -------
                pandas.core.dataframe.Dataframe
                    New dataframe containing Trajectory ID as index and latitude and longitude
                    as other 2 columns.
        """
        results = []

        # Loops over the length of trajectory ids. Filter the dataframe according to each of the ids
        # and then further filter that dataframe according to the latest(maximum) time.
        # And then append the end location of that latest time into a list.
        for i in range(len(ids_)):
            filt = (dataframe.loc[dataframe[const.TRAJECTORY_ID] == ids_[i],
                                  [const.DateTime, const.LAT, const.LONG]])
            start_loc = (filt.loc[filt[const.DateTime] == filt[const.DateTime].max(),
                                  [const.LAT, const.LONG]]).reset_index()
            results.append([start_loc[const.LAT][0], start_loc[const.LONG][0], ids_[i]])

        # Make a new dataframe containing Latitude Longitude and Trajectory id
        df = pd.DataFrame(results).reset_index(drop=True).rename(columns={0: const.LAT,
                                                                          1: const.LONG,
                                                                          2: const.TRAJECTORY_ID})
        # Return the dataframe by setting Trajectory id as index
        return df.set_index(const.TRAJECTORY_ID)

    @staticmethod
    def number_of_location_helper(dataframe, ids_):
        """
            This is the helper function for the get_number_of_locations() function. The
            get_number_of_locations() delegates the actual task of calculating the number of
            unique locations visited by a particular object to this function. This function
            calculates the number of unique locations by each of the unique object and returns
            a dataframe containing the results.

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                    The dataframe containing all the original data.
                ids_: list
                    The list of ids for which the number of unique locations visited
                    is to be calculated.

            Returns
            -------
                pandas.core.dataframe.DataFrame
                    dataframe containing the results.
        """
        results = []  # A list for storing results.

        # Iterate over each ID and calculate the number of locations visited by each ID.
        for i in range(len(ids_)):
            # Filter out only the points of the ID in question.
            filt = (dataframe.loc[dataframe[const.TRAJECTORY_ID] == ids_[i],
                                  [const.DateTime, const.LAT, const.LONG]])

            # Calculate the total number of unique (lat, lon) points visited
            # by the ID and append a row containing [# of unique locations, traj_id]
            # to the results list.
            results.append([filt.groupby([const.LAT, const.LONG]).ngroups, ids_[i]])

        # Convert the list containing results to a pandas dataframe, reset the index
        # and then rename the columns.
        df = pd.DataFrame(results).reset_index(drop=True).rename(columns={0: "Number of Unique Coordinates",
                                                                          1: const.TRAJECTORY_ID})
        # Set the index to traj_id and return it.
        return df.set_index(const.TRAJECTORY_ID)

    # ------------------------------------ Semantic Helpers ------------------------------------- #
    @staticmethod
    def visited_poi_helper(df, surrounding_data, dist_column_label, nearby_threshold):
        """
            Given a Trajectory dataframe and another dataset with the surrounding data,
            find whether the given object is nearby a point of interest or not.

            Parameters
            ----------
                df:
                    The dataframe containing the trajectory data.
                surrounding_data:
                    The dataframe containing the data of the surroundings.
                dist_column_label: Text
                    The label of the column containing the distance of the coords from
                    the nearest POI.
                nearby_threshold: int
                    The maximum distance between the POI and the current location of the object
                    within which the object is considered to be crossing/visiting the POI.

            Returns
            -------
                The original dataframe with another column added to it indicating whether
                each point is within
        """
        # A boolean list to store if that point in trajectory lies around a POI.
        POI = []
        df2 = surrounding_data.copy()

        try:
            # Loop for every point in the dataframe and create a distance column with distance from every point in the
            # surrounding data. Then use this distance column and the distance of POI column and compare each value.
            # Store the comparison in an array True if they lie within the threshold else false. And if any of the
            # value in it is true then that point in the dataframe was near a POI.
            for i in range(len(df)):
                dist_array = Helpers.distance_from_given_point_helper(df2, (df['lat'][i], df['lon'][i]))[
                    f'Distance_from_{df["lat"][i], df["lon"][i]}'].to_numpy()
                poi_array = df2[dist_column_label].to_numpy()

                dist_comp = np.abs(poi_array - dist_array) <= nearby_threshold
                POI.append(np.any(dist_comp))

            # Append the boolean list containing whether each point was near the POI of interest or not
            df['Nearby_POI'] = POI
            print("Done")
            return df
        except KeyError:
            raise KeyError(f"The column {dist_column_label} does not exist in the dataset.")

    # ------------------------------------ General Utilities ------------------------------------ #
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
        num = os.cpu_count()
        num = int(num)
        NUM_CPU = ceil((num * 2) / 3)

        # Integer divide the total number of Trajectory IDs by the number of available CPUs
        # and square the number because if too many partitions are made, then it does more
        # harm than good for the execution speed. The factor of 1 is added to avoid errors
        # when the integer division yields a 0.
        factor = ((size // NUM_CPU) ** 2) + 1

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

            Note
            ----
                The dataframe is split based on the number of CPU cores available for.
                For more info, take a look at the documentation of the get_partition_size()
                function.

            Parameters
            ----------
                dataframe: PTRAILDataFrame
                The dataframe that is to be split.

            Returns
            -------
                list:
                    The list containing smaller dataframe chunks.
        """
        # First, create a list containing all the ids of the data and then further divide that
        # list items and split it into sub-lists of ids equal to split_factor.
        ids_ = list(dataframe.reset_index().traj_id.value_counts().keys())

        # Get the ideal number of IDs by which the dataframe is to be split.
        split_factor = Helpers._get_partition_size(len(ids_))
        ids_ = [ids_[i: i + split_factor] for i in range(0, len(ids_), split_factor)]

        # Now split the dataframes based on set of Trajectory ids.
        # As of now, each smaller chunk is supposed to have data of 100
        # trajectory IDs max
        df_chunks = [dataframe.loc[dataframe.index.get_level_values(const.TRAJECTORY_ID).isin(ids_[i])]
                     for i in range(len(ids_))]
        return df_chunks

    # @staticmethod
    # def _partition_size(size, cpu_count):
    #     """
    #         Takes number of ids and makes use of a formula that gives a factor to makes set of ids
    #         according to the number of processors available to work with.
    #
    #         Parameters
    #         ----------
    #             size: int
    #                 The total number of trajectory IDs in the dataset.
    #
    #         Returns
    #         -------
    #             int
    #                The factor by which the datasets are to be split.
    #     """
    #     # Based on the Operating system, get the number of CPUs available for
    #     # multiprocessing.
    #     available_cpus = len(os.sched_getaffinity(0)) if os.name == 'posix' \
    #         else psutil.cpu_count()  # Number of available CPUs.
    #
    #     # Integer divide the total number of Trajectory IDs by the number of available CPUs
    #     # and square the number because if too many partitions are made, then it does more
    #     # harm than good for the execution speed. The factor of 1 is added to avoid errors
    #     # when the integer division yields a 0.
    #     factor = ((size // cpu_count)) + 1
    #
    #     # Return the factor if it is less than 100 otherwise return 100.
    #     # This factor hence is capped at 100.
    #     return factor if factor < 100 else 100
