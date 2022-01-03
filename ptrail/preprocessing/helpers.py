"""
    Warning
    -------
        | 1. None of the methods in this module should be used directly while performing operations on data.
        | 2. These methods are helpers for the interpolation methods in the interpolation.py
             module and hence run linearly and not in parallel which will result in slower execution time.
        | 3. All the methods in this module perform calculation on a single Trajectory ID due to which
             it will wrong results on data with multiple trajectories. Instead, use the interpolation.py
             methods for faster and reliable calculations.

    The helpers class has the functionalities that interpolate a point based
    on the given data by the user. The class contains the following 4
    interpolation calculators:

        1. Linear Interpolation
        2. Cubic Interpolation
        3. Random-Walk Interpolation
        4. Kinematic Interpolation

    Besides the interpolation helpers, there are also general utilities which
    are used in splitting up dataframes for running the code in parallel.

    | Authors: Yaksh J Haranwala, Salman Haidri
"""
import math
import os
from typing import Text, Union

import numpy as np
import pandas as pd
import datetime as dt
from hampel import hampel
from scipy.interpolate import CubicSpline

from ptrail.core.TrajectoryDF import PTRAILDataFrame
from ptrail.features.kinematic_features import KinematicFeatures as spatial
from ptrail.utilities import constants as const
from ptrail.utilities.exceptions import *


class Helpers:
    # ------------------------------------ Interpolation Helpers --------------------------------------- #
    @staticmethod
    def linear_help(dataframe: Union[pd.DataFrame, PTRAILDataFrame], id_: Text, time_jump: float):
        """
            This method takes a dataframe and uses linear interpolation to determine coordinates
            of location on Datetime where the time difference between 2 consecutive points exceeds
            the user-specified time_jump and inserts the interpolated point those between 2 points.

            Warning
            -------
                This method should not be used for dataframes with multiple trajectory ids as it will
                yield wrong results and there might be a significant drop in performance.

            Parameters
            ----------
                dataframe: Union[pd.DataFrame, NumTrajDF]
                     The dataframe containing the original trajectory data.
                id_: Text
                    The Trajectory ID of the points in the dataframe.
                time_jump: float
                    The maximum time difference between 2 points greater than which
                    a point will be inserted between 2 points.

            Returns
            -------
                pandas.core.dataframe.DataFrame
                    The dataframe containing the trajectory enhanced with interpolated
                    points.

        """
        # Now, for each unique ID in the dataframe, interpolate the points.
        # Create a Series containing new times which are calculated as follows:
        #    new_time[i] = original_time[i] + time_jump.
        new_times = dataframe.reset_index()[const.DateTime] + pd.to_timedelta(time_jump, unit='seconds')

        # Now, interpolate the latitudes using numpy based on the new times calculated above.
        ip_lat = np.interp(new_times,
                           dataframe.reset_index()[const.DateTime],
                           dataframe.reset_index()[const.LAT])

        # Now, interpolate the longitudes using numpy based on the new times calculated above.
        ip_long = np.interp(new_times,
                            dataframe.reset_index()[const.DateTime],
                            dataframe.reset_index()[const.LONG])

        # Here, store the time difference between all the consecutive points in an array.
        time_deltas = dataframe.reset_index()[const.DateTime].diff().dt.total_seconds()

        # Now, for each point in the trajectory, check whether the time difference between
        # 2 consecutive points is greater than the user-specified time_jump, and if so then
        # insert a new point that is linearly interpolated between the 2 original points.
        for j in range(len(time_deltas)):
            if time_deltas[j] > time_jump:
                dataframe.loc[new_times[j - 1]] = [id_, ip_lat[j - 1], ip_long[j - 1]]

        return dataframe

    @staticmethod
    def cubic_help(df: Union[pd.DataFrame, PTRAILDataFrame], id_: Text, time_jump: float):
        """
            This method takes a dataframe and uses cubic interpolation to determine coordinates
            of location on Datetime where the time difference between 2 consecutive points exceeds
            the user-specified time_jump and inserts the interpolated point those between 2 points.

            Warning
            -------
                This method should not be used for dataframes with multiple trajectory ids as it will
                yield wrong results and there might be a significant drop in performance.

            Parameters
            ----------
                df: Union[pd.DataFrame, NumTrajDF]
                     The dataframe containing the original trajectory data.
                id_: Text
                    The Trajectory ID of the points in the dataframe.
                time_jump: float
                    The maximum time difference between 2 points greater than which
                    a point will be inserted between 2 points.

            Returns
            -------
                pandas.core.dataframe.DataFrame
                    The dataframe containing the trajectory enhanced with interpolated
                    points.
        """
        # Create a Series containing new times which are calculated as follows:
        #    new_time[i] = original_time[i] + time_jump.
        new_times = df.reset_index()[const.DateTime] + pd.to_timedelta(time_jump, unit='seconds')

        # Extract the Latitude, Longitude pairs for each point and store it in a
        # numpy array.
        coords = df.reset_index()[[const.LAT, const.LONG]].to_numpy()

        # Now, using Scipy's Cubic spline, create a spline object for interpolation of
        # points for the dataframes which have a length greater than 3 else CubicSpline
        # doesn't execute.
        if len(df) > 3:
            cubic_spline = CubicSpline(x=df.reset_index()[const.DateTime].sort_values(),
                                       y=coords,
                                       extrapolate=True, bc_type='not-a-knot')

            # Now, calculate the interpolated position of the points at all the new_times
            #    calculated above.
            ip_coords = cubic_spline(new_times)

        # Here, store the time difference between all the consecutive points in an array.
        time_deltas = df.reset_index()[const.DateTime].diff().dt.total_seconds()

        # Now, for each point in the trajectory, check whether the time difference between
        # 2 consecutive points is greater than the user-specified time_jump, and if so then
        # insert a new point that is cubic-spline interpolated between the 2 original points.
        for j in range(len(time_deltas)):
            # If the trajectory has less than 3 points, then skip the trajectory
            # from the interpolation.
            if len(df) > 3:
                if time_deltas[j] > time_jump:
                    df.loc[new_times[j - 1]] = [id_, ip_coords[j - 1][0], ip_coords[j - 1][1]]

        return df

    @staticmethod
    def random_walk_help(dataframe: PTRAILDataFrame, id_: Text, time_jump: float):
        """
            This method takes a dataframe and uses random-walk interpolation to determine coordinates
            of location on Datetime where the time difference between 2 consecutive points exceeds
            the user-specified time_jump and inserts the interpolated point those between 2 points.

            Warning
            -------
                This method should not be used for dataframes with multiple trajectory ids as it will
                yield wrong results and there might be a significant drop in performance.

            Parameters
            ----------
                dataframe: Union[pd.DataFrame, NumTrajDF]
                     The dataframe containing the original trajectory data.
                id_: Text
                    The Trajectory ID of the points in the dataframe.
                time_jump: float
                    The maximum time difference between 2 points greater than which
                    a point will be inserted between 2 points.

            Returns
            -------
                pandas.core.dataframe.DataFrame
                    The dataframe containing the trajectory enhanced with interpolated
                    points.

            References
            ----------
                Etemad, M., Soares, A., Etemad, E. et al. SWS: an unsupervised trajectory
                segmentation algorithm based on change detection with interpolation kernels.
                Geoinformatica (2020)
        """
        # Create a Series containing new times which are calculated as follows:
        #    new_time[i] = original_time[i] + time_jump.
        new_times = dataframe.reset_index()[const.DateTime] + pd.to_timedelta(time_jump, unit='seconds')

        # First, create a distance between the consecutive points of the dataframe,
        # then, calculate the mean and standard deviation of all the distances between
        # consecutive points.
        df1 = spatial.create_distance_column(dataframe)
        d_mean = (df1['Distance'].mean(skipna=True))
        d_std = (df1['Distance'].std(skipna=True))

        # Now, create a bearing between the consecutive points of the dataframe,
        # then, calculate the mean and standard deviation of all the bearings between
        # consecutive points.
        df = spatial.create_bearing_column(df1)
        b_mean = df['Bearing'].mean(skipna=True)
        b_std = df['Bearing'].std(skipna=True)

        calc_a = np.random.normal(d_mean, d_std, 1) / 1000
        calc_b = np.radians(np.random.normal(b_mean, b_std, 1))

        # print(f"Calc: {calc_a, calc_b}")
        # if d_std == 0 or b_std == 0:
        #     return dataframe
        #
        # # Here, using Scipy's truncnorm() function, create an object that gives out random
        # # values. It is to be noted that the values are restricted between latitude.min()
        # # and latitude.max().
        # d_mean = truncnorm((df1['lat'].min() - d_mean) / d_std,
        #                    (df1['lat'].max() - d_mean) / d_std,
        #                    loc=d_mean,
        #                    scale=d_std)
        #
        # # Here, using Scipy's truncnorm() function, create an object that gives out random
        # # values. It is to be noted that the values are restricted between longitude.min()
        # # and longitude.max().
        # b_mean = truncnorm((df['lon'].min() - b_mean) / b_std,
        #                    (df['lon'].max() - b_mean) / b_std,
        #                    loc=b_mean,
        #                    scale=b_std)
        #
        # # Using the 2 objects created above, generate a random value from them. The value
        # # is selected randomly from a uniformly distributed sample
        # calc_a = d_mean.rvs()
        # calc_b = math.radians(b_mean.rvs())

        dy = calc_a * np.cos(calc_b)
        dx = calc_a * np.sin(calc_b)

        # Here, store the time difference between all the consecutive points in an array.
        time_deltas = df.reset_index()[const.DateTime].diff().dt.total_seconds()

        # Look for a time diff that exceeds the time_jump and if one is found, calculate the
        # latitude and longitude and then append them to the dataframe at the location where
        # the threshold is crossed.
        for i in range(len(time_deltas)):
            if len(df) > 3:
                if time_deltas[i] > time_jump:
                    new_lat = df[const.LAT].iloc[i - 1] + \
                              (dy / const.RADIUS_OF_EARTH) * (180 / np.pi)
                    new_lon = df[const.LONG].iloc[i - 1] + \
                              (dx / const.RADIUS_OF_EARTH) * (180 / np.pi) / np.cos(
                        df[const.LAT].iloc[i - 1] * np.pi / 180)
                    dataframe.loc[new_times[i - 1]] = [id_, new_lat[0], new_lon[0]]

        # Return the new dataframe
        return dataframe

    @staticmethod
    def kinematic_help(dataframe: Union[pd.DataFrame, PTRAILDataFrame], id_: Text, time_jump: float):
        """
            This method takes a dataframe and uses kinematic interpolation to determine coordinates
            of location on Datetime where the time difference between 2 consecutive points exceeds
            the user-specified time_jump and inserts the interpolated point those between 2 points.

            Warning
            -------
                This method should not be used for dataframes with multiple trajectory ids as it will
                yield wrong results and there might be a significant drop in performance.

            Parameters
            ----------
                dataframe: Union[pd.DataFrame, NumTrajDF]
                     The dataframe containing the original trajectory data.
                id_: Text
                    The Trajectory ID of the points in the dataframe.
                time_jump: float
                    The maximum time difference between 2 points greater than which
                    a point will be inserted between 2 points.

            Returns
            -------
                pandas.core.dataframe.DataFrame
                    The dataframe containing the trajectory enhanced with interpolated
                    points.

            References
            ----------
                Nogueira, T.O., "kinematic_interpolation.py", (2016), GitHub repository,
                https://gist.github.com/talespaiva/128980e3608f9bc5083b.js
        """
        # Create a Series containing new times which are calculated as follows:
        #    new_time[i] = original_time[i] + time_jump.
        new_times = dataframe.reset_index()[const.DateTime] + pd.to_timedelta(time_jump, unit='seconds')

        # Here, store the time difference between all the consecutive points in an array.
        time_deltas = dataframe.reset_index()[const.DateTime].diff().dt.total_seconds()
        lat_diff = dataframe.reset_index()[const.LAT].diff()
        lon_diff = dataframe.reset_index()[const.LONG].diff()

        lat_velocity = lat_diff / time_deltas
        lon_velocity = lon_diff / time_deltas

        lat = list(dataframe.reset_index()[const.LAT].values)
        lon = list(dataframe.reset_index()[const.LONG].values)

        # Look for a time diff that exceeds the time_jump and if one is found, calculate the
        # latitude and longitude and then append them to the dataframe at the location where
        # the threshold is crossed.
        for i in range(len(time_deltas)):
            if time_deltas[i] > time_jump and not np.isnan(lat_velocity[i - 1]):
                ax = np.array([[(time_deltas[i] ** 2) / 2, (time_deltas[i] ** 3) / 6],
                               [float(time_deltas[i]), (time_deltas[i] ** 2) / 2]])
                bx = [lat[i] - lat[i - 1] - lat_velocity[i - 1] * time_deltas[i], lat_velocity[i] - lat_velocity[i - 1]]
                coef_x = np.linalg.solve(ax, bx)

                ay = ax
                by = [lon[i] - lon[i - 1] - lon_velocity[i - 1] * time_deltas[i], lon_velocity[i] - lon_velocity[i - 1]]
                coef_y = np.linalg.solve(ay, by)

                td = new_times[i - 1].timestamp() / 10e9
                # x = lat[i-1] + lat_velocity[i-1] * td + \
                #     (td**2)*coef_x[0]/2 + (td**3)*coef_x[1]/6
                # y = lon[i-1] + lon_velocity[i-1] * td + \
                #     (td ** 2) * coef_y[0] / 2 + (td ** 3) * coef_y[1] / 6
                x = Helpers._pos(t=td, x1=lat[i - 1], v1=lat_velocity[i - 1], b=coef_x[0], c=coef_x[1])
                y = Helpers._pos(t=td, x1=lon[i - 1], v1=lon_velocity[i - 1], b=coef_y[0], c=coef_y[1])
                dataframe.loc[new_times[i - 1]] = [id_, x, y]

        return dataframe

    @staticmethod
    def hampel_help(df, column_name):
        """
            This function is the helper function for the hampel_outlier_detection()
            function present in the filters module. The purpose of the function is to
            run the hampel filter on a single trajectory ID, remove the outliers
            and return the smaller dataframe.

            Warning
            -------
                This function should not be used directly as it will result in a
                slower execution of the function and might result in removal of
                points that are actually not outliers.

            Warning
            -------
                Do not use Hampel filter outlier detection and try to detect outliers
                with DateTime as it will raise a NotImplementedError as it has not been
                implemented yet by the original author of the Hampel filter.

            Parameters
            ----------
                df: PTRAILDataFrame/pd.core.dataframe.DataFrame
                    The dataframe which the outliers are to be removed
                column_name: Text
                    The column based on which the outliers are to be removed.

            Returns
            -------
                pd.core.dataframe.DataFrame
                    The dataframe where the outlier points are removed.

        """
        try:
            # First, extract the column from the dataframe and then obtain the
            # outlier indices which are to be removed.
            col = df[column_name]
            outlier_indices = hampel(col)

            # Now, drop the indices given out by the hampel filter.
            to_return = df.drop(df.index[outlier_indices])
            #

            return to_return

        except KeyError:
            raise MissingColumnsException(f"The column {column_name} does not exist in the dataset."
                                          f"Please check the column name and try again.")

    @staticmethod
    def _pos(t, x1, v1, b, c):
        return x1 + v1 * t + (t ** 2) * b / 2 + (t ** 3) * c / 6

    # ------------------------------------------ Statistics Helpers ----------------------------------- #
    @staticmethod
    def split_traj_helper(df, num_days):
        # First, create the date column and get all the unique traj_ids
        # in the dataframe.
        df['Date'] = df[const.DateTime].dt.date
        ids_ = list(df.traj_id.value_counts().keys())

        df_chunks = []
        for i in range(len(ids_)):
            small_df = df.reset_index().loc[df.reset_index()[const.TRAJECTORY_ID] == ids_[i]]
            df_chunks.append(small_df)

        # Now, iterate over the entire dataframe and then segment
        # the trajectories by num_days each.
        results = []
        for i in range(len(ids_)):
            # Take the traj_df of a single Trajectory out from the
            # list of chunks and find their max and min timestamps.
            traj = df_chunks[i]
            t_max = traj[const.DateTime].max()
            t_min = traj[const.DateTime].min()

            # For iteration purposes, set t_1 to min and t_2 to
            # t_1 + num_days days.
            t_1 = t_min
            t_2 = t_1 + dt.timedelta(days=num_days)
            seg_id = 1

            # Now, segment the trajectories into smaller segments
            # wherein each segment contains the points of a span
            # of num_days days only.
            while t_2 < t_max:
                if t_2 < t_max:
                    seg = Helpers.filt_df_by_date(traj,
                                                  start_date=t_1.strftime('%Y-%m-%d'),
                                                  end_date=t_max.strftime('%Y-%m-%d'))
                    # Once filtered, assign the segment with a segment ID.
                    seg['seg_id'] = seg_id

                    # Increment the segment id, t_1 and t_2 values by
                    # 1, num_days days each respectively to continue the iteration.
                    t_1 += dt.timedelta(days=num_days)
                    t_2 += dt.timedelta(days=num_days)

                    if len(seg) > 0:
                        seg_id += 1

                    results.append(seg.drop(columns=['index', 'level_0']))

                # If, t_2 is greater than the max time present in the
                # trajectory, then assign t_2 = max and proceed
                # further with segmentation.
                elif t_2 >= t_max:
                    seg = Helpers.filt_df_by_date(traj,
                                                  start_date=t_1.strftime('%Y-%m-%d'),
                                                  end_date=t_max.strftime('%Y-%m-%d'))
                    # Once filtered, assign the segment with a segment ID.
                    seg['seg_id'] = seg_id

                    # Increment the segment id, t_1 and t_2 values by
                    # 1, num_days each respectively to continue the iteration.
                    t_1 += dt.timedelta(days=num_days)
                    t_2 += dt.timedelta(days=num_days)

                    if len(seg) > 0:
                        seg_id += 1

                    results.append(seg.drop(columns=['index', 'level_0']))
                    seg_id += 1

        # Finally, concat the dataframes, set the index as
        # [traj_id, seg_id, DateTime].
        return pd.concat(results).reset_index().set_index(['traj_id', 'seg_id', 'DateTime']).sort_values(by=['traj_id',
                                                                                                             'seg_id'])

    @staticmethod
    def filt_df_by_date(dataframe, start_date, end_date):
        # Convert the user-given string dates to pandas datetime format.
        start_date = pd.to_datetime(start_date) if start_date is not None else None
        end_date = pd.to_datetime(end_date) if end_date is not None else None

        # Case-1: No start and end date are give. Hence just return the original dataframe.
        if start_date is None and end_date is None:
            filtered_df = dataframe

        # Case-2: No start_date is given. Hence, return all the points upto and including
        #         the points on the end date.
        elif start_date is None and end_date is not None:
            filt = dataframe['Date'] <= end_date
            filtered_df = dataframe.loc[filt]

        # Case-3: No end date is given. Hence, return all the point after and including the
        #         points on the start date.
        elif start_date is not None and end_date is None:
            filt = dataframe['Date'] >= start_date
            filtered_df = dataframe.loc[filt]

        # Case-4: Both the start date and end date are given. Hence, return the points between
        #         and including the points on start and end date.
        else:
            if end_date < start_date:
                raise ValueError(f"End Date should be later than Start Date.")
            else:
                filt = np.logical_and(dataframe['Date'] >= start_date, dataframe['Date'] <= end_date)
                filtered_df = dataframe.loc[filt].reset_index()

        # Convert the smaller dataframe back to PTRAILDataFrame and return it.
        return filtered_df

    @staticmethod
    def stats_helper(df, target_col_name, segmented):
        """
            Generate the stats of the kinematic features present in the Dataframe.

            Parameters
            ----------
                df: pandas.core.dataframe.DataFrame
                    The dataframe containing the trajectory data and their features.
                target_col_name: str
                    This is the 'y' value that is used for ML tasks, this is
                    asked to append the species back at the end.
                segmented: Optional[bool]
                    Indicate whether the trajectory has segments or not.

            Returns
            -------
                pd.core.dataframe.DataFrame:
                    A dataframe containing the stats of the given trajectory.

        """
        # Grab the columns that we need.
        if not segmented:
            new_df = df.reset_index()[['traj_id', 'Distance', 'Distance_from_start', 'Speed',
                                       'Acceleration', 'Jerk', 'Bearing', 'Bearing_Rate',
                                       'Rate_of_bearing_rate']]
            # Generate the stats along with the needed percentiles and arrange the dataframe
            # properly.
            stats = new_df.reset_index(drop=True).describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).transpose()

            # Assign the traj_id column.
            stats['traj_id'] = new_df['traj_id'].iloc[0]
            stats = stats[['traj_id', 'mean', 'std', 'min', '10%',
                           '25%', '50%', '75%', '90%', 'max']]
            stats[target_col_name] = df[target_col_name].iloc[0]

            return stats.reset_index().rename(
                columns={'index': 'Columns'}).reset_index(drop=True).set_index(['traj_id', 'Columns'])
        else:
            seg_id = df['seg_id'].iloc[0]
            new_df = df.reset_index()[['traj_id', 'Distance', 'Distance_from_start', 'Speed',
                                       'Acceleration', 'Jerk', 'Bearing', 'Bearing_Rate',
                                       'Rate_of_bearing_rate']]

            # Generate the stats along with the needed percentiles and arrange the dataframe
            # properly.
            stats = new_df.reset_index(drop=True).describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).transpose()

            # Assign the traj_id column.
            stats['traj_id'] = new_df['traj_id'].iloc[0]
            stats['seg_id'] = seg_id
            stats = stats[['traj_id', 'seg_id', 'mean', 'std', 'min', '10%',
                           '25%', '50%', '75%', '90%', 'max']]
            stats[target_col_name] = df[target_col_name].iloc[0]

            to_return = stats.reset_index().rename(
                columns={'index': 'Columns'}).reset_index(drop=True).set_index(['traj_id', 'seg_id', 'Columns'])
            return to_return

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
        num = os.cpu_count()
        NUM_CPU = math.ceil((num * 2) / 3)

        # Integer divide the total number of Trajectory IDs by the number of available CPUs
        # The factor of 1 is added to avoid errors when the integer division yields a 0.
        factor = (size // NUM_CPU) + 1

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
        df_chunks = [dataframe.loc[dataframe[const.TRAJECTORY_ID].isin(ids_[i])]
                     for i in range(len(ids_))]
        return df_chunks
