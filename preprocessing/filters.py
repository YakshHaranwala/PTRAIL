"""
    The filters module contains several data filtering functions like
    filtering the data based on time, date, proximity to a point and
    several others.

    @Authors: Yaksh J Haranwala, Salman Haidri
    @Date: 8th June, 2021
    @Version: 1.0
"""
import math
from typing import Text, Optional

import numpy as np
import pandas as pd

import utilities.constants as const
from core.TrajectoryDF import NumPandasTraj as NumTrajDF
from utilities.exceptions import *


# TODO: When we filter out the trajectories based o distance and time, i think we need to
#      calculate the actual distances again as I feel that the distance reported are 
#      incorrect because there might be some kinematic features which are calculated from the 
#      points that were removed and those features now would be reported incorrectly.


class Filters:
    @staticmethod
    def remove_duplicates(dataframe: NumTrajDF):
        """
            Drop duplicates based on the four following columns:
                1. Trajectory ID
                2. DateTime
                3. Latitude
                4. Longitude
            Duplicates will be dropped only when all the values in the above mentioned
            four columns are the same.

            Returns
            -------
                NumPandasTraj
                    The dataframe with dropped duplicates.
        """
        return dataframe.reset_index().drop_duplicates(
            subset=[const.DateTime, const.TRAJECTORY_ID, const.LAT, const.LONG],
            keep='first')

    @staticmethod
    def filter_by_traj_id(dataframe: NumTrajDF, traj_id: Text):
        """
            Extract all the trajectory points of a particular trajectory specified
            by the trajectory's ID.

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The dataframe on which the filtering by ID is to be done.
                traj_id: Text
                    The ID of the trajectory which is to be extracted.

            Returns
            -------
                pandas.core.dataframe.DataFrame
                    The dataframe containing all the trajectory points of the specified trajectory.

            Raises
            ------
                MissingTrajIDException:
                    This exception is raised when the Trajectory ID given by the user does not exist
                    in the dataset.

        """
        to_return = dataframe.reset_index().loc[dataframe.reset_index()[const.TRAJECTORY_ID] == traj_id]
        if len(to_return) > 0:
            return to_return
        else:
            raise MissingTrajIDException(f"{traj_id} is not present in the dataset. "
                                         f"Please check Trajectory ID and try again.")

    @staticmethod
    def get_bounding_box_by_radius(lat: float, lon: float, radius: float):
        """
            Calculates bounding box from a point according to the given radius.

            Parameters
            ----------
                lat: float
                    The latitude of centroid point of the bounding box.
                lon: float
                    The longitude of centroid point of the bounding box.
                radius: float
                    The max radius of the bounding box.
                    The radius is given in metres.

            Returns
            -------
                tuple:
                    The bounding box of the user specified size.

            References
            ----------
                https://mathmesquita.dev/2017/01/16/filtrando-localizacao-em-um-raio.html
        """
        lat, lon = math.radians(lat), math.radians(lon)  # Convert latitude, longitude to radians.

        # Calculate the delta factor for the latitudes and then
        # find the minimum and maximum latitudes.
        latitude_delta = radius / (const.RADIUS_OF_EARTH * 1000)
        lat_one = math.degrees(lat - latitude_delta)
        lat_two = math.degrees(lat + latitude_delta)

        # Calculate the delta factor for the longitudes and then
        # find the minimum and maximum longitudes.
        longitude_delta = math.asin((math.sin(latitude_delta)) / math.cos(lat))
        lon_one = math.degrees(lon - longitude_delta)
        lon_two = math.degrees(lon + longitude_delta)
        # Return the bounding box.
        return (lat_one, lon_one,
                lat_two, lon_two)

    @staticmethod
    def filter_by_bounding_box(dataframe: NumTrajDF, bounding_box: tuple, inside: bool = True):
        """
            Given a bounding box, filter out all the points that are within/outside
            the bounding box and return a dataframe containing the filtered points.

            Parameters
            ----------
                dataframe: NumTrajDF
                    The dataframe from which the data is to be filtered out.
                bounding_box: tuple
                    The bounding box which is to be used to filter the data.
                inside: bool
                    Indicate whether the data outside the bounding box is required
                    or the data inside it.

            Returns
            -------
                NumPandasTraj
                    The filtered dataframe.
        """
        filt = (
                (dataframe[const.LAT] >= bounding_box[0])
                & (dataframe[const.LONG] >= bounding_box[1])
                & (dataframe[const.LAT] <= bounding_box[2])
                & (dataframe[const.LONG] <= bounding_box[3])
        )
        df = dataframe.loc[filt] if inside else dataframe.loc[~filt]
        return NumTrajDF(df.reset_index(), const.LAT, const.LONG, const.DateTime, const.TRAJECTORY_ID)

    @staticmethod
    def filter_by_date(dataframe: NumTrajDF, start_date: Optional[Text] = None, end_date: Optional[Text] = None):
        """
            Filter the dataset by user-given time range.
            NOTE: The following options are to be noted for filtering the data.
                    1. If the start_date and end_date both are not given, then entire
                       dataset itself is returned.
                    2. If only start_date is given, then the trajectory data after
                       (including the start date) the start date is returned.
                    3. If only end_date is given, then the trajectory data before
                       (including the end date) the end date is returned.
                    4. If start_date and end_date both are given then the data between
                       the start_date and end_date (included) are returned.

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The dataframe that is to be filtered.
                start_date: Optional[Text]
                    The start date from which the points are to be filtered.
                end_date: Optional[Text]
                    The end date before which the points are to be filtered.

            Returns
            -------
                NumPandasTraj
                    The filtered dataframe containing the resultant data.

            Raises
            ------
                KeyError:
                    When the Date column is not present in the data.
                ValueError:
                    When the start date is later than the end date.
        """
        try:
            # Reset Index and create a variable for storing filtered DataFrame.
            dataframe = dataframe.reset_index()
            filtered_df = None

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

            # Convert the smaller dataframe back to NumPandasTraj and return it.
            return NumTrajDF(filtered_df, const.LAT, const.LONG, const.DateTime, const.TRAJECTORY_ID)

        except KeyError:
            # Ask the user first to create a date.
            raise MissingColumnsException(f"The Date column is missing in the dataset. Please run the "
                                          f"create_date_column() function first before running this filter.")

    @staticmethod
    def filter_by_datetime(dataframe: NumTrajDF, start_dateTime: Optional[Text] = None,
                           end_dateTime: Optional[Text] = None):
        """
            Filter the dataset by user-given time range.
            NOTE: The following options are to be noted for filtering the data.
                    1. If the start_dateTime and end_dateTime both are not given, then entire
                       dataset itself is returned.
                    2. If only start_dateTime is given, then the trajectory data after
                       (including the start datetime) the start date is returned.
                    3. If only end_dateTime is given, then the trajectory data before
                       (including the end datetime) the end date is returned.
                    4. If start_dateTime and end_dateTime both are given then the data between
                       the start_dateTime and end_dateTime (included) are returned.

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The dataframe that is to be filtered.
                start_dateTime: Optional[Text]
                    The start dateTime from which the points are to be filtered.
                end_dateTime: Optional[Text]
                    The end dateTime before which the points are to be filtered.

            Returns
            -------
                NumPandasTraj
                    The filtered dataframe containing the resultant data.
        """
        # Reset Index and create a variable for storing filtered DataFrame.
        dataframe = dataframe.reset_index()
        filtered_df = None

        # Convert the user-given string datetime to pandas datetime format.
        start_dateTime = pd.to_datetime(start_dateTime) if start_dateTime is not None else None
        end_dateTime = pd.to_datetime(end_dateTime) if end_dateTime is not None else None

        # Case-1: No start and end datetime are give. Hence just return the original dataframe.
        if start_dateTime is None and end_dateTime is None:
            filtered_df = dataframe

        # Case-2: No start_dateTime is given. Hence, return all the points upto and including
        #         the points on the end datetime.
        elif start_dateTime is None and end_dateTime is not None:
            filt = dataframe[const.DateTime] <= end_dateTime
            filtered_df = dataframe.loc[filt]

        # Case-3: No end datetime is given. Hence, return all the point after and including the
        #         points on the start datetime.
        elif start_dateTime is not None and end_dateTime is None:
            filt = dataframe[const.DateTime] >= start_dateTime
            filtered_df = dataframe.loc[filt]

        # Case-4: Both the start datetime and end datetime are given. Hence, return the points between
        #         and including the points on start and end datetime.
        else:
            if end_dateTime < start_dateTime:
                raise ValueError(f"End Datetime should be later than Start Datetime.")
            else:
                filt = np.logical_and(dataframe[const.DateTime] >= start_dateTime,
                                      dataframe[const.DateTime] <= end_dateTime)
                filtered_df = dataframe.loc[filt].reset_index()

        # Convert the smaller dataframe back to NumPandasTraj and return it.
        return NumTrajDF(filtered_df, const.LAT, const.LONG, const.DateTime, const.TRAJECTORY_ID)

    @staticmethod
    def filter_by_max_speed(dataframe: NumTrajDF, max_speed: float):
        """
            Remove the data points which have speed more than a user given speed.
            NOTE: The max_speed is given in the units m/s (metres per second).
            Parameters
            ----------
                dataframe: NumPandasTraj
                    The dataframe which is to be filtered.
                max_speed: float
                    The speed threshold above which the points are to be removed.

            Returns
            -------
                NumPandasTraj:
                    NumPandasTraj containing the resultant dataframe.

            Raises
            ------
                KeyError:
                    The column 'Speed_prev_to_curr' is not present in the dataframe.
        """
        try:
            dataframe = dataframe.reset_index()
            # Filter out all the values lesser than the max speed
            # The NaNs are filled with max_speed+1 to avoid data loss as well as
            # to avoid false positives in calculation and comparison.
            filt = dataframe['Speed_prev_to_curr'].fillna(max_speed + 1) <= max_speed
            filtered_df = dataframe.loc[filt].reset_index(drop=True)

            # Convert the smaller dataframe back to NumPandasTraj and return it.
            return NumTrajDF(filtered_df, const.LAT, const.LONG, const.DateTime, const.TRAJECTORY_ID)
        except KeyError:
            raise MissingColumnsException(f"The column 'Speed_prev_to_curr is not present in the dataset. "
                                          f"Please run the function create_speed_from_prev_column() before"
                                          f" running this filter.")

    @staticmethod
    def filter_by_min_speed(dataframe, min_speed: float):
        """
            Remove the data points which have speed less than a user given speed.
            NOTE: The min_speed is given in the units m/s (metres per second).
            Parameters
            ----------
                dataframe: NumPandasTraj
                    The dataframe which is to be filtered.
                min_speed: float
                    The speed threshold below which the points are to be removed.

            Returns
            -------
                NumPandasTraj:
                    NumPandasTraj containing the resultant dataframe.

            Raises
            ------
                KeyError:
                    The column 'Speed_prev_to_curr' is not present in the dataframe.
        """
        try:
            dataframe = dataframe.reset_index()
            # Filter out all the values lesser than the max speed
            # The NaNs are filled with min_speed-1 to avoid data loss as well as
            # to avoid false positives in calculation and comparison.
            filt = dataframe['Speed_prev_to_curr'].fillna(min_speed - 1) >= min_speed
            filtered_df = dataframe.loc[filt].reset_index()

            # Convert the smaller dataframe back to NumPandasTraj and return it.
            return NumTrajDF(filtered_df, const.LAT, const.LONG, const.DateTime, const.TRAJECTORY_ID)
        except KeyError:
            raise MissingColumnsException(f"The column 'Speed_prev_to_curr is not present in the dataset. "
                                          f"Please run the function create_speed_from_prev_column() before"
                                          f" running this filter.")

    @staticmethod
    def filter_by_min_consecutive_distance(dataframe, min_distance: float):
        """
            Remove the points that have a distance between 2 consecutive points
            lesser than a user specified value.
            NOTE: min_distance is given in metres.

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The dataframe which is to be filtered.
                min_distance: float
                    The consecutive distance threshold below which the points are to
                    be removed.

            Returns
            -------
                NumPandasTraj:
                    The filtered dataframe.

            Raises
            ------
                KeyError:
                    The column 'Distance_prev_to_curr' is not present in the dataframe.

        """
        try:
            dataframe = dataframe.reset_index()
            # Filter out all the values lesser than the minimum consecutive distance.
            # The NaNs are filled with min_distance-1 to avoid data loss as well as
            # to avoid false positives in calculation and comparison.
            filt = dataframe['Distance_prev_to_curr'].fillna(min_distance - 1) >= min_distance
            filtered_df = dataframe.loc[filt].reset_index(drop=True)

            # Convert the smaller dataframe back to NumPandasTraj and return it.
            return NumTrajDF(filtered_df, const.LAT, const.LONG, const.DateTime, const.TRAJECTORY_ID)
        except KeyError:
            raise MissingColumnsException(f"The column 'Distance_prev_to_curr is not present in the dataset. "
                                          f"Please run the function create_distance_between_consecutive_column() before"
                                          f" running this filter.")

    @staticmethod
    def filter_by_max_consecutive_distance(dataframe, max_distance: float):
        """
            Remove the points that have a distance between 2 consecutive points
            greater than a user specified value.
            NOTE: max_distance is given in metres.

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The dataframe which is to be filtered.
                max_distance: float
                    The consecutive distance threshold above which the points are to
                    be removed.

            Returns
            -------
                NumPandasTraj:
                    The filtered dataframe.

            Raises
            ------
                KeyError:
                    The column 'Distance_prev_to_curr' is not present in the dataframe.

                """
        try:
            dataframe = dataframe.reset_index()
            # Filter out all the values greater than the maximum consecutive distance.
            # The NaNs are filled with max_distance+1 to avoid data loss as well as
            # to avoid false positives in calculation and comparison.
            filt = dataframe['Distance_prev_to_curr'].fillna(max_distance + 1) <= max_distance
            filtered_df = dataframe.loc[filt]

            # Convert the smaller dataframe back to NumPandasTraj and return it.
            return NumTrajDF(filtered_df, const.LAT, const.LONG, const.DateTime, const.TRAJECTORY_ID)
        except KeyError:
            raise MissingColumnsException(f"The column 'Distance_prev_to_curr is not present in the dataset. "
                                          f"Please run the function create_distance_between_consecutive_column() before"
                                          f" running this filter.")

    @staticmethod
    def filter_by_max_distance_and_speed(dataframe, max_distance: float, max_speed: float):
        """
            Filter out values that have distance between consecutive points greater
            than a user-given distance speed between consecutive points greater than
            a user-given speed.
            NOTE: The max_distance is given in metres
            NOTE: The max_speed is given in metres/second (m/s).

            Parameters
            ----------
            dataframe: NumPandasTraj
                The dataframe which is to be filtered.
            max_distance: float
                The maximum distance between 2 consecutive points.
            max_speed: float
                The maximum speed between 2 consecutive points.

            Returns
            -------
                NumPandasTraj:
                    The filtered dataframe.
        """
        try:
            # filt = Filters.filter_by_max_consecutive_distance(dataframe, max_distance)
            # filtered_df = Filters.filter_by_max_speed(filt, max_speed)

            # Filter the dataframe based on maximum distance and speed.
            filt = np.logical_and(dataframe['Distance_prev_to_curr'].fillna(max_distance + 1) <= max_distance,
                                  dataframe['Speed_prev_to_curr'].fillna(max_speed + 1) <= max_speed)
            filtered_df = dataframe.loc[filt]

            return filtered_df  # Return the df filtered on the basis of 2 constraints.
        except KeyError:
            # The system asks the user to only run create_speed_from_prev_column() function because
            # running create_speed_from_prev_column() function create the 'Distance_prev_to_curr'
            # function automatically if the column is already not present.
            raise MissingColumnsException(f"Either of the columns 'Distance_prev_to_curr' or 'Speed_prev_to_curr'"
                                          f"are missing in the dataset. Please run the function "
                                          f"create_speed_from_prev_column() first before running the function.")

    @staticmethod
    def filter_by_min_distance_and_speed(dataframe, min_distance: float, min_speed: float):
        """
            Filter out values that have distance between consecutive points lesser
            than a user-given distance speed between consecutive points lesser than
            a user-given speed.
            NOTE: The min_distance is given in metres
            NOTE: The min_speed is given in metres/second (m/s).

            Parameters
            ----------
            dataframe: NumPandasTraj
                The dataframe which is to be filtered.
            min_distance: float
                The minimum distance between 2 consecutive points.
            min_speed: float
                The minimum speed between 2 consecutive points.

            Returns
            -------
                NumPandasTraj:
                    The filtered dataframe.
        """
        try:
            # filt = Filters.filter_by_max_consecutive_distance(dataframe, max_distance)
            # filtered_df = Filters.filter_by_max_speed(filt, max_speed)

            # Filter the dataframe based on minimum distance and speed.
            filt = np.logical_and(dataframe['Distance_prev_to_curr'] >= min_distance,
                                  dataframe['Speed_prev_to_curr'] >= min_speed)
            filtered_df = dataframe.loc[filt]

            # Return the df filtered on the basis of 2 constraints.
            return filtered_df
        except KeyError:
            # The system asks the user to only run create_speed_from_prev_column() function because
            # running create_speed_from_prev_column() function create the 'Distance_prev_to_curr'
            # function automatically if the column is already not present.
            raise MissingColumnsException(f"Either of the columns 'Distance_prev_to_curr' or 'Speed_prev_to_curr'"
                                          f"are missing in the dataset. Please run the function "
                                          f"create_speed_from_prev_column() first before running the function.")

    @staticmethod
    def filter_outliers_by_consecutive_distance(dataframe: NumTrajDF):
        """
            Check the outlier points based on distance between 2 consecutive points.
            Outlier formula:
                Lower outlier = Q1 - (1.5*IQR)
                Higher outlier = Q3 + (1.5*IQR)

                IQR = Inter quartile range = Q3 - Q1
            We need to find points between lower and higher outlier

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The dataframe which is to be filtered.

            Returns
            -------
                NumPandasTraj:
                    The dataframe which has been filtered.

            Raises
            ------
                KeyError:
                    The column 'Distance_prev_to_curr' is not present in the dataset.
        """
        try:
            # Find the lower and higher quantile first along with the inter-quantile range.
            dataframe = dataframe.reset_index()
            q_low = dataframe['Distance_prev_to_curr'].quantile(0.25)
            q_high = dataframe['Distance_prev_to_curr'].quantile(0.75)
            iqr = q_high - q_low
            cut_off = iqr * 1.5  # Cut off value.

            # Now, find the upper limit and the lower limit for the data.
            lower = q_low - cut_off
            higher = q_high + cut_off

            # Filter out the dataframe based on the range calculated and return it.
            df_filt = np.logical_and(dataframe['Distance_prev_to_curr'] > lower,
                                     dataframe['Distance_prev_to_curr'] < higher)

            filtered_df = dataframe.loc[df_filt]
            return NumTrajDF(filtered_df, const.LAT, const.LONG, const.DateTime, const.TRAJECTORY_ID)

        except KeyError:
            raise MissingColumnsException(f"The column 'Distance_prev_to_curr' is missing in the dataset. "
                                          f"Please run the function create_distance_between_consecutive_column() "
                                          f"before running this filter.")

    @staticmethod
    def filter_outliers_by_consecutive_speed(dataframe):
        """
            Check the outlier points based on distance between 2 consecutive points.
            Outlier formula:
                Lower outlier = Q1 - (1.5*IQR)
                Higher outlier = Q3 + (1.5*IQR)

                IQR = Inter quartile range = Q3 - Q1
            We need to find points between lower and higher outlier

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The dataframe which is to be filtered.

            Returns
            -------
                NumPandasTraj:
                    The dataframe which has been filtered.

            Raises
            ------
                KeyError:
                The column 'Distance_prev_to_curr' is not present in the dataset.
        """
        try:
            q_low = dataframe['Speed_prev_to_curr'].quantile(0.25)
            q_high = dataframe['Speed_prev_to_curr'].quantile(0.75)
            iqr = q_high - q_low
            cut_off = iqr * 1.5

            lower = q_low - cut_off
            higher = q_high + cut_off

            df_filt = np.logical_and(dataframe['Speed_prev_to_curr'] > lower,
                                     dataframe['Speed_prev_to_curr'] < higher)
            return dataframe.loc[df_filt]

        except KeyError:
            raise MissingColumnsException(f"The column 'Speed_prev_to_curr' is missing in the dataset. "
                                          f"Please run the function create_speed_from_prev_column() "
                                          f"before running this filter.")

    @staticmethod
    def remove_trajectories_with_less_points(dataframe, num_min_points: Optional[int] = 3):
        """
            Remove out the trajectories from the dataframe which have few points.

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The dataframe from which trajectories with few points are to be
                    removed.
                num_min_points: Optional[int], default = 2
                    The minimum number of points that a trajectory should have if it
                    is to be retained in the dataset.

            Returns
            -------
                NumPandasTraj:
                    The filtered dataframe which does not contain the trajectories
                    with few points anymore.
        """
        dataframe = dataframe.reset_index()
        # Mapping the trajectory id column with only those trajectory ids that contain more
        # or equal points than specified by the user.
        filt = dataframe[const.TRAJECTORY_ID].map(dataframe[const.TRAJECTORY_ID].value_counts()) >= num_min_points

        # Apply the filter, convert the resultant dataframe to NumTrajDF and return it.
        df = dataframe[filt]
        return NumTrajDF(df, const.LAT, const.LONG, const.DateTime, const.TRAJECTORY_ID)

    # @staticmethod
    # def filter_outliers_knn(dataframe, k: int):
    #     pass
