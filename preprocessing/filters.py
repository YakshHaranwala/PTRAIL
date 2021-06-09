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

import utilities.constants as const
from utilities.exceptions import *
from core.TrajectoryDF import NumPandasTraj as NumTrajDF


class Filters:
    @staticmethod
    def remove_duplicates(dataframe):
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
    def filter_by_traj_id(dataframe, traj_id: Text):
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
    def filter_by_date(dataframe, date: Text):
        pass

    @staticmethod
    def filter_by_datetime(dataframe, dateTime: Text):
        pass

    @staticmethod
    def filter_by_max_speed(dataframe, max_speed: float):
        pass

    @staticmethod
    def filter_by_min_speed(dataframe, min_speed: float):
        pass

    @staticmethod
    def filter_by_min_consecutive_distance(dataframe, min_distance: float):
        pass

    @staticmethod
    def filter_by_max_consecutive_distance(dataframe, max_distance: float):
        pass

    @staticmethod
    def filter_by_radius_and_speed(dataframe, distance_threshold: float, speed_threshold: float):
        pass

    @staticmethod
    def filter_outliers_by_consecutive_distance(dataframe):
        pass

    @staticmethod
    def filter_outliers_by_consecutive_speed(dataframe):
        pass

    @staticmethod
    def remove_trajectories_with_less_points(dataframe, num_min_points: Optional[int] = 2):
        pass

    @staticmethod
    def remove_short_and_trajectories_with_few_points(dataframe, min_dist: float, num_min_points: Optional[int] = 2):
        pass

    @staticmethod
    def filter_by_label(dataframe, column_name, value):
        pass

    @staticmethod
    def filter_outliers_knn(dataframe, k: int):
        pass
