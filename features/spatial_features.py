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

import multiprocessing
from typing import Optional, Text

import pandas as pd

from core.TrajectoryDF import NumPandasTraj
from utilities import constants as const
from utilities.helper_functions import Helpers as helpers


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
            Note that if the data does not have an Object ID column and does not have unique objects,
            then the entire dataset's starting location is returned.

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
        """
        # If traj_id is None, filter out a dataframe with the earliest time and then return the first 
        # latitude and longitude at that time.
        # Else first filter out a dataframe containing the given traj_id and then perform the same steps as
        # mentioned above
        dataframe = dataframe.copy().reset_index()
        if traj_id is None:
            start_loc = (dataframe.loc[dataframe[const.DateTime] == dataframe[const.DateTime].min(),
                                       [const.LAT, const.LONG]]).reset_index()
            return start_loc[const.LAT][0], start_loc[const.LONG][0]
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
        # If traj_id is None, filter out a dataframe with the latest time and then return the last
        # latitude and longitude at that time.
        # Else first filter out a dataframe containing the given traj_id and then perform the same steps as
        # mentioned above
        dataframe = dataframe.copy().reset_index()
        if traj_id is None:
            start_loc = (dataframe.loc[dataframe[const.DateTime] == dataframe[const.DateTime].max(),
                                       [const.LAT, const.LONG]]).reset_index()
            return start_loc[const.LAT][0], start_loc[const.LONG][0]
        else:
            filt = (dataframe.loc[dataframe[const.TRAJECTORY_ID] == traj_id, [const.DateTime, const.LAT, const.LONG]])
            start_loc = (filt.loc[filt[const.DateTime] == filt[const.DateTime].max(),
                                  [const.LAT, const.LONG]]).reset_index()
            if len(start_loc) == 0:
                return f"Trajectory ID: {traj_id} does not exist in the dataset. Please try again!"
            else:
                return start_loc[const.LAT][0], start_loc[const.LONG][0]

    @staticmethod
    def create_distance_between_consecutive_column(dataframe: NumPandasTraj, inplace=False, metres=False):
        """
            Create a column called Dist_prev_to_curr containing distance between 2 consecutive points.

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The data where speed is to be calculated.
               inplace: bool
                    Indication on whether the answer is to be returned as a NumPandasTraj DF
                    or a pandas DF.
                metres: bool
                    Indicate whether to return the distances in metres or kilometres.

            Returns
            -------
                core.TrajectoryDF.NumPandasTraj
                    The dataframe containing the resultant column if inplace is True.
                pandas.core.dataframe.DataFrame
                    The dataframe containing the resultant column if inplace is False.
        """
        chunks = []  # list for storing the smaller parts of the original dataframe.

        # Now, lets split the given dataframe into smaller pieces of 33000 rows each
        # so that we can run parallel tasks on each smaller piece.
        for i in range(0, len(dataframe), 33000):
            chunks.append(dataframe.reset_index().loc[i: i + 33000])

        # Now, lets create a pool of processes which contains processes equal to the number
        # of smaller chunks and then run them in parallel so that we can calculate
        # the distance for each smaller chunk and then merge all of them together.
        multi_pool = multiprocessing.Pool(len(chunks))
        result = multi_pool.map(helpers.consecutive_distance_helper, chunks)

        # Now lets, merge the smaller pieces and then return the dataframe based on the value
        # of the inplace parameter.
        final_result = pd.concat(result)

        # Also note  that if the metres parameter is true, then we have to return the answers in metres.
        # To convert to metres, we will just multiply the column by 1000.
        final_result['Distance_prev_to_curr'] = final_result['Distance_prev_to_curr'] * 1000 if metres \
            else final_result['Distance_prev_to_curr']

        if inplace:
            return NumPandasTraj(final_result, const.LAT, const.LONG, const.DateTime, const.TRAJECTORY_ID)
        else:
            return final_result.set_index([const.LAT, const.LONG], inplace=True)
