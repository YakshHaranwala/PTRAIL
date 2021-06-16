"""
    The interpolation module contains several interpolation techniques
    for the trajectory data. Interpolation techniques is used to
    smoothen the otherwise incomplete or rough trajectory data.

    @Authors: Yaksh J Haranwala, Salman Haidri
    @Date: 10th June, 2021
    @Version 1.0
"""
from typing import Optional, Text

import pandas as pd

from core.TrajectoryDF import NumPandasTraj as NumTrajDF
from utilities.DistanceCalculator import FormulaLog as calc
import utilities.constants as const
from features.spatial_features import SpatialFeatures as spatial
from features.helper_functions import Helpers as help
from utilities.exceptions import *

# TODO: So basically, the WS-II Repository is performing calculations and
#       returning a point that is interpolated on the basis of 7 rows of dataframes.
#       Now, the task for us is to figure out how we interpolate points for
#       trajectories with less than 7 points. As from what I can see, for Linear,
#       it is only taking average of coordinates of 2 points as per the formula.
#       So we need to take that approach as well to go forward with it.

# TODO: The approach that we are going to take is:
#           1. Instead of passing a fixed number of values, we will just pass a dataframe
#              with a certain number of trajectories or just a single trajectory.
#           2. The dataframe will be based on the value count of the trajectory, i.e., maybe
#              a trajectory just over x number of values.

# TODO: In terms of the algorithm, what I was thinking is this:
#       1. Pass in a dataframe along with a threshold of maximum distance
#          between 2 points of a particular trajectory.
#       2. We then check the consecutive distances between all the points
#          and determine if the distance between the 2 points is lesser
#          or greater than the user-given threshold.
#       3. If the distance is greater than the threshold, we take the certain
#          number of points and interpolate using a suitable method.
#       4. We keep repeating the step-3 until the distance between the older
#          points and the newly interpolated points does not come down below
#          or equal to the threshold.
#       5. We interpolate the time in the same way as distance and then append
#          those points to the dataset where they need to be.

class Interpolate:
    @staticmethod
    def interpolate_position(dataframe: NumTrajDF, distance_threshold: float,
                             ip_type: Optional[Text] = 'linear'):
        """
            Interpolate the position of an object when the distance jump
            is greater than a user-given threshold using the user-specified
            interpolation method.Currently, the library supports the following 4
            interpolation methods:
                1. Linear Interpolation
                2. Cubic Interpolation
                3. Random-Walk Interpolation
                4. Kinematic Interpolation

            WARNING: THE INTERPOLATION METHODS WILL ONLY RETURN THE 4 FUNDAMENTAL LIBRARY
                     COLUMNS AND A 'Distance_prev_to_curr' COLUMN BECAUSE IT IS NOT POSSIBLE
                     TO INTERPOLATE OTHER DATA THAT MIGHT BE PRESENT IN THE DATASET APART
                     FROM LATITUDE, LONGITUDE AND DateTime. AS A RESULT, OTHER COLUMNS ARE
                     DROPPED AND LEFT TO USER TO TAKE CARE OF THAT.

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The dataframe containing the entire dataset.
                distance_threshold: float
                    The maximum distance between 2 consecutive points.
                ip_type: Optional[Text], default = 'linear'
                    The type of interpolation to be used.

            Returns
            -------
                NumPandasTraj:
                    The dataframe containing enhanced trajectories with interpolated
                    points

            Raises
            ------
                KeyError:
                    Distance_prev_to_curr column is not present in the dataframe
        """
        try:
            # First, break down the originally given dataframe by the user
            # into smaller chunks based on the trajectory ID and store all the
            # smaller dataframes in a list.
            # dataframe = spatial.create_distance_between_consecutive_column(dataframe)
            dataframe[const.PREV_DIST] = dataframe[const.PREV_DIST].fillna(0)
            dataframe = dataframe.reset_index()
            ids_ = dataframe[const.TRAJECTORY_ID].value_counts(ascending=True).keys()
            chunks = [dataframe.loc[dataframe[const.TRAJECTORY_ID] == ids_[i]] for i in range(len(ids_))]

            # On all the separate dataframes, run the linear interpolation calculation
            # on all of the smaller dataframes containing only a smaller set of points.
            results = []
            for i in range(len(chunks)):
                results.append(Interpolate.linear_interpolate(chunks[i], distance_threshold))

            # Convert the resultant dataframe from the above calculations into a NumPandasTraj
            # and again perform the consecutive distance calculation in order to avoid false
            # reporting of the values.
            to_return = NumTrajDF(pd.concat(results), const.LAT, const.LONG, const.DateTime, const.TRAJECTORY_ID)
            return spatial.create_distance_between_consecutive_column(to_return)

        except KeyError:
            raise MissingColumnsException("The column 'Distance_prev_to_curr' is missing in the data."
                                          "Please run the function create_distance_between_consecutive() from"
                                          "the spatial features module first before running the interpolation.")


    @staticmethod
    def linear_interpolate(dataframe, distance_threshold):
        """
            Use the Linear Interpolation method to smoothen the trajectory
            when the distance between 2 consecutive points is greater than
            the user-specified value.

            Parameters
            ----------
                dataframe:
                    The dataframe consisting of the points of a single trajectory.
                distance_threshold: float
                    The maximum distance between 2 consecutive points of a trajectory.

            Returns
            -------
                pandas.core.dataframe.DataFrame:
                    The dataframe with enhanced trajectory points calculated by linear
                    interpolation method.
        """
        df = dataframe.copy()
        df1 = df[[const.TRAJECTORY_ID, const.DateTime, const.LAT, const.LONG, const.PREV_DIST]]
        dists = dataframe[const.PREV_DIST].to_list()
        for i in range(len(df)):
            if dists[i] > distance_threshold:
                if dists[i - 1] and dists[i - 2]:
                    vals = Interpolate.linear(dataframe.iloc[i - 2: i + 1])
                else:
                    vals = Interpolate.linear(dataframe.iloc[i: i + 3])

                # dataframe.loc[(2 * i - 1) / 2] = [dataframe[const.TRAJECTORY_ID].iloc[0], vals['inter_time'],
                #                                   vals['inter_x'], vals['inter_y'], 0]
                df1.loc[(2 * i - 1) / 2] = [dataframe[const.TRAJECTORY_ID].iloc[0], vals['inter_time'],
                                            vals['inter_x'], vals['inter_y'], 0]

        return df1.sort_index().reset_index(drop=True)

    @staticmethod
    def linear(dataframe):
        mid = int(len(dataframe) / 2)

        lat = dataframe.lat.values
        lon = dataframe.lon.values
        time = dataframe[const.DateTime].values

        interpolated_x, interpolated_y = (lat[mid - 1] + lat[mid + 1]) / 2, (lon[mid - 1] + lon[mid + 1]) / 2
        interpolated_time_diff = (time[mid + 1] - time[mid - 1]) / 2
        interpolated_time = time[mid - 1] + interpolated_time_diff

        return {'inter_x': interpolated_x,
                'inter_y': interpolated_y,
                'inter_time': interpolated_time
                }

    @staticmethod
    def cubic_interpolation(dataframe):
        pass

    @staticmethod
    def kinetic_interpolation(dataframe):
        pass

    @staticmethod
    def random_walk_interpolation(dataframe):
        pass
