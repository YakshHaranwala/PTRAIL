"""
    The interpolation module contains several interpolation techniques
    for the trajectory data. Interpolation techniques is used to
    smoothen the otherwise incomplete or rough trajectory data.

    @Authors: Yaksh J Haranwala, Salman Haidri
    @Date: 10th June, 2021
    @Version 1.0
"""
from utilities.exceptions import *
from typing import Optional, Text
from core.TrajectoryDF import NumPandasTraj as NumTrajDF
from features.spatial_features import SpatialFeatures as spatial
from preprocessing.helpers import InterpolationHelpers as ip_help

import multiprocessing
import pandas as pd
import itertools
import utilities.constants as const

pd.set_option('mode.chained_assignment', None)


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
                    points and sorted according to traj_id and Datetime.

            Raises
            ------
                KeyError:
                    Distance_prev_to_curr column is not present in the dataframe
        """
        try:

            # First, split the dataframe based on the number of IDs and store all the
            # smaller chunks into a list.
            # dataframe[const.PREV_DIST] = dataframe[const.PREV_DIST].fillna(0)
            dataframe = dataframe.reset_index()
            df_chunks = ip_help._df_split_helper(dataframe)

            # Now, create a multiprocessing pool and run all the processes in
            # parallel to interpolate the dataframes.
            mp_pool = multiprocessing.Pool(len(df_chunks))
            print(len(df_chunks))
            results = mp_pool.starmap(Interpolate.cubic_interpolation,
                                      (zip(df_chunks, itertools.repeat(distance_threshold))))

            # Now, convert the results DF into NumPandasTraj, calculate distance
            # between consecutive columns and return the dataframe.
            to_return = NumTrajDF(pd.concat(results), const.LAT, const.LONG, const.DateTime, const.TRAJECTORY_ID)
            return spatial.create_distance_between_consecutive_column(to_return).sort_values(by=[const.TRAJECTORY_ID,
                                                                                                 const.DateTime])
        except KeyError:
            raise MissingColumnsException("The column 'Distance_prev_to_curr' is missing in the dataset."
                                          "Please run the function create_distance_between_consecutive_column()"
                                          "function from the spatial_features module of the features package"
                                          "before running interpolation.")

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

            References
            ----------
                "Etemad, M., Soares, A., Etemad, E. et al. SWS: an unsupervised trajectory
                segmentation algorithm based on change detection with interpolation kernels.
                Geoinformatica (2020)"
        """
        df = dataframe.copy()
        df1 = df[[const.TRAJECTORY_ID, const.DateTime, const.LAT, const.LONG, const.PREV_DIST]]
        ids_ = df1[const.TRAJECTORY_ID].to_list()
        dists = dataframe[const.PREV_DIST].fillna(0).to_list()
        for i in range(len(df)):
            if dists[i] > distance_threshold:
                if (dists[i - 1] and dists[i - 2]) and (ids_[i - 2] == ids_[i - 1] == ids_[i]):
                    vals = ip_help._linear_helper(dataframe.iloc[i - 2: i + 1])
                elif ids_[i - 1] == ids_[i] == ids_[i + 1]:
                    vals = ip_help._linear_helper(dataframe.iloc[i - 1: i + 2])
                df1.loc[(2 * i - 1) / 2] = [dataframe[const.TRAJECTORY_ID].iloc[0], vals['inter_time'],
                                            vals['inter_x'], vals['inter_y'], 0]

        return df1.sort_index().reset_index(drop=True)

    @staticmethod
    def cubic_interpolation(dataframe, distance_threshold):
        """
            Use the Cubic Interpolation method to smoothen the trajectory
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

            References
            ----------
                "Etemad, M., Soares, A., Etemad, E. et al. SWS: an unsupervised trajectory
                segmentation algorithm based on change detection with interpolation kernels.
                Geoinformatica (2020)"
        """
        df = dataframe.copy()
        df1 = df[[const.TRAJECTORY_ID, const.DateTime, const.LAT, const.LONG, const.PREV_DIST]]
        ids_ = df1[const.TRAJECTORY_ID].to_list()
        dists = dataframe[const.PREV_DIST].fillna(0).to_list()
        for i in range(len(df)):
            if dists[i] > distance_threshold:
                if (dists[i - 1] and dists[i - 2]) and (ids_[i - 2] == ids_[i - 1] == ids_[i]):
                    vals = ip_help._cubic_helper(dataframe.iloc[i - 2: i + 1])

                elif ids_[i - 1] == ids_[i] == ids_[i + 1]:
                    vals = ip_help._cubic_helper(dataframe.iloc[i - 1: i + 2])

                df1.loc[(2 * i - 1) / 2] = [dataframe[const.TRAJECTORY_ID].iloc[0], vals['inter_time'],
                                            vals['inter_x'], vals['inter_y'], 0]

        return df1.sort_index().reset_index(drop=True)

    @staticmethod
    def cubic_alt(dataframe, distance_threshold):
        df = dataframe.reset_index().copy()
        ids_ = list(dataframe.traj_id.value_counts().keys())
        chunks = [df.loc[df[const.TRAJECTORY_ID] == ids_[i]] for i in range(len(ids_))]
        results = []
        for i in range(len(chunks)):
            results.append(Interpolate.cubic_inter(chunks[i], distance_threshold))

        return pd.concat(results).reset_index().set_index([const.TRAJECTORY_ID, const.DateTime]).sort_values([const.TRAJECTORY_ID, const.DateTime])
        #return NumTrajDF(pd.concat(results).reset_index(), const.LAT, const.LONG, const.DateTime, const.TRAJECTORY_ID)

    @staticmethod
    def cubic_inter(dataframe, distance_threshold):
        df = dataframe.copy().set_index([const.DateTime])
        idx = df[const.TRAJECTORY_ID].iloc[0]
        if len(dataframe) > 6:
            for i in range(len(dataframe)-7):
                seven_points = df.iloc[i:i+7, :]
                if Interpolate.check_threshold(seven_points, distance_threshold):
                    vals = ip_help.cubic(seven_points)
                    df.loc[pd.to_datetime(vals['inter_time'])] = [idx, vals['inter_x'], vals['inter_y'], 0]


        return df

    @staticmethod
    def check_threshold(df, threshold):
        dists = df['Distance_prev_to_curr'].to_list()
        flag = False
        for i in range(len(dists)):
            if dists[i] > threshold:
                flag = True

        return flag

    @staticmethod
    def calculate_error(df, f1=None, rang=(0, 0), window_size=7):
        start = rang[0]
        if rang[1] == 0:
            limit = df.shape[0]
        else:
            limit = rang[1]
        end = limit
        ln = int(window_size / 2)
        da = [0] * ln
        print(window_size, "dd", da)
        for ix in range(end - start - window_size):
            try:
                seven_points = df.iloc[start + ix:start + ix + window_size, :]
                lat = seven_points.lat.values
                lon = seven_points.lon.values
                p1, p2, pc, d = f1(seven_points)
            except:
                d = 0
            da.append(d)

        for i in range(ln + 1):
            da.append(0)
        print(window_size, "dd", da[-10:])
        return da

    @staticmethod
    def kinetic_interpolation(dataframe):
        pass

    @staticmethod
    def random_walk_interpolation(dataframe):
        pass
