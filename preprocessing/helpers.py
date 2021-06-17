"""
    The helpers class has the functionalities that interpolate a point based
    on the given data by the user. The class contains the following 4
    interpolation calculators:
        1. Linear Interpolation
        2. Cubic Interpolation
        3. Random-Walk Interpolation
        4. Kinematic Interpolation

    Besides the interpolation helpers, there are also general utilities which
    are used in splitting up dataframes for running the code in parallel.

    @Authors: Yaksh J Haranwala, Salman Haidri
    @Version: 1.0
    @Date: 16th June, 2021
"""
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
import os
import numpy as np
import psutil
import utilities.constants as const


class InterpolationHelpers:
    # ------------------------------------ Interpolation calculators ------------------------- #
    @staticmethod
    def _linear_helper(dataframe):
        """
            Use the linear interpolation formula and calculate the interpolated
            position and time of a point, in a given 3 point dataframe chunk.

            Parameters
            ----------
                dataframe: NumPandsTraj/pandas.core.dataframe.DataFrame
                    The dataframe containing 3 points that are to be interpolated.

            Returns
            -------
                dict:
                    The interpolated latitude,longitude and time of the missing point.

            References
            ----------
                "Etemad, M., Soares, A., Etemad, E. et al. SWS: an unsupervised trajectory
                segmentation algorithm based on change detection with interpolation kernels.
                Geoinformatica (2020)"
        """
        mid = int(len(dataframe) / 2)  # The middle point of the dataframe.

        # Extract the latitude, longitude and time values of the points.
        lat = dataframe.lat.values
        lon = dataframe.lon.values
        time = dataframe[const.DateTime].values

        # Interpolate the latitude, longitude.
        interpolated_x, interpolated_y = (lat[mid - 1] + lat[mid + 1]) / 2, (lon[mid - 1] + lon[mid + 1]) / 2

        # Here, this equation is used since the pandas datetime format does not
        # support addition.
        interpolated_time_diff = (time[mid + 1] - time[mid - 1]) / 2
        interpolated_time = time[mid - 1] + interpolated_time_diff

        # Return a dictionary containing interpolated latitude, longitude and datetime.
        return {'inter_x': interpolated_x,
                'inter_y': interpolated_y,
                'inter_time': interpolated_time
                }

    @staticmethod
    def _cubic_helper(dataframe):
        """
            Use the cubic interpolation formula and calculate the interpolated
            position and time of a point, in a given 3 point dataframe chunk.

            Parameters
            ----------
                dataframe: NumPandsTraj/pandas.core.dataframe.DataFrame
                    The dataframe containing 3 points that are to be interpolated.

            Returns
            -------
                dict:
                    The interpolated latitude,longitude and time of the missing point.

            References
            ----------
                "Etemad, M., Soares, A., Etemad, E. et al. SWS: an unsupervised trajectory
                segmentation algorithm based on change detection with interpolation kernels.
                Geoinformatica (2020)"
        """
        # Set the index as just DateTime because it is being used
        dataframe = dataframe.reset_index().set_index([const.DateTime]).sort_values(['DateTime'])
        lat = dataframe.lat.values
        lon = dataframe.lon.values
        datetime = dataframe.reset_index()[const.DateTime].values
        mid = int(len(lon) / 2)

        x3 = lat[mid]
        y3 = lon[mid]
        lat = np.delete(lat, mid)
        lon = np.delete(lon, mid)
        t = np.diff(dataframe.index) / 1000000000
        t3 = t[mid]
        t = np.delete(t, mid)
        t = np.cumsum(t)
        t = np.insert(t, 0, 0).astype(float)

        latcs = CubicSpline(np.abs(t), lat)
        new_x = latcs(t3)

        loncs = CubicSpline(np.abs(t), lon)
        new_y = loncs(t3)

        pf = (new_x, new_y)

        # reverse
        lat = dataframe.lat.values[::-1]
        lon = dataframe.lon.values[::-1]
        tidx = dataframe.index[::-1]

        lat = np.delete(lat, mid)
        lon = np.delete(lon, mid)
        t = np.diff(tidx)
        t3 = t[mid]
        t = np.delete(t, mid)
        t = np.cumsum(t)
        t = np.insert(t, 0, 0).astype(float)

        latcs = CubicSpline(np.abs(t), lat)
        new_x = latcs(t3)

        loncs = CubicSpline(np.abs(t), lon)
        new_y = loncs(t3)

        new_time = datetime[mid] - ((t3/2))

        pb = (new_x, new_y)
        pc = ((pf[0] + pb[0]) / 2, (pf[1] + pb[1]) / 2)


        return {'inter_x': pc[0],
                'inter_y': pc[1],
                'inter_time': new_time
                }

        # dataframe.loc[pd.to_datetime(new_time)] = [dataframe[const.TRAJECTORY_ID].iloc[0], pc[0], pc[1], 0]
        # return dataframe

    @staticmethod
    def cubic(dataframe):
        lat = dataframe.lat.values
        lon = dataframe.lon.values
        datetime = dataframe.reset_index()[const.DateTime].values
        mid = int(len(lon) / 2)

        x3 = lat[3]
        y3 = lon[3]
        lat = np.delete(lat, mid)
        lon = np.delete(lon, mid)
        t = np.diff(dataframe.index) / 1000000000
        t3 = t[mid]
        t = np.delete(t, mid)
        t = np.cumsum(t)
        t = np.insert(t, 0, 0).astype(float)
        fx = interp1d(t, lat, kind='cubic', fill_value='extrapolate')
        fy = interp1d(t, lon, kind='cubic', fill_value='extrapolate')
        new_x = fx(t3)
        new_y = fy(t3)

        pc = (new_x, new_y)
        new_time = datetime[mid] - ((t3 / 2)*10e9)
        return {'inter_x': pc[0],
                'inter_y': pc[1],
                'inter_time': new_time
                }

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
        available_cpus = len(os.sched_getaffinity(0)) if os.name == 'posix' \
            else psutil.cpu_count()  # Number of available CPUs.

        # Integer divide the total number of Trajectory IDs by the number of available CPUs
        # and square the number because if too many partitions are made, then it does more
        # harm than good for the execution speed. The factor of 1 is added to avoid errors
        # when the integer division yields a 0.
        factor = (size // available_cpus) + 1

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
                dataframe: NumPandasTraj
                    The dataframe that is to be split.

            Returns
            -------
                list:
                    The list containing smaller dataframe chunks.
        """
        # First, create a list containing all the ids of the data and then further divide that
        # list items and split it into sub-lists of ids equal to split_factor.
        ids_ = list(dataframe.traj_id.value_counts().keys())

        # Get the ideal number of IDs by which the dataframe is to be split.
        split_factor = InterpolationHelpers._get_partition_size(len(ids_))
        ids_ = [ids_[i: i + split_factor] for i in range(0, len(ids_), split_factor)]

        # Now split the dataframes based on set of Trajectory ids.
        # As of now, each smaller chunk is supposed to have data of 100
        # trajectory IDs max
        df_chunks = [dataframe.loc[dataframe[const.TRAJECTORY_ID].isin(ids_[i])]
                     for i in range(len(ids_))]
        return df_chunks
