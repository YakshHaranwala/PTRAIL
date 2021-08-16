"""
    The TrajectoryDF module is the main module containing the NumPandasTraj Dataframe
    for storing the Trajectory Data with PTRAIL Library. The Dataframe has
    certain restrictions on what type of data is mandatory in order to be stored as a
    NumPandasTraj which is mentioned in the documentation of the constructor.

    | Authors: Yaksh J Haranwala, Salman Haidri
    | Date: May 26, 2021.
    | Version: 0.2 Beta

"""
import random
from parser import ParserError
from typing import Dict, List, Union, Optional, Text

import folium
import numpy as np
import pandas as pd
import pandas.core.dtypes.common
from pandas import DataFrame
from pandas._libs import lib

from ptrail.utilities import constants as const
from ptrail.utilities.exceptions import *


class NumPandasTraj(DataFrame):
    def __init__(self, data_set: Union[DataFrame, List, Dict], latitude: Text, longitude: Text, datetime: Text,
                 traj_id: Text, rest_of_columns: Optional[List[Text]] = None):
        """
            Construct a trajectory dataframe to store and represent the Trajectory Data.

            Note
            ----
                | The mandatory columns in the dataset are:
                |    1. DateTime
                |    2. Trajectory ID
                |    3. Latitude
                |    4. Longitude

                | ``rest_of_columns`` makes sure that if the data_set is a list, it has appropriate headers
                  that the user wants instead of the default numerical values.

            Parameters
            ----------
                data_set: List, Dictionary or pandas DF.
                    The data provided by the user that needs to be represented and stored.
                datetime: str
                    The header of the datetime column.
                traj_id: str
                    The header of the Trajectory ID column.
                latitude: str
                    The header of the latitude column.
                longitude: str
                    The header of the longitude column.
                rest_of_columns: Optional[list[Text]]
                    A list containing headers of the columns other than the mandatory ones.
        """
        # Case-1: The data is from a dictionary.
        # Here, first check whether the data is in dictionary form and if it is so, then convert into
        # pandas DataFrame first.
        rest_of_columns = [] if rest_of_columns is None else rest_of_columns
        column_list = [latitude, longitude, datetime, traj_id] + rest_of_columns
        if isinstance(data_set, dict):
            data_set = DataFrame.from_dict(data_set)
            data_set = data_set.rename(columns=dict(zip(data_set.columns, [const.LAT, const.LONG,
                                                                           const.DateTime, const.TRAJECTORY_ID])))

        # Case-2: The data is from a list.
        # Here, first check whether the data is in list form and if it is so, then convert into
        # pandas DataFrame first.
        elif isinstance(data_set, list) or isinstance(data_set, np.ndarray):
            data_set = DataFrame(data_set, columns=column_list)

        # Case-3: The data is from a pandas DF.
        # Here, all we have to do is to rename the column names from the data to default names.
        elif isinstance(data_set, DataFrame):
            data_set = self._rename_df_col_headers(data_set, latitude, longitude, datetime, traj_id)

        # Now, renaming the default column names to library default column names.
        column_names = self._get_default_column_names(DateTime=datetime, traj_id=traj_id,
                                                      latitude=latitude, longitude=longitude)
        data_set.rename(columns=column_names, inplace=True)

        # Now checking whether all the columns are present in the data and then verifying the data types
        # of all the columns abd then calling the super() to create and return the dataframe.
        if self._validate_columns(data_set):
            self._validate_data_types(data_set)
            data_set.set_index([const.TRAJECTORY_ID, const.DateTime], inplace=True)
            data_set.sort_values([const.TRAJECTORY_ID, const.DateTime], inplace=True)
            super(NumPandasTraj, self).__init__(data_set)

    # ------------------------------ General Utilities ----------------------------- #
    def _rename_df_col_headers(self, data: DataFrame, lat: Text, lon: Text,
                               datetime: Text, traj_id: Text):
        """
            Change the column headers of the columns when the user given data is in the
            form of a pandas DF while creating the NumPandasTraj. This method is mainly
            used when the user reads in data from a csv because the CSV file might
            contain different names for the columns.

            Parameters
            ----------
                data: DataFrame
                    The dataframe whose column names are to be changed.
                lat: Text
                    The header of the Latitude column.
                lon: Text
                    The header of the Longitude column.
                datetime: Text
                    The header of the DateTime column.
                traj_id: Text
                    The header of the Trajectory ID column.

            Returns
            -------
                pandas.DataFrame
                    The pandas dataframe containing the library default column headers.
        """
        cols = data.columns.to_list()  # List of all column names
        # Iterate over the list of column names and check for keywords in the header
        # to help renaming with appropriate terms.
        for i in range(len(cols)):
            if lat in cols[i]:
                cols[i] = const.LAT
            if lon in cols[i]:
                cols[i] = const.LONG
            if datetime in cols[i]:
                cols[i] = const.DateTime
            if traj_id in cols[i]:
                cols[i] = const.TRAJECTORY_ID

        data = data.set_axis(cols, axis=1)  # Change all the column names to modified values.
        return data

    def _validate_data_types(self, data: DataFrame):
        """
            Check whether all the data given by the user is of valid type and if it isn't,
            it converts them to the specified data types.
                1. Trajectory_ID: Any -> str
                2. LAT: Any -> float64
                3. LONG: Any -> float64
                4. DateTime: Any -> dask.datetime64[ns]

            Parameters
            ----------
                data: pd.DataFrame
                    This is the dataframe that contains the data that was passed in by the user.
                    The data is converted to pandas DF eventually in the import_data function anyway.

            Raises
            ------
                KeyError:
                    Dataframe has one of the mandatory columns missing.
                ParserError:
                    The DateTime format provided is invalid and cannot be parsed as pandas DateTime.
                ValueError:
                    One of the data-types cannot be converted.
        """
        try:
            if data.dtypes[const.LAT] != 'float64':
                data[const.LAT] = data[const.LAT].astype('float64')
            if data.dtypes[const.LONG] != 'float64':
                data[const.LONG] = data[const.LONG].astype('float64')
            if data.dtypes[const.DateTime] != 'datetime64[ns]':
                data[const.DateTime] = data[const.DateTime].astype('datetime64[ns]')
            if data.dtypes[const.TRAJECTORY_ID] != 'str':
                data[const.TRAJECTORY_ID] = data[const.TRAJECTORY_ID].astype('str')
        except KeyError:
            raise KeyError('dataframe missing one of lat, lon, datetime columns.')
        except ParserError:
            raise ParserError('DateTime column cannot be parsed')
        except ValueError:
            raise ValueError('dtypes cannot be converted.')

    def _validate_columns(self, data: DataFrame) -> bool:
        """
            Check whether all the mandatory columns are present in the DataFrame or not.

            Parameters
            ----------
                data
                    The DataFrame containing all the data passed in by the user.

            Returns
            -------
                bool
                    Indicate whether or not all the columns that are mandatory are
                    present in the Data given by the user.

            Raises
            ------
                MissingColumnsException
                    One or more of the mandatory columns (Latitude, Longitude, DateTime, Traj_ID)
                    are missing in the data.
        """
        try:
            if np.isin(const.LAT, data.columns) and \
                    np.isin(const.LONG, data.columns) and \
                    np.isin(const.TRAJECTORY_ID, data.columns) and \
                    np.isin(const.DateTime, data.columns):
                return True
        except KeyError:
            raise MissingColumnsException("One of the columns are missing. Please check your data and try again.")

    def _get_default_column_names(self, DateTime, traj_id, latitude, longitude) -> dict:
        """
            Get a dictionary containing the key, value pairs of the library default
            column names for the following columns:
                1. Latitude
                2. Longitude
                3. DateTime
                4. Trajectory ID

            Parameters
            ----------
                DateTime: Text
                    The datetime header of the column already given by the user.
                traj_id: Text
                    The traj_id header of the column already given by the user.
                latitude:
                    The latitude header of the column already given by the user.
                longitude:
                    The longitude header of the column already given by the user.

            Returns
            -------
                dict
                    A dictionary of mappings from the given header by the user to the
                    library default headers.
        """
        return {
            DateTime: const.DateTime,
            traj_id: const.TRAJECTORY_ID,
            latitude: const.LAT,
            longitude: const.LONG
        }

    def set_default_index(self):
        """
            Set the Index of the dataframe back to traj_id and DateTime.

            Raises
            ------
                MissingColumnsException
                    DateTime/traj_id column is missing from the dataset.
        """
        try:
            self.set_index([const.DateTime, const.TRAJECTORY_ID], inplace=True)
        except KeyError:
            raise MissingColumnsException(f"Either of {const.DateTime} or {const.TRAJECTORY_ID} columns are missing.")

    # ------------------------------- Properties ---------------------------------- #
    @property
    def latitude(self):
        """
            Accessor method for the latitude column of the NumPandasTraj DataFrame.

            Returns
            -------
                pandas.core.series.Series
                    The Series containing all the latitude values from the DataFrame.

            Raises
            ------
                MissingColumnsException
                    Latitude column is missing from the data.
        """
        try:
            return self[const.LAT]
        except KeyError or IndexError:
            raise MissingColumnsException("The Latitude column is not present in the DataFrame, please verify again.")

    @property
    def longitude(self):
        """
            Accessor method for the longitude column of the NumPandasTraj DataFrame.

            Returns
            -------
                pandas.core.series.Series
                    The Series containing all the longitude values from the DataFrame.

            Raises
            ------
                MissingColumnsException
                    Longitude column is missing from the data
        """
        try:
            return self[const.LONG]
        except KeyError or IndexError:
            raise MissingColumnsException("The Longitude column is not present in the DataFrame, please verify again.")

    @property
    def datetime(self):
        """
            Accessor method for the DateTime column of the NumPandasTraj DataFrame.

            Returns
            -------
                pandas.core.series.Series
                    The Series containing all the DateTime values from the DataFrame.

            Raises
            ------
                MissingColumnsException
                    DateTime column is missing from the data.
        """
        try:
            return self.index.get_level_values(const.DateTime).to_series()
        except KeyError or IndexError:
            raise MissingColumnsException("The DateTime column is not present in the DataFrame, please verify again.")

    @property
    def traj_id(self):
        """
            Accessor method for the Trajectory_ID column of the DaskTrajectoryDF.

            Returns
            -------
                pandas.core.series.Series
                    The Series containing all the Trajectory_ID values from the DataFrame.

            Raises
            ------
                MissingColumnsException
                    traj_id column is missing from the data.
        """
        try:
            return self.index.get_level_values(const.TRAJECTORY_ID).to_series()
        except KeyError or IndexError:
            raise MissingColumnsException("The Trajectory_ID column is not present in the DataFrame, please verify "
                                          "again.")

    def __str__(self):
        return f"------------------------ Dataset Facts ------------------------------\n\n" \
               f"Number of unique Trajectories in the data: {self.traj_id.nunique()}\n" \
               f"Number of points in the data: {len(self)}\n" \
               f"Dataset time range: {self.datetime.max() - self.datetime.min()}\n" \
               f"Datatype of the DataFrame: {type(self)}\n" \
               f"Dataset Bounding Box:" \
               f" {(self.latitude.min(), self.longitude.min(), self.latitude.max(), self.longitude.max())}\n\n" \
               f"---------------------------------------------------------------------"

    # ------------------------------- File and DF Operations ----------------------------- #

    def to_numpy(self, dtype=None, copy: bool = False, na_value=lib.no_default) -> np.ndarray:
        """
            Convert the DataFrame to a NumPy array.By default, the dtype of the returned array will
            be the common dtype of all types in the DataFrame. For example, if the dtypes are float16
            and float32, the results dtype will be float32. This may require copying data and coercing
            values, which may be expensive

            Parameters
            ----------
                dtype:
                    The dtype to pass to :meth:`numpy.asarray`.
                copy:
                    Whether to ensure that the returned value is not a view on another array.
                    Note that ``copy=False`` does not *ensure* that ``to_numpy()`` is no-copy.
                    Rather, ``copy=True`` ensure that a copy is made, even if not strictly necessary.
                na_value:
                    The value to use for missing values. The default value depends on `dtype` and the
                    dtypes of the DataFrame columns.
        """
        return self.reset_index(drop=False).to_numpy()

    def sort_by_traj_id_and_datetime(self, ascending=True):
        """
            Sort the trajectory in Ascending or descending order based on the following 2
            columns in order:
                1. Trajectory ID
                2. DateTime

            Parameters
            ----------
                ascending: bool
                    Whether to sort the values in ascending order or descending order.

            Returns
            -------
                NumPandasTraj
                    The sorted dataframe.
        """
        return self.sort_values([const.TRAJECTORY_ID, const.DateTime], ascending=ascending)

    # ------------------------------------- Visualization ----------------------------------- #
    def plot_folium_traj(self, weight: float = 3, opacity: float = 0.8):
        """
            Use folium to plot the trajectory on a map.

            Parameters
            ----------
                weight: float
                    The weight of the trajectory line on the map.
                opacity: float
                    The opacity of the trajectory line on the map.

            Returns
            -------
                folium.folium.Map
                    The map with plotted trajectory.
        """
        sw = self[['lat', 'lon']].min().values.tolist()
        ne = self[['lat', 'lon']].max().values.tolist()

        # Create a map with the initial point.
        map_ = folium.Map(location=(self.latitude[0], self.longitude[0]))

        ids_ = list(self.traj_id.value_counts().keys())
        colors = ["#" + ''.join([random.choice('123456789BCDEF') for j in range(6)])
                  for i in range(len(ids_))]

        for i in range(len(ids_)):
            # First, filter out the smaller dataframe.
            small_df = self.reset_index().loc[self.reset_index()[const.TRAJECTORY_ID] == ids_[i],
                                              [const.LAT, const.LONG]]

            # Then, create (lat, lon) pairs for the data points.
            locations = []
            for j in range(len(small_df)):
                locations.append((small_df['lat'].iloc[j], small_df['lon'].iloc[j]))

            # Create popup text for the

            # Create start and end markers for the trajectory.
            folium.Marker([small_df['lat'].iloc[0], small_df['lon'].iloc[0]],
                          color='green',
                          popup=f'Trajectory ID: {ids_[i]} \n'
                                f'Latitude: {locations[0][0]} \n'
                                f'Longitude: {locations[0][1]}',
                          marker_color='green',
                          icon=folium.Icon(icon_color='green', icon=None)).add_to(map_)

            folium.Marker([small_df['lat'].iloc[-1], small_df['lon'].iloc[-1]],
                          color='green',
                          popup=f'Trajectory ID: {ids_[i]} \n'
                                f'Latitude: {locations[-1][0]} \n'
                                f'Longitude: {locations[-1][1]}',
                          marker_color='red',
                          icon=folium.Icon(icon_color='red', icon=None)).add_to(map_)

            # Add trajectory to map.
            folium.PolyLine(locations,
                            color=colors[i],
                            weight=weight,
                            opacity=opacity).add_to(map_)

        map_.fit_bounds([sw, ne])
        return map_
