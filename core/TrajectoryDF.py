from parser import ParserError
from typing import Dict, List, Union, Optional, Text

import numpy as np
import pandas as pd
import pandas.core.dtypes.common
from pandas import DataFrame
from pandas._libs import lib

import utilities.constants as const
from utilities.exceptions import *


class NumPandasTraj(DataFrame):
    def __init__(self, data_set: Union[DataFrame, List, Dict], latitude: Text, longitude: Text, datetime: Text,
                 traj_id: Text, rest_of_columns: Optional[List[Text]] = []):
        """
            Construct a trajectory dataframe. Note that the mandatory columns in the dataset are:
            Note that the below mentioned columns also need their headers to be provided.
                1. DateTime (will be converted to pandas DateTime format)
                2. Trajectory ID (will be converted to string format)
                3. Latitude (will be converted to float64 format)
                4. Longitude (will be converted to float64 format)

            rest_of_columns makes sure that if the data_set is a list, it has appropriate headers
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
                rest_of_columns: list[str]
                    A list containing headers of the columns other than the mandatory ones.
        """
        # Case-1: The data is from a dictionary.
        # Here, first check whether the data is in dictionary form and if it is so, then convert into
        # pandas DataFrame first.
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
            data_set = self.rename_df_col_headers(data_set, latitude, longitude, datetime, traj_id)

        # Now, renaming the default column names to library default column names.
        column_names = self.get_default_column_names(DateTime=datetime, traj_id=traj_id,
                                                     latitude=latitude, longitude=longitude)
        data_set.rename(columns=column_names, inplace=True)

        # Now checking whether all the columns are present in the data and then verifying the data types
        # of all the columns abd then calling the super() to create and return the dataframe.
        if self.validate_columns(data_set):
            self.validate_data_types(data_set)
            data_set.set_index([const.DateTime, const.TRAJECTORY_ID], inplace=True)
            super(NumPandasTraj, self).__init__(data_set)

    # ------------------------------ General Utilities ----------------------------- #
    def rename_df_col_headers(self, data: DataFrame, lat, lon, datetime, traj_id) -> DataFrame:
        """
            Change the column headers of the columns when the user given data is in the
            form of a pandas DF while creating the NumPandasTraj. This method is mainly
            used when the user reads in data from a csv because the CSV file might
            contain different names for the columns.

            Parameters
            ----------
                data: the dataframe whose column names are to be changed.

            Returns
            -------
                pandas.DataFrame
                    The pandas dataframe containing the library default column headers.
        """
        cols = data.columns.to_list()  # List of all column names
        # Iterate over the list of column names and check for keywords in the header to help renaming with appropriate
        # terms.
        for i in range(len(cols)):
            if lat in cols[i]:  # if 'lati' is in header change the header with latitude
                cols[i] = const.LAT
            if lon in cols[i]:  # if 'longi' is in header change the header with longitude
                cols[i] = const.LONG
            if datetime in cols[i]:
                cols[i] = const.DateTime
            if traj_id in cols[i]:
                cols[i] = const.TRAJECTORY_ID

        data = data.set_axis(cols, axis=1)
        return data

    def validate_data_types(self, data: DataFrame):
        """
            Check whether all the data given by the user is of valid type and if it isn't,
            it converts them to the specified data types.
                1. Trajectory_ID: Python: string -> pd.datetime
                2. LAT and LONG: Python: float -> float64
                3. DateTime: Python: string -> dask.datetime64[ns]

            Parameters
            ----------
                data: pd.DataFrame
                    This is the dataframe that contains the data that was passed in by the user.
                    The data is converted to pandas DF eventually in the import_data function anyway.
            Returns
            -------
                bool
                    A boolean value indicating whether all the data is of valid type or not.
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
            raise ParserError('datetime column cannot be parsed')
        except ValueError:
            raise ValueError('dtypes cannot be converted.')

    def validate_columns(self, data: DataFrame) -> bool:
        """
            Check whether all the columns are present in the DataFrame or not.

            Parameters
            ----------
                data
                    Data is the dask DataFrame containing all the data passed in by the user.

            Returns
            -------
                bool
                    Indicate whether or not all the columns that are mandatory are
                    present in the Data given by the user.
        """
        try:
            if np.isin(const.LAT, data.columns) and \
                    np.isin(const.LONG, data.columns) and \
                    np.isin(const.TRAJECTORY_ID, data.columns) and \
                    np.isin(const.DateTime, data.columns):
                return True
        except KeyError:
            raise MissingColumnsException("One of the columns are missing. Please check your data and try again.")

    def get_default_column_names(self, DateTime, traj_id, latitude, longitude) -> dict:
        """
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

    def set_index(self, keys, drop=True, append=False, inplace=False, verify_integrity=False):
        """
            Change the index of the DataFrame.
            !!!!                                                            !!!!
                WARNING:
                -------
                    CHANGING OF THE INDEX IS NOT ALLOWED IN DaskTrajectoryDF.
                    BY MANDATORY CONSTRAINTS, THE INDEX NEEDS TO BE DateTime.
            !!!!                                                            !!!!

            Raises
            ------
                NotAllowedError
                    The change of index is not allowed in DaskTrajectoryDF.

        """
        raise NotAllowedError("Changing of index is not allowed.\n"
                              "The index must be DateTime at all times.")

    def __reset_default_index(self):
        """
            Set the Index of the dataframe back to DateTime and traj_id.

            WARNING
            -------
                This must be used everytime after the reset_index is called
                in order to set the index back to library default values as
                it is necessary to perform various other functionalities.
        """
        try:
            self.set_index([const.DateTime, const.TRAJECTORY_ID], inplace=True)
        except KeyError:
            raise MissingColumnsException(f"Either of {const.DateTime} or {const.TRAJECTORY_ID} columns are missing.")

    # ------------------------------- Properties ---------------------------------- #
    @property
    def latitude(self):
        """
            Accessor method for the latitude column of the DaskTrajectoryDF.

            Returns
            -------
                dask.dataframe.core.Series
                    The Series containing all the latitude values from the DataFrame.

            Raises
            ------
                KeyError/ IndexError
                    Latitude column is missing from the data.
        """
        try:
            return self[const.LAT]
        except KeyError or IndexError:
            raise MissingColumnsException("The Latitude column is not present in the DataFrame, please verify again.")

    @property
    def longitude(self):
        """
            Accessor method for the longitude column of the DaskTrajectoryDF.

            Returns
            -------
                dask.dataframe.core.Series
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
            Accessor method for the DateTime column of the DaskTrajectoryDF.

            Returns
            -------
                dask.dataframe.core.Series
                    The Series containing all the DateTime values from the DataFrame.

            Raises
            ------
                KeyError/ValueError
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
                dask.dataframe.core.Series
                    The Series containing all the Trajectory_ID values from the DataFrame.

            Raises
            ------
                KeyError/ValueError
                    traj_id column is missing from the data.
        """
        try:
            return self.index.get_level_values(const.TRAJECTORY_ID).to_series()
        except KeyError or IndexError:
            raise MissingColumnsException("The Trajectory_ID column is not present in the DataFrame, please verify "
                                          "again.")

    # ------------------------------- File and DF Operations ----------------------------- #
    # @classmethod
    # def read_csv(cls, filename):
    #     """
    #         Read the data from a csv file and then store the data in a DaskTrajectoryDF.
    #         It is to be noted that the csv file provided must have the 4 mandatory columns
    #         which are:
    #             1. Latitude
    #             2. Longitude
    #             3. DateTime
    #             4. traj_id
    #
    #         WARNING:
    #         -------
    #             Only use this function when the dataset meets the following conditions:
    #                 1. Latitude is of the float format and does not contain directions like N, S.
    #                    if it does, please first convert it to float direction with + and - signs.
    #                 2. Longitude is of the float format and does not contain directions like N, S.
    #                    if it does, please first convert it to float direction with + and - signs.
    #                 3. DateTime are combined together.
    #                 4. traj_id is present.
    #
    #             The above restrictions are in place because the library indexes the trajectory
    #             by DateTime and Traj_ID. As a result, it needs to have the following 2 columns
    #             proper.
    #
    #         Parameters
    #         ----------
    #             filename: The name of the csv file (provide full path).
    #
    #         Raises
    #         ------
    #             FileNotFoundError
    #                 The file requested could not be found.
    #     """
    #     try:
    #         dataframe = pandas.read_csv(filename, index_col=False)
    #         return cls(data_set=dataframe, latitude=const.LAT, longitude=const.LONG,
    #                              datetime=const.DateTime, traj_id=const.TRAJECTORY_ID)
    #     except FileNotFoundError:
    #         raise FileNotFoundError(f"Could not open the %s, please try again." % str(filename))

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
