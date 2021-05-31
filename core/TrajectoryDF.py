from parser import ParserError

import pandas as pd
from typing import Dict, List, Union, Optional, Text, Any

import numpy as np
import pandas.core.dtypes.common
from pandas import DataFrame, read_csv
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
            print("Finished this")

        # Case-3: The data is from a pandas DF.
        # Here, all we have to do is to rename the column names from the data to default names.
        elif isinstance(data_set, DataFrame):
            data_set = self.rename_df_col_headers(data_set)
            print(data_set.head())

        # Now, renaming the default column names to library default column names.
        column_names = self.get_default_column_names(DateTime=datetime, traj_id=traj_id,
                                                     latitude=latitude, longitude=longitude)
        data_set.rename(columns=column_names, inplace=True)

        # Now checking whether all the columns are present in the data and then verifying the data types
        # of all the columns abd then calling the super() to create and return the dataframe.
        #if self.validate_columns(data_set):
        #self.validate_data_types(data_set)
        data_set.set_index([const.DateTime, const.TRAJECTORY_ID], inplace=True, drop=True)
        super(NumPandasTraj, self).__init__(data_set)

    def rename_df_col_headers(self, data: DataFrame):
        cols = data.columns.to_list()
        print(cols)
        for i in range(len(cols)):
            if 'lat' in cols[i].lower().strip():
                cols[i] = const.LAT
            if 'long' in cols[i].lower().strip():
                cols[i] = const.LONG
            if 'datetime' in cols[i].lower().strip():
                cols[i] = const.DateTime
            if '%id' in cols[i].lower().strip():
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
            if data.dtypes[const.TRAJECTORY_ID] == 'float64':
                data[const.TRAJECTORY_ID] = data[const.TRAJECTORY_ID].astype('float64')

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
                    The change of index is not alloed in DaskTrajectoryDF.

        """
        raise NotAllowedError("Changing of index is not allowed.\n"
                              "The index must be DateTime at all times.")

    def reset_index(self, drop=False):
        """
            Reset the index of the DataFrame i.e. remove the index.
            !!!!                                                            !!!!
                WARNING:
                -------
                    REMOVAL OF THE INDEX IS NOT ALLOWED IN DaskTrajectoryDF.
                    BY MANDATORY CONSTRAINTS, THE INDEX NEEDS TO BE DateTime.
            !!!!                                                            !!!!

            Raises
            ------
                NotAllowedError
                    The change of index is not alloed in DaskTrajectoryDF.

        """
        raise NotAllowedError("Resetting of index is not allowed.\n"
                              "The index must be DateTime at all times.")

    # ------------------------------- File and DF Operations ----------------------------- #
    @classmethod
    def read_csv(cls, filename):
        """
            Read the data from a csv file and then store the data in a DaskTrajectoryDF.
            It is to be noted that the csv file provided must have the 4 mandatory columns
            which are:
                1. Latitude
                2. Longitude
                3. DateTime
                4. traj_id

            Parameters
            ----------
                filename: The name of the csv file (provide full path).

            Raises
            ------
                FileNotFoundError
                    The file requested could not be found.
        """
        try:
            dataframe = read_csv(filename)
            return cls(data_set=dataframe, latitude=const.LAT, longitude=const.LONG,
                                 datetime=const.DateTime, traj_id=const.TRAJECTORY_ID)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not open the %s, please try again." % str(filename))
