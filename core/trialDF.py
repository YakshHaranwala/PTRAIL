import pandas as pd
from typing import Dict, List, Union, Optional, Text, Any

from parser import ParserError
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
        if isinstance(data_set, dict):
            data_set = DataFrame.from_dict(data_set)

        # Case-2: The data is from a list.
        # Here, first check whether the data is in list form and if it is so, then convert into
        # pandas DataFrame first.
        # elif isinstance(data_set, list) or isinstance(data_set, np.ndarray):
        #     data_set = DataFrame(data_set, columns=column_list)

        # Case-3: The data is from a pandas DF.
        elif isinstance(data_set, DataFrame):
            data_set = DataFrame(data_set)

        column_names = self.get_default_column_names(datetime, traj_id, latitude, longitude)

        column_names = self.get_default_column_names(DateTime=datetime, traj_id=traj_id,
                                                     latitude=latitude, longitude=longitude)
        df2 = data_set.rename(columns=column_names)

        if self.column_check(df2):
            self.validate_data_types(df2)
            super(NumPandasTraj, self).__init__(df2)

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

    def column_check(self, data: DataFrame):
        cols = data.columns
        if const.TRAJECTORY_ID in cols and const.DateTime in cols and const.LAT and const.LONG:
            return True
        return False

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

    def rename(self, x, y):
        pass