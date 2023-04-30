"""
    The conversions modules contains various available methods
    that can be used to convert given data into another format.

    | Authors: Yaksh J Haranwala, Salman Haidri
"""
from typing import Text, Union
from ptrail.core.TrajectoryDF import PTRAILDataFrame

import pandas as pd
import numpy as np


class Conversions:
    @staticmethod
    def convert_directions_to_degree_lat_lon(data, latitude: Text, longitude: Text):
        """
            Convert the latitude and longitude format from degrees (NSEW)
            to float values. This is used for datasets like the Atlantic Hurricane dataset
            where the coordinates are not given as float values but are instead given as
            degrees.

            References
            ----------
                "Arina De Jesus Amador Monteiro Sanches. “Uma Arquitetura E Imple-menta ̧c ̃ao Do M ́odulo De
                Pr ́e-processamento Para Biblioteca Pymove”.Bachelor’s thesis. Universidade Federal Do Cear ́a, 2019"
        """

        def decimal_degree_to_decimal(col):
            if col[latitude][-1:] == 'N':
                col[latitude] = float(col[latitude][:-1])
            else:
                col[latitude] = float(col[latitude][:-1]) * -1

            if col[longitude][-1:] == 'E':
                col[longitude] = float(col[longitude][:-1])
            else:
                col[longitude] = float(col[longitude][:-1]) * -1 + 360 if float(col[longitude][:-1]) * -1 < -180 \
                    else float(col[longitude][:-1]) * -1
            return col

        return data.apply(decimal_degree_to_decimal, axis=1)

    @staticmethod
    def dict_to_pandas(dataset: dict):
        """
            Convert the dictionary dataset to pandas dataframe.

            Parameters
            ----------
                dataset: dictionary
                    The dictionary dataset that is to be converted to pandas dataframe.

            Returns
            -------
                pd.DataFrame
                    The dictionary dataset converted to pandas dataframe.
        """
        new_dataset = pd.DataFrame()
        ids = dataset.keys()
        for i in ids:
            curr_traj = pd.DataFrame.from_dict(dataset[i])
            new_dataset = pd.concat([new_dataset, curr_traj], axis=0)

        return new_dataset

    @staticmethod
    def pandas_to_dict(dataset: Union[pd.DataFrame, PTRAILDataFrame]):
        """
            Convert the given dataframe into a dictionary format that is used
            by the compression module.

            Parameters
            ----------
                dataset: Union[pd.DataFrame, PTRAILDataFrame]

            Returns
            -------
                dict:
                    The dataframe converted into a dictionary.
        """
        new_dataset = {}
        dataset = dataset.reset_index()
        ids = dataset['traj_id'].unique()

        for traj_id in ids:
            # Get a trajectory with the traj_id in iteration.
            traj = dataset[dataset['traj_id'] == traj_id]
            traj.set_index("traj_id")

            # Convert the trajectory into a dict.
            new_dataset[traj_id] = {}
            for col in traj.columns:
                new_dataset[traj_id][col] = np.array(traj[col])

        return new_dataset
