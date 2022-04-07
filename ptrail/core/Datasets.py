"""
    The Datasets.py module is used to load built-in datasets to variables.
    All the datasets loaded are stored and returned in a PTRAILDataFrame
    Currently, the library has the following datasets available to use:

        | 1. Atlantic Hurricanes Dataset
        | 2. Traffic Dataset (a smaller subset)
        | 3. Geo-life Dataset (a smaller subset)
        | 4. Seagulls Dataset
        | 5. Ships Dataset (a smaller subset)
        | 6. Starkey Animals Dataset
        | 7. Starkey Habitat Dataset (accompanies the starkey dataset)

        Note
        ----
            The Starkey Habitat Dataset is not loaded is not loaded into a PTrailDataframe since
            it is not a movement dataset and rather contains contextual information about the starkey
            habitat. It is rather loaded into a pandas dataframe and returned as is.

    | Authors: Yaksh J Haranwala
"""
import pandas as pd
from ptrail.core.TrajectoryDF import PTRAILDataFrame


class Datasets:
    @staticmethod
    def load_hurricanes():
        """
            Load the Atlantic Hurricane dataset into the PTRAILDataFrame
            and return it.

            Returns
            -------
                PTRAILDataFrame:
                    The atlantic hurricanes dataset loaded into a PTrailDataFrame.
        """
        # read the CSV file from the repository using pandas.
        df = pd.read_csv('https://raw.githubusercontent.com/YakshHaranwala/PTRAIL/main/examples/data'
                         '/atlantic_hurricanes.csv')

        # Load the dataset into a PTrailDataFrame and print the dataframe
        # information.
        to_return = PTRAILDataFrame(data_set=df,
                                    latitude='lat',
                                    longitude='long',
                                    datetime='DateTime',
                                    traj_id='traj_id',
                                    rest_of_columns=[])
        print(to_return)

        # return the PTRAILDataFrame
        return to_return

    @staticmethod
    def load_traffic_data():
        """
            Load the Traffic dataset into the PTRAILDataFrame and return it.

            Returns
            -------
                PTRAILDataFrame:
                    The traffic dataset loaded into a PTrailDataFrame.
        """
        # read the CSV file from the repository using pandas.
        df = pd.read_csv('https://raw.githubusercontent.com/YakshHaranwala/PTRAIL/main/examples/data/car_traffic.csv')

        # Load the dataset into a PTrailDataFrame and print the dataframe
        # information.
        to_return = PTRAILDataFrame(data_set=df,
                                    latitude='lat',
                                    longitude='long',
                                    datetime='DateTime',
                                    traj_id='traj_id',
                                    rest_of_columns=[])
        print(to_return)

        # return the PTRAILDataFrame
        return to_return

    @staticmethod
    def load_geo_life_sample():
        """
            Load the Geo-Life Sample dataset into the PTRAILDataFrame and return it.

            Returns
            -------
                PTRAILDataFrame:
                    The geo-life sample dataset loaded into a PTrailDataFrame.
        """
        # read the CSV file from the repository using pandas.
        df = pd.read_csv(
            'https://raw.githubusercontent.com/YakshHaranwala/PTRAIL/main/examples/data/geolife_sample.csv'
        )

        # Load the dataset into a PTrailDataFrame and print the dataframe
        # information.
        to_return = PTRAILDataFrame(data_set=df,
                                    latitude='lat',
                                    longitude='long',
                                    datetime='datetime',
                                    traj_id='id',
                                    rest_of_columns=[])
        print(to_return)

        # return the PTRAILDataFrame
        return to_return

    @staticmethod
    def load_seagulls():
        """
            Load the Sea-Gulls dataset into the PTRAILDataFrame and return it.

            Returns
            -------
                PTRAILDataFrame:
                    The seagulls dataset loaded into a PTrailDataFrame.
        """
        # read the CSV file from the repository using pandas.
        df = pd.read_csv(
            'https://raw.githubusercontent.com/YakshHaranwala/PTRAIL/main/examples/data/seagulls.csv'
        )

        # Load the dataset into a PTrailDataFrame and print the dataframe
        # information.
        to_return = PTRAILDataFrame(data_set=df,
                                    latitude='lat',
                                    longitude='long',
                                    datetime='DateTime',
                                    traj_id='traj_id',
                                    rest_of_columns=[])
        print(to_return)

        # return the PTRAILDataFrame
        return to_return

    @staticmethod
    def load_ships():
        """
            Load the Sea-Gulls dataset into the PTRAILDataFrame and return it.

            Returns
            -------
                PTRAILDataFrame:
                    The Ships dataset loaded into a PTrailDataFrame.
        """
        # read the CSV file from the repository using pandas.
        df = pd.read_csv(
            'https://raw.githubusercontent.com/YakshHaranwala/PTRAIL/main/examples/data/ships.csv'
        )

        # Load the dataset into a PTrailDataFrame and print the dataframe
        # information.
        to_return = PTRAILDataFrame(data_set=df.dropna(),
                                    latitude='Lat',
                                    longitude='Lon',
                                    datetime='DateTime',
                                    traj_id='VesselName')
        print(to_return)

        # return the PTRAILDataFrame
        return to_return

    @staticmethod
    def load_starkey():
        """
            Load the Starkey dataset into the PTRAILDataFrame and return it.

            Returns
            -------
                PTRAILDataFrame:
                    The Starkey dataset loaded into a PTrailDataFrame.
        """
        # read the CSV file from the repository using pandas.
        df = pd.read_csv(
            'https://raw.githubusercontent.com/YakshHaranwala/PTRAIL/main/examples/data/starkey.csv'
        )

        # Load the dataset into a PTrailDataFrame and print the dataframe
        # information.
        to_return = PTRAILDataFrame(data_set=df,
                                    latitude='lat',
                                    longitude='long',
                                    datetime='DateTime',
                                    traj_id='traj_id',
                                    rest_of_columns=[])
        print(to_return)

        # return the PTRAILDataFrame
        return to_return

    @staticmethod
    def load_starkey_habitat():
        """
            Load the Starkey dataset into a pandas dataframe and return it.

            Returns
            -------
                PTRAILDataFrame:
                    The Starkey habitat dataset.
        """
        starkey_habitat = pd.read_csv(
            'https://raw.githubusercontent.com/YakshHaranwala/PTRAIL/main/examples/data/starkey_habitat.csv'
        )
        print(f"The shape of the starkey habitat dataset: {starkey_habitat.shape}")

        return starkey_habitat
