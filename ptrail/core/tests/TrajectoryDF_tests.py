import datetime
import unittest

import folium
import numpy as np
import pandas as pd

from ptrail.core.TrajectoryDF import PTRAILDataFrame
from typing import Text


class TestPTRAILDF(unittest.TestCase):
    _pdf_data = pd.read_csv('https://raw.githubusercontent.com/YakshHaranwala/PTRAIL/main/examples/data/seagulls.csv')

    _list_data = [
        [39.984094, 116.319236, '2008-10-23 05:53:05', 1],
        [39.984198, 116.319322, '2008-10-23 05:53:06', 1],
        [39.984224, 116.319402, '2008-10-23 05:53:11', 1],
        [39.984224, 116.319404, '2008-10-23 05:53:11', 1],
        [39.984224, 116.568956, '2008-10-23 05:53:11', 1],
        [39.984224, 116.568956, '2008-10-23 05:53:11', 1]
    ]

    _dict_data = {
        'lat': [39.984198, 39.984224, 39.984094, 40.98, 41.256],
        'lon': [116.319402, 116.319322, 116.319402, 116.3589, 117],
        'datetime': ['2008-10-23 05:53:11', '2008-10-23 05:53:06', '2008-10-23 05:53:30', '2008-10-23 05:54:06',
                     '2008-10-23 05:59:06'],
        'id': [1, 1, 1, 3, 3],
    }

    # ----------------------------- DataFrame Creation testing ---------------------------------- #
    def test_df_from_list(self):
        from_list_df = PTRAILDataFrame(data_set=TestPTRAILDF._list_data,
                                       latitude='lat',
                                       longitude='lon',
                                       datetime='datetime',
                                       traj_id='id')
        self.assertIsInstance(from_list_df, PTRAILDataFrame)

    def test_df_from_dict(self):
        from_dict_df = PTRAILDataFrame(data_set=TestPTRAILDF._dict_data,
                                       latitude='lat',
                                       longitude='lon',
                                       datetime='datetime',
                                       traj_id='id')
        self.assertIsInstance(from_dict_df, PTRAILDataFrame)

    def test_df_from_pdf_positive(self):
        from_pdf = PTRAILDataFrame(data_set=TestPTRAILDF._pdf_data,
                                   latitude='lat',
                                   longitude='lon',
                                   datetime='DateTime',
                                   traj_id='traj_id',
                                   rest_of_columns=[])
        self.assertIsInstance(from_pdf, PTRAILDataFrame)
        self.assertGreater(len(from_pdf), 1)

    def test_df_from_pdf_negative(self):
        """
            Check whether the dataframe's usage yields an AttributeError
            when a wrong column name is passed upon the creation of a
            PTRAILDataFrame from a pandas dataframe.

            This further emphasizes that the user needs to give in the
            correct column names when passing in a dataframe as the
            dataset when constructing a PTRAILDataFrame.
        """
        from_pdf = PTRAILDataFrame(data_set=TestPTRAILDF._pdf_data,
                                   latitude='',
                                   longitude='lon',
                                   datetime='DateTime',
                                   traj_id='traj_id',
                                   rest_of_columns=[])
        with self.assertRaises(AttributeError):
            print(from_pdf.head())

    # ---------------------------------- DataFrame Properties Testing ----------------------------------- #
    def test_lat(self):
        df = PTRAILDataFrame(data_set=TestPTRAILDF._pdf_data,
                             latitude='location-lat',
                             longitude='location-long',
                             datetime='timestamp',
                             traj_id='tag-local-identifier',
                             rest_of_columns=[])
        self.assertIsInstance(df.latitude, pd.Series)
        self.assertGreater(len(df.latitude), 0)
        self.assertIsInstance(df.latitude[0], float)

    def test_lon(self):
        df = PTRAILDataFrame(data_set=TestPTRAILDF._pdf_data,
                             latitude='location-lat',
                             longitude='location-long',
                             datetime='timestamp',
                             traj_id='tag-local-identifier',
                             rest_of_columns=[])
        self.assertIsInstance(df.longitude, pd.Series)
        self.assertGreater(len(df.longitude), 0)
        self.assertIsInstance(df.latitude[0], float)

    def test_datetime(self):
        df = PTRAILDataFrame(data_set=TestPTRAILDF._pdf_data,
                             latitude='location-lat',
                             longitude='location-long',
                             datetime='timestamp',
                             traj_id='tag-local-identifier',
                             rest_of_columns=[])
        self.assertIsInstance(df.datetime, pd.Series)
        self.assertGreater(len(df.datetime), 0)
        self.assertIsInstance(df.datetime[0], datetime.datetime)

    def test_traj_id(self):
        df = PTRAILDataFrame(data_set=TestPTRAILDF._pdf_data,
                             latitude='location-lat',
                             longitude='location-long',
                             datetime='timestamp',
                             traj_id='tag-local-identifier',
                             rest_of_columns=[])
        self.assertIsInstance(df.traj_id, pd.Series)
        self.assertGreater(len(df.traj_id), 0)
        self.assertIsInstance(df.traj_id[0], Text)

    # ------------------------------- Other Tests -------------------------------- #
    def test_sort(self):
        df = PTRAILDataFrame(data_set=TestPTRAILDF._pdf_data,
                             latitude='location-lat',
                             longitude='location-long',
                             datetime='timestamp',
                             traj_id='tag-local-identifier',
                             rest_of_columns=[])
        assert np.all(df.reset_index()[['traj_id', 'DateTime']].values ==
                      df.sort_by_traj_id_and_datetime().reset_index()[['traj_id', 'DateTime']].values)


if __name__ == '__main__':
    unittest.main()
