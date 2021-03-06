import unittest
from ptrail.core.TrajectoryDF import PTRAILDataFrame
from ptrail.preprocessing.filters import Filters
from ptrail.features.temporal_features import TemporalFeatures
from ptrail.features.kinematic_features import KinematicFeatures
from ptrail.utilities.exceptions import *
from ptrail.preprocessing.helpers import Helpers
import pandas as pd


class FiltersTest(unittest.TestCase):
    _pdf_data = pd.read_csv('https://raw.githubusercontent.com/YakshHaranwala/PTRAIL/main/examples/data/seagulls.csv')
    _gulls = PTRAILDataFrame(data_set=_pdf_data,
                             latitude='location-lat',
                             longitude='location-long',
                             datetime='timestamp',
                             traj_id='tag-local-identifier',
                             rest_of_columns=[])

    _atlantic = pd.read_csv('https://raw.githubusercontent.com/YakshHaranwala/PTRAIL/main/examples/data/atlantic_hurricanes.csv')
    _atlantic = PTRAILDataFrame(_atlantic,
                                latitude='lat',
                                longitude='lon',
                                datetime='DateTime',
                                traj_id='traj_id',
                                rest_of_columns=[])

    def test_remove_duplicates(self):
        remove_dupl = Filters.remove_duplicates(self._gulls)
        self.assertGreaterEqual(len(self._gulls), len(remove_dupl))

    def test_filter_by_traj_id_positive(self):
        filt_traj_id = Filters.filter_by_traj_id(dataframe=self._atlantic,
                                                 traj_id='AL011851')
        self.assertEqual(14, len(filt_traj_id))

    def test_filter_by_traj_id_(self):
        with self.assertRaises(MissingTrajIDException):
            filt_traj_id = Filters.filter_by_traj_id(dataframe=self._atlantic,
                                                     traj_id='PTRAIL')

    def test_get_bbox_by_radius(self):
        bbox = Filters.get_bounding_box_by_radius(lat=39, lon=116, radius=100000)
        expected = [38.100678394081264, 114.84275815636957, 39.89932160591873, 117.15724184363044]
        self.assertListEqual(list(bbox), expected)

    def test_filter_by_bbox(self):
        bbox = Filters.get_bounding_box_by_radius(lat=61, lon=24, radius=100000)
        filt_df = Filters.filter_by_bounding_box(dataframe=self._gulls, bounding_box=bbox, inside=True)
        self.assertGreaterEqual(len(self._gulls), len(filt_df))

    def test_filter_by_date_positive(self):
        new_df = TemporalFeatures.create_date_column(self._gulls)
        filt_df = Filters.filter_by_date(dataframe=new_df,
                                         start_date='2009-05-27',
                                         end_date='2009-12-31')
        self.assertGreaterEqual(len(new_df), len(filt_df))

    def test_filter_by_date_negative_2(self):
        with self.assertRaises(ValueError):
            filt_df = Filters.filter_by_date(dataframe=self._gulls,
                                             start_date='2009-12-31',
                                             end_date='2009-08-27')

    def test_filter_by_datetime_positive(self):
        filt_df = Filters.filter_by_datetime(dataframe=self._gulls,
                                             start_dateTime='2009-05-27 14:00:00',
                                             end_dateTime='2009-05-31 00:00:00')
        self.assertGreaterEqual(len(self._gulls), len(filt_df))

    def test_filter_by_datetime_negative(self):
        with self.assertRaises(ValueError):
            filt_df = Filters.filter_by_datetime(dataframe=self._gulls,
                                                 end_dateTime='2009-05-27 14:00:00',
                                                 start_dateTime='2009-05-31 00:00:00')

    def test_filter_by_max_speed_positive(self):
        new_df = KinematicFeatures.create_speed_column(self._gulls)
        filt_df = Filters.filter_by_max_speed(dataframe=new_df,
                                              max_speed=5)
        self.assertGreaterEqual(len(new_df), len(filt_df))

    def test_filter_by_min_speed_positive(self):
        new_df = KinematicFeatures.create_speed_column(self._gulls)
        filt_df = Filters.filter_by_min_speed(dataframe=new_df,
                                              min_speed=1)
        self.assertGreaterEqual(len(new_df), len(filt_df))

    def test_filter_by_min_consecutive_distance_positive(self):
        new_df = KinematicFeatures.create_distance_column(dataframe=self._gulls)
        filt_df = Filters.filter_by_min_consecutive_distance(dataframe=new_df,
                                                             min_distance=1000)
        self.assertGreaterEqual(len(new_df), len(filt_df))

    def filter_by_min_consecutive_distance_negative(self):
        with self.assertRaises(MissingColumnsException):
            filt_df = Filters.filter_by_min_consecutive_distance(dataframe=self._gulls,
                                                                 min_distance=1000)

    def test_filter_by_max_consecutive_distance_positive(self):
        new_df = KinematicFeatures.create_distance_column(dataframe=self._gulls)
        filt_df = Filters.filter_by_max_consecutive_distance(dataframe=new_df,
                                                             max_distance=10000)
        self.assertGreaterEqual(len(new_df), len(filt_df))

    def filter_by_max_consecutive_distance_negative(self):
        with self.assertRaises(MissingColumnsException):
            filt_df = Filters.filter_by_max_consecutive_distance(dataframe=self._gulls,
                                                                 max_distance=10000)

    def test_filter_by_max_distance_and_speed_positive(self):
        new_df = KinematicFeatures.create_speed_column(self._gulls)
        filt_df = Filters.filter_by_max_distance_and_speed(dataframe=new_df,
                                                           max_speed=25,
                                                           max_distance=1000)
        self.assertGreaterEqual(len(new_df), len(filt_df))

    def test_filter_by_min_distance_and_speed_positive(self):
        new_df = KinematicFeatures.create_speed_column(self._gulls)
        filt_df = Filters.filter_by_min_distance_and_speed(dataframe=new_df,
                                                           min_speed=5,
                                                           min_distance=10)
        self.assertGreaterEqual(len(new_df), len(filt_df))

    def test_filter_outliers_by_consecutive_distance_positive(self):
        new_df = KinematicFeatures.create_distance_column(dataframe=self._gulls)
        filt_df = Filters.filter_outliers_by_consecutive_distance(dataframe=new_df)
        self.assertGreaterEqual(len(new_df), len(filt_df))

    def test_filter_outliers_by_consecutive_speed_positive(self):
        new_df = KinematicFeatures.create_speed_column(dataframe=self._gulls)
        filt_df = Filters.filter_outliers_by_consecutive_speed(dataframe=new_df)
        self.assertGreaterEqual(len(new_df), len(filt_df))

    def test_remove_trajectories_with_less_points(self):
        filt_df = Filters.remove_trajectories_with_less_points(dataframe=self._atlantic)
        self.assertGreater(len(self._atlantic), len(filt_df))

    def test_hampel_positive(self):
        new_df = KinematicFeatures.create_distance_column(self._atlantic)
        filt_df = Filters.hampel_outlier_detection(dataframe=new_df,
                                                   column_name='Distance')
        self.assertGreater(len(self._atlantic), len(filt_df))

    def test_hampel_negative(self):
        with self.assertRaises(MissingColumnsException):
            filt_df = Helpers.hampel_help(df=self._gulls,
                                          column_name='Distance')


if __name__ == '__main__':
    unittest.main()
