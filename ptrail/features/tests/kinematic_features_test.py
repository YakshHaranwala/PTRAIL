import unittest

import numpy as np
import pandas as pd
from ptrail.core.TrajectoryDF import PTRAILDataFrame
from ptrail.features.kinematic_features import KinematicFeatures
import ptrail.utilities.constants as const
from ptrail.utilities.exceptions import MissingTrajIDException


class KinematicFeaturesTest(unittest.TestCase):
    _pdf_data = pd.read_csv('examples/data/gulls.csv')
    _test_df = PTRAILDataFrame(data_set=_pdf_data,
                               latitude='location-lat',
                               longitude='location-long',
                               datetime='timestamp',
                               traj_id='tag-local-identifier',
                               rest_of_columns=[])

    def test_get_bb(self):
        bb = KinematicFeatures.get_bounding_box(self._test_df)
        self.assertIsNotNone(bb)
        # Check whether the lat max and max are at least equal
        # or greater than the lat min and min.
        self.assertGreaterEqual(bb[2], bb[0])
        self.assertGreaterEqual(bb[3], bb[1])

    def test_get_start_location(self):
        new_df = KinematicFeatures.get_start_location(self._test_df)
        if len(new_df) > 1:
            self.assertIsNotNone(new_df[[const.LAT, const.LONG]])
            self.assertIsInstance(new_df[const.LAT][0], float)
            self.assertIsInstance(new_df[const.LONG][0], float)
        else:
            self.assertIsInstance(new_df[0], float)
            self.assertIsInstance(new_df[1], float)
            self.assertIsNotNone(new_df)

    def test_get_end_location(self):
        new_df = KinematicFeatures.get_end_location(self._test_df)
        if len(new_df) > 1:
            self.assertIsNotNone(new_df[[const.LAT, const.LONG]])
            self.assertIsInstance(new_df[const.LAT][0], float)
            self.assertIsInstance(new_df[const.LONG][0], float)
        else:
            self.assertIsInstance(new_df[0], float)
            self.assertIsInstance(new_df[1], float)
            self.assertIsNotNone(new_df)

    def test_dist_between_consecutive(self):
        new_df = KinematicFeatures.create_distance_column(self._test_df)
        self.assertIsNotNone(new_df['Distance'])

        ids_ = list(new_df.traj_id.value_counts().keys())
        for i in range(len(ids_)):
            filt_df = new_df.reset_index().loc[new_df.reset_index()[const.TRAJECTORY_ID] == ids_[i]]
            if len(filt_df) > 3:
                self.assertIsInstance(filt_df['Distance_prev_to_curr'].iloc[1], float)
                assert np.isnan(filt_df['Distance_prev_to_curr'].iloc[0])

    def test_dist_from_start(self):
        new_df = KinematicFeatures.create_distance_from_start_column(self._test_df)
        self.assertIsNotNone(new_df['Distance_start_to_curr'])

        ids_ = list(new_df.traj_id.value_counts().keys())

        for i in range(len(ids_)):
            filt_df = new_df.reset_index().loc[new_df.reset_index()[const.TRAJECTORY_ID] == ids_[i]]
            if len(filt_df) > 3:
                self.assertIsInstance(filt_df['Distance_start_to_curr'].iloc[1], float)
                assert np.isnan(filt_df['Distance_start_to_curr'].iloc[0])

    def test_distance_travelled_by_date_traj_id_positive(self):
        dist = KinematicFeatures.distance_travelled_by_date_and_traj_id(dataframe=self._test_df,
                                                                        date='2009-05-27',
                                                                        traj_id='91732')
        self.assertGreater(dist, 0)
        self.assertIsInstance(dist, float)

    def test_distance_travelled_by_date_traj_id_negative(self):
        with self.assertRaises(KeyError):
            KinematicFeatures.distance_travelled_by_date_and_traj_id(dataframe=self._test_df,
                                                                     date='2009-05-27',
                                                                     traj_id='91000')

    def test_point_within_range(self):
        new_df = KinematicFeatures.create_point_within_range_column(dataframe=self._test_df,
                                                                    coordinates=(0, 0),
                                                                    dist_range=100000)
        self.assertIsNotNone(new_df['Within_100000_m_from_(0, 0)'])
        self.assertIsInstance(new_df['Within_100000_m_from_(0, 0)'].iloc[0], np.bool_)

    def test_distance_from_given_point(self):
        new_df = KinematicFeatures.create_distance_from_point_column(dataframe=self._test_df,
                                                                     coordinates=(0, 0))
        self.assertIsNotNone(new_df['Distance_from_(0, 0)'])
        self.assertIsInstance(new_df['Distance_from_(0, 0)'].iloc[0], float)

    def test_speed_between_consecutive(self):
        new_df = KinematicFeatures.create_speed_column(self._test_df)
        self.assertIsNotNone(new_df['Speed'])
        self.assertIsInstance(new_df['Speed'][1], float)

        ids_ = list(new_df.traj_id.value_counts().keys())
        for i in range(len(ids_)):
            filt_df = new_df.reset_index().loc[new_df.reset_index()[const.TRAJECTORY_ID] == ids_[i]]
            assert np.isnan(filt_df['Speed_prev_to_curr'].iloc[0])

    def test_acceleration_between_consecutive(self):
        new_df = KinematicFeatures.create_acceleration_column(self._test_df)
        self.assertIsNotNone(new_df['Acceleration'])

        ids_ = list(new_df.traj_id.value_counts().keys())
        for i in range(len(ids_)):
            filt_df = new_df.reset_index().loc[new_df.reset_index()[const.TRAJECTORY_ID] == ids_[i]]
            if len(filt_df) > 3:
                assert np.isnan(filt_df['Acceleration_prev_to_curr'].iloc[0])
                assert np.isnan(filt_df["Acceleration_prev_to_curr"].iloc[1])
                self.assertIsInstance(filt_df['Acceleration_prev_to_curr'].iloc[2], float)

    def test_jerk_between_consecutive(self):
        new_df = KinematicFeatures.create_jerk_column(self._test_df)
        self.assertIsNotNone(new_df['Jerk'])

        ids_ = list(new_df.traj_id.value_counts().keys())
        for i in range(len(ids_)):
            filt_df = new_df.reset_index().loc[new_df.reset_index()[const.TRAJECTORY_ID] == ids_[i]]
            if len(filt_df) > 4:
                assert np.isnan(filt_df['Jerk_prev_to_curr'].iloc[0])
                assert np.isnan(filt_df['Jerk_prev_to_curr'].iloc[1])
                assert np.isnan(filt_df["Jerk_prev_to_curr"].iloc[2])
                self.assertIsInstance(filt_df['Jerk_prev_to_curr'].iloc[3], float)

    def test_bearing(self):
        new_df = KinematicFeatures.create_bearing_column(self._test_df)
        self.assertIsNotNone(new_df['Bearing'])

        ids_ = list(new_df.traj_id.value_counts().keys())
        for i in range(len(ids_)):
            filt_df = new_df.reset_index().loc[new_df.reset_index()[const.TRAJECTORY_ID] == ids_[i]]
            if len(filt_df) > 3:
                assert np.isnan(filt_df['Bearing'].iloc[0])
                self.assertIsInstance(filt_df['Bearing'].iloc[1], float)

    def test_bearing_rate(self):
        new_df = KinematicFeatures.create_bearing_rate_column(self._test_df)
        self.assertIsNotNone(new_df['Bearing_Rate'])

        ids_ = list(new_df.traj_id.value_counts().keys())
        for i in range(len(ids_)):
            filt_df = new_df.reset_index().loc[new_df.reset_index()[const.TRAJECTORY_ID] == ids_[i]]
            if len(filt_df) > 3:
                assert np.isnan(filt_df['Bearing_Rate'].iloc[0])
                assert np.isnan(filt_df['Bearing_Rate'].iloc[1])
                self.assertIsInstance(filt_df['Bearing_Rate'].iloc[2], float)

    def test_rate_of_bearing_rate(self):
        new_df = KinematicFeatures.create_rate_of_br_column(self._test_df)
        self.assertIsNotNone(new_df['Rate_of_bearing_rate'])

        ids_ = list(new_df.traj_id.value_counts().keys())
        for i in range(len(ids_)):
            filt_df = new_df.reset_index().loc[new_df.reset_index()[const.TRAJECTORY_ID] == ids_[i]]
            if len(filt_df) > 4:
                assert np.isnan(filt_df['Rate_of_bearing_rate'].iloc[0])
                assert np.isnan(filt_df['Rate_of_bearing_rate'].iloc[1])
                self.assertIsInstance(filt_df['Rate_of_bearing_rate'].iloc[2], float)

    def test_distance_travelled_by_traj_id_positive(self):
        dist = KinematicFeatures.get_distance_travelled_by_traj_id(dataframe=self._test_df,
                                                                   traj_id='91732')
        self.assertGreater(dist, 0)
        self.assertIsInstance(dist, float)

    def test_distance_travelled_by_traj_id_negative(self):
        with self.assertRaises(MissingTrajIDException):
            KinematicFeatures.get_distance_travelled_by_traj_id(dataframe=self._test_df,
                                                                traj_id='91000')

    def test_number_of_locations(self):
        new_df = KinematicFeatures.get_number_of_locations(self._test_df)
        if len(new_df) > 1:
            self.assertIsNotNone(new_df[['Number of Unique Coordinates']])
            self.assertIsInstance(new_df['Number of Unique Coordinates'][0], np.int_)
        else:
            self.assertIsInstance(new_df, int)
            self.assertIsNotNone(new_df)


if __name__ == '__main__':
    unittest.main()
