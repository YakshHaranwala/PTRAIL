import datetime
import unittest
from ptrail.core.TrajectoryDF import PTRAILDataFrame
from ptrail.features.temporal_features import TemporalFeatures
import pandas as pd
import ptrail.utilities.constants as const


class TemporalFeaturesTest(unittest.TestCase):
    _pdf_data = pd.read_csv('https://raw.githubusercontent.com/YakshHaranwala/PTRAIL/main/examples/data/seagulls.csv')
    _test_df = PTRAILDataFrame(data_set=_pdf_data,
                               latitude='location-lat',
                               longitude='location-long',
                               datetime='timestamp',
                               traj_id='tag-local-identifier',
                               rest_of_columns=[])

    def test_date_column(self):
        new_df = TemporalFeatures.create_date_column(self._test_df)
        self.assertIsInstance(new_df['Date'][0], datetime.date)
        self.assertIsNotNone(new_df['Date'])
        self.assertGreater(len(new_df['Date']), 0)

    def test_time_column(self):
        new_df = TemporalFeatures.create_time_column(self._test_df)
        self.assertIsInstance(new_df['Time'][0], datetime.time)
        self.assertIsNotNone(new_df['Time'])
        self.assertGreater(len(new_df['Time']), 0)

    def test_day_of_week(self):
        new_df = TemporalFeatures.create_day_of_week_column(self._test_df)
        self.assertIsInstance(new_df['Day_Of_Week'][0], str)
        self.assertIsNotNone(new_df['Day_Of_Week'])
        self.assertGreater(len(new_df['Day_Of_Week']), 0)

    def test_weekend(self):
        days = self._test_df.reset_index()['DateTime'].dt.day_name()
        expected_values = []
        for val in days:
            if val == 'Saturday' or val == 'Sunday':
                expected_values.append(True)
            else:
                expected_values.append(False)

        new_df = TemporalFeatures.create_weekend_indicator_column(self._test_df)
        self.assertListEqual(expected_values, new_df['Weekend'].values.tolist())

    def test_time_of_day(self):
        new_df = TemporalFeatures.create_time_of_day_column(self._test_df)
        self.assertIsInstance(new_df['Time_Of_Day'][0], str)
        self.assertIsNotNone(new_df['Time_Of_Day'])
        self.assertGreater(len(new_df['Time_Of_Day']), 0)

    def test_traj_duration(self):
        new_df = TemporalFeatures.get_traj_duration(self._test_df)
        if len(new_df) > 1:
            self.assertIsNotNone(new_df['Traj_Duration'])
            self.assertIsInstance(new_df['Traj_Duration'][0], pd.Timedelta)
        else:
            self.assertIsInstance(new_df, pd.Timedelta)
            self.assertIsNotNone(new_df)

    def test_start_time(self):
        new_df = TemporalFeatures.get_start_time(self._test_df)
        if len(new_df) > 1:
            self.assertIsNotNone(new_df['DateTime'])
            self.assertIsInstance(new_df['DateTime'][0], datetime.datetime)
        else:
            self.assertIsInstance(new_df, datetime.datetime)
            self.assertIsNotNone(new_df)

    def test_end_time(self):
        new_df = TemporalFeatures.get_end_time(self._test_df)
        if len(new_df) > 1:
            self.assertIsNotNone(new_df['DateTime'])
            self.assertIsInstance(new_df['DateTime'][0], datetime.datetime)
        else:
            self.assertIsInstance(new_df, datetime.datetime)
            self.assertIsNotNone(new_df)


if __name__ == '__main__':
    unittest.main()
