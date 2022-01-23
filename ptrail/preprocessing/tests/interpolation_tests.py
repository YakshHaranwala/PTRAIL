import unittest
from ptrail.core.TrajectoryDF import PTRAILDataFrame
from ptrail.preprocessing.interpolation import Interpolation
import pandas as pd


class InterpolationTests(unittest.TestCase):
    _pdf_data = pd.read_csv('examples/data/seagulls.csv')
    _test_df = PTRAILDataFrame(data_set=_pdf_data,
                               latitude='location-lat',
                               longitude='location-long',
                               datetime='timestamp',
                               traj_id='tag-local-identifier',
                               rest_of_columns=[])

    def test_linear_ip(self):
        linear_ip = Interpolation.interpolate_position(self._test_df,
                                                       time_jump=3600 * 4,
                                                       ip_type='linear')
        self.assertGreaterEqual(len(linear_ip), len(self._test_df))
        self.assertEqual(len(linear_ip.reset_index().columns.to_list()), 4)

    def test_cubic_ip(self):
        cubic_ip = Interpolation.interpolate_position(self._test_df,
                                                      time_jump=3600 * 4,
                                                      ip_type='cubic')
        self.assertGreaterEqual(len(cubic_ip), len(self._test_df))
        self.assertEqual(len(cubic_ip.reset_index().columns.to_list()), 4)

    def test_rw_ip(self):
        rw_ip = Interpolation.interpolate_position(self._test_df,
                                                   time_jump=3600 * 4,
                                                   ip_type='random-walk')
        self.assertGreaterEqual(len(rw_ip), len(self._test_df))
        self.assertEqual(len(rw_ip.reset_index().columns.to_list()), 4)

    def test_kin_ip(self):
        kin_ip = Interpolation.interpolate_position(self._test_df,
                                                    time_jump=3600 * 4,
                                                    ip_type='kinematic')
        self.assertGreaterEqual(len(kin_ip), len(self._test_df))
        self.assertEqual(len(kin_ip.reset_index().columns.to_list()), 4)


if __name__ == '__main__':
    unittest.main()
