import unittest
from json import JSONDecodeError

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from ptrail.core.TrajectoryDF import PTRAILDataFrame
from ptrail.features.contextual_features import ContextualFeatures


class SemanticTests(unittest.TestCase):
    pdf = pd.read_csv('examples/data/starkey.csv')
    starkey_traj = PTRAILDataFrame(data_set=pdf,
                                   latitude='lat',
                                   longitude='lon',
                                   datetime='DateTime',
                                   traj_id='Id')
    starkey_habitat = pd.read_csv('examples/data/starkey_habitat.csv')

    mini_pasture = starkey_habitat.loc[starkey_habitat['CowPast'] == 'BEAR']
    coords = list(zip(mini_pasture['lon'], (mini_pasture['lat'])))
    poly = Polygon(coords)

    single_traj = starkey_traj.reset_index().loc[
        starkey_traj.reset_index()['traj_id'] == '880109D01']
    single_traj = PTRAILDataFrame(single_traj,
                                  latitude='lat',
                                  longitude='lon',
                                  datetime='DateTime',
                                  traj_id='traj_id')

    def test_visited_location_positive(self):
        visited_location = ContextualFeatures.visited_location(df=self.starkey_traj,
                                                               geo_layers=self.starkey_habitat,
                                                               visited_location_name='BEAR',
                                                               location_column_name='CowPast')
        self.assertIsNotNone(visited_location['Visited_BEAR'])
        self.assertIsInstance(visited_location['Visited_BEAR'][0], np.int64)

    def test_visited_location_negative(self):
        with self.assertRaises(KeyError):
            visited_location = ContextualFeatures.visited_location(df=self.starkey_traj,
                                                                   geo_layers=self.starkey_habitat,
                                                                   visited_location_name='FAKE_NAME',
                                                                   location_column_name='CowPast')

    def test_visited_poi_positive(self):
        water_visited = ContextualFeatures.visited_poi(df=self.single_traj,
                                                       surrounding_data=self.mini_pasture,
                                                       dist_column_label='DistEWat',
                                                       nearby_threshold=10)
        self.assertIsNotNone(water_visited['Nearby_POI'])
        self.assertIsInstance(water_visited['Nearby_POI'][0], np.bool_)

    def test_visited_poi_negative(self):
        with self.assertRaises(KeyError):
            water_visited = ContextualFeatures.visited_poi(df=self.single_traj,
                                                           surrounding_data=self.mini_pasture,
                                                           dist_column_label='Fake_Name',
                                                           nearby_threshold=10)


    def test_trajectories_inside_polygon(self):
        traj_inside_poly = ContextualFeatures.trajectories_inside_polygon(df=self.starkey_traj,
                                                                          polygon=self.poly)
        self.assertGreaterEqual(len(traj_inside_poly), 1)
        self.assertListEqual(list(traj_inside_poly.reset_index().columns),
                             list(self.starkey_traj.reset_index().columns))

    def test_traj_intersect_inside_polygon(self):
        t1 = self.starkey_traj.reset_index().loc[self.starkey_traj.reset_index()['traj_id'] == '910313E37']
        t1 = PTRAILDataFrame(t1,
                             latitude='lat',
                             longitude='lon',
                             datetime='DateTime',
                             traj_id='traj_id')

        t2 = self.starkey_traj.reset_index().loc[self.starkey_traj.reset_index()['traj_id'] == '890424E08']
        t2 = PTRAILDataFrame(t2,
                             latitude='lat',
                             longitude='lon',
                             datetime='DateTime',
                             traj_id='traj_id')

        intersect = ContextualFeatures.traj_intersect_inside_polygon(t1, t2, self.poly)
        self.assertGreaterEqual(len(intersect), 1)
        self.assertEqual(len(intersect.columns), 6)

    def test_nearest_poi_positive(self):
        poi = ContextualFeatures.nearest_poi(coords=(47.5759762, -52.7031302),
                                             tags={'amenity': ['bank', 'atm']},
                                             dist_threshold=2500)
        self.assertGreaterEqual(len(poi), 1)

    def test_nearest_poi_negative(self):
        poi = ContextualFeatures.nearest_poi(coords=(47.5759762, -52.7031302),
                                             tags={'amenity': ['waterpark']},
                                             dist_threshold=2500)
        self.assertEqual(len(poi), 0)


if __name__ == '__main__':
    unittest.main()
