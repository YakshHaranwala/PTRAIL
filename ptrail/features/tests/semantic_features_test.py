import unittest
from json import JSONDecodeError

import pandas as pd
from shapely.geometry import Polygon

from ptrail.core.TrajectoryDF import PTRAILDataFrame
from ptrail.features.semantic_features import SemanticFeatures


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

    def test_visited_location(self):
        pass

    def test_visited_poi(self):
        pass

    def test_trajectories_inside_polygon(self):
        traj_inside_poly = SemanticFeatures.trajectories_inside_polygon(df=self.starkey_traj,
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

        intersect = SemanticFeatures.traj_intersect_inside_polygon(t1, t2, self.poly)
        self.assertGreaterEqual(len(intersect), 1)
        self.assertEqual(len(intersect.columns), 6)

    def test_nearest_poi_positive(self):
        poi = SemanticFeatures.nearest_poi(coords=(47.5759762, -52.7031302),
                                           tags={'amenity': ['bank', 'atm']},
                                           dist_threshold=2500)
        self.assertGreaterEqual(len(poi), 1)

    def test_nearest_poi_negative(self):
        poi = SemanticFeatures.nearest_poi(coords=(47.5759762, -52.7031302),
                                           tags={'amenity': ['waterpark']},
                                           dist_threshold=2500)
        self.assertEqual(len(poi), 0)


if __name__ == '__main__':
    unittest.main()
