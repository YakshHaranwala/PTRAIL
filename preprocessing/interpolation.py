"""
    The interpolation module contains several interpolation techniques
    for the trajectory data. Interpolation techniques is used to
    smoothen the otherwise incomplete or rough trajectory data.

    @Authors: Yaksh J Haranwala, Salman Haidri
    @Date: 10th June, 2021
    @Version 1.0
"""
from typing import Optional, Text

import pandas as pd

from core.TrajectoryDF import NumPandasTraj as NumTrajDF
from utilities.DistanceCalculator import FormulaLog as calc
import utilities.constants as const


# TODO: So basically, the WS-II Repository is performing calculations and
#       returning a point that is interpolated on the basis of 7 rows of dataframes.
#       Now, the task for us is to figure out how we interpolate points for
#       trajectories with less than 7 points. As from what I can see, for Linear,
#       it is only taking average of coordinates of 2 points as per the formula.
#       So we need to take that approach as well to go forward with it.

# TODO: The approach that we are going to take is:
#           1. Instead of passing a fixed number of values, we will just pass a dataframe
#              with a certain number of trajectories or just a single trajectory.
#           2. The dataframe will be based on the value count of the trajectory, i.e., maybe
#              a trajectory just over x number of values.


class Interpolate:
    @staticmethod
    def interpolation(dataframe, type: Optional[Text] = 'linear'):
        ans = []
        for i in range(dataframe.shape[0]-7):
            se7en = dataframe.iloc[i:i+7, :]
            ans.append(Interpolate.linear_helper(se7en))

        return pd.DataFrame(ans).rename({0: 'P1',
                                         1: 'P2',
                                         2: 'Interpolated?',
                                         3: 'Distance'}, axis=1)

    @staticmethod
    def linear_helper(dataframe):
        lat = dataframe[const.LAT].values
        lon = dataframe[const.LONG].values
        pc = ((lat[2] + lat[4]) / 2, (lon[2] + lon[4]) / 2)
        d = calc.interpolate_haversine(pc, (lat[3], lon[3]))
        return [(lat[2], lon[2]), (lat[4], lon[4]), pc, d]

    @staticmethod
    def cubic_interpolation(dataframe):
        pass

    @staticmethod
    def kinetic_interpolation(dataframe):
        pass

    @staticmethod
    def random_walk_interpolation(dataframe):
        pass
