import pandas as pd

from core.TrajectoryDF import NumPandasTraj
from utilities import constants as const

class SpatialFeatures:
    @staticmethod
    def get_bounding_box(dataframe: NumPandasTraj):
        """
            Return the bounding box of the Trajectory data. Essentially, the bounding box is of
            the following format:
                (mini Latitude, min Longitude, max Latitude, max Longitude).

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The dataframe containing the trajectory data.

            Returns
            -------
                tuple
                    The bounding box of the trajectory
        """
        return dataframe(
            dataframe[const.LAT].min(),
            dataframe[const.LONG].min(),
            dataframe[const.LAT].max(),
            dataframe[const.LONG].max(),
        )