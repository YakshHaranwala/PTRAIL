"""
    DistanceCalculator module contains various types of distance formulas that can be used to calculate
    distance between 2 points on the surface of earth depending on the CRS being used.

    @Authors: Salman Haidri, Yaksh J Haranwala
    @Date: 28th May, 2021
    @Version 1.0
"""
import math

import numpy as np

import utilities.constants as const


class DistanceFormulaLog:
    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """
            The haversine formula calculates the great-circle distance between 2 points.
            The great-circle distance is the shortest distance over the earth's surface.

            Parameters
            ----------
                lat1: float
                    The latitude value of point 1.
                lon1: float
                    The longitude value of point 1.
                lat2: float
                    The latitude value of point 2.
                lon2: float
                    The longitude value of point 2.

            Returns
            -------
                float
                    The great-circle distance between the 2 points.
        """
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
        a = (np.sin((lat2 - lat1) / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2)

        return (const.RADIUS_OF_EARTH * 2 * np.arctan2(a ** 0.5, (1 - a) ** 0.5)) * 1000

    @staticmethod
    def bearing_calculation(lat1, lon1, lat2, lon2):
        """
            Calculates bearing between 2 points. Bearing can be defined as direction or
            an angle, between the north-south line of earth or meridian and the line connecting
            the target and the reference point.

            Parameters
            ----------
                lat1: float
                    The latitude value of point 1.
                lon1: float
                    The longitude value of point 1.
                lat2: float
                    The latitude value of point 2.
                lon2: float
                    The longitude value of point 2.

            Returns
            -------
                float
                    Bearing between 2 points
        """
        y = math.cos(math.radians(lat2)) * math.sin(math.radians(lon2) - math.radians(lon1))

        x = (math.cos(math.radians(lat1)) * math.sin(math.radians(lat2))) - \
            (math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(
                math.radians(lon2) - math.radians(lon1)))

        bearing = math.atan2(y, x)

        return math.degrees(bearing)
