"""
    DistanceCalculator module contains various types of distance formulas that can be used to calculate
    distance between 2 points on the surface of earth depending on the CRS being used.

    @Authors: Salman Haidri, Yaksh J Haranwala
    @Date: 28th May, 2021
    @Version 1.0
"""
import math

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
                metres: bool
                    Indicate whether to return the distance in metres or kilometres.

            Returns
            -------
                float
                    The great-circle distance between the 2 points.
        """
        dLat = math.radians(lat2 - lat1)
        dLon = math.radians(lon2 - lon1)
        a = ((math.sin(dLat / 2)) ** 2) + \
            (math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * (math.sin(dLon / 2)) ** 2)

        crow_distance = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return const.RADIUS_OF_EARTH * crow_distance
