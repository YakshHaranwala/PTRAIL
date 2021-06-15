"""
    DistanceCalculator module contains various types of distance formulas that can be used to calculate
    distance between 2 points on the surface of earth depending on the CRS being used.

    @Authors: Salman Haidri, Yaksh J Haranwala
    @Date: 28th May, 2021
    @Version 1.0
"""

import numpy as np

import utilities.constants as const


class FormulaLog:
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
        val_one = (np.sin((lat2 - lat1) / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2)

        crow_distance = 2 * np.arctan2(val_one ** 0.5, (1 - val_one) ** 0.5)

        return (const.RADIUS_OF_EARTH * crow_distance) * 1000

    @staticmethod
    def bearing_calculation(lat1, lon1, lat2, lon2):
        """
            Calculates bearing between 2 points. Bearing can be defined as direction or
            an angle, between the north-south line of earth or meridian and the line connecting
            the target and the reference point.

            Parameters
            ----------
                lat1:
                    The latitude value of point 1.
                lon1:
                    The longitude value of point 1.
                lat2:
                    The latitude value of point 2.
                lon2:
                    The longitude value of point 2.

            Returns
            -------
                float
                    Bearing between 2 points
        """
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

        y = np.cos(lat2) * np.sin(lon2 - lon1)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)

        bearing = np.arctan2(y, x)

        return np.rad2deg(bearing)

    @staticmethod
    def interpolate_haversine(p1: tuple, p2: tuple):
        """
            Parameters
            ----------
                p1: tuple
                    The coordinates of point 1.
                p2: tuple
                    The coordinates of point 2.

            Returns
            -------
                float:
                    The distance between 2 points.

            References
            ----------
                "Etemad, M., Etemad, Z., Soares, A., Bogorny, V., Matwin12, S., & Torgo, L., 2020.
                Wise Sliding Window Segmentation: A classification-aided approach for trajectory segmentation."

        """
        lat, lon = p1
        lat2, lon2 = p2

        # Convert the required values to radians and then calculate the crow-distance.
        d_lat, d_lon, lat1, lat2 = map(np.radians, (lat2 - lat, lon2 - lon, lat, lat2))
        a = np.sin(d_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2) ** 2

        # Calculate the final distance.
        distance_val = 2 * np.arcsin(np.sqrt(a)) * 6372.8 * 1000  # convert to meter
        return distance_val
