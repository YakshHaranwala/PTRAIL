"""
    DistanceCalculator module contains various types of distance formulas that can be used to calculate
    distance between 2 points on the surface of earth depending on the CRS being used.

    | Authors: Salman Haidri, Yaksh J Haranwala
"""

import numpy as np

from ptrail.utilities import constants as const

np.seterr(invalid='ignore')


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
    def bearing_calculator(lat1, lon1, lat2, lon2):
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

        return (np.rad2deg(bearing)) % 360.0

    @staticmethod
    def synchronous_euclidean_distance(initial: tuple, middle: tuple, final: tuple):
        """
            Given the starting, middle and the ending point of the trajectory,
            calculate the Synchronous Euclidean Distance (SED) error between the points.

            Parameters
            ----------
                initial: tuple
                    The initial co-ordinates, time tuple of the trajectory.
                middle: tuple
                    The middle co-ordinates, time tuple of the trajectory.
                final: tuple
                    The final co-ordinates, time tuple of the trajectory.

            Returns
            -------
                float:
                    The SED error between the given points.
        """
        # Unpack the values that we have for the initial, middle and final point.
        i_lat, i_lon, i_time = initial
        m_lat, m_lon, m_time = middle
        f_lat, f_lon, f_time = final

        # Calculate the time ratio.
        time_to_middle = m_time - i_time
        total_time = f_time - i_time
        if total_time == 0:
            time_ratio = 0
        else:
            time_ratio = time_to_middle / total_time

        # Calculate the final error and return it.
        lat = i_lat + (f_lat - i_lat) * time_ratio
        lon = i_lon + (f_lon - i_lon) * time_ratio

        lat_diff = lat - i_lat
        lon_diff = lon - i_lon
        return np.sqrt(np.power(lat_diff, 2) + np.power(lon_diff, 2))

    @staticmethod
    def perpendicular_distance(initial: tuple, middle: tuple, final: tuple):
        """
            Given the starting, middle and the ending point of the trajectory,
            calculate the Perpendicular Distance (PD).

            Parameters
            ----------
                initial: tuple
                    The initial co-ordinates, time tuple of the trajectory.
                middle: tuple
                    The middle co-ordinates, time tuple of the trajectory.
                final: tuple
                    The final co-ordinates, time tuple of the trajectory.

            Returns
            -------
                float:
                    The PD between the given points.
        """
        # Unpack the values that we have for the initial, middle and final point.
        i_lat, i_lon, i_time = initial
        m_lat, m_lon, m_time = middle
        f_lat, f_lon, f_time = final

        # Equation for PD: (yA - yB)x - (xA - xB) + xAyB -xByA = 0
        A = i_lon - f_lon
        B = -(i_lat - f_lat)
        C = (i_lat*f_lon) - (f_lat*i_lon)

        # Based on coefficients calculated above, return the final PD.
        if A == 0 and B == 0:
            return 0
        return np.abs((A * m_lat + B * m_lon + C) / np.sqrt(np.power(A, 2) + np.power(B, 2)))

    @staticmethod
    def absolute_speed_value(initial: tuple, middle: tuple, final: tuple):
        """
            Given the starting, middle and the ending point of the trajectory,
            calculate the Absolute Value of Speed between 2 initial and final
            points.

            Parameters
            ----------
                initial: tuple
                    The initial co-ordinates, time tuple of the trajectory.
                middle: tuple
                    The middle co-ordinates, time tuple of the trajectory.
                final: tuple
                    The final co-ordinates, time tuple of the trajectory.

            Returns
            -------
                float:
                    The AVS between the given points.
        """
        # Unpack the values that we have for the initial, middle and final point.
        i_lat, i_lon, i_time = initial
        m_lat, m_lon, m_time = middle
        f_lat, f_lon, f_time = final

        # Calculate distance from start to middle and middle to end.
        d1 = np.sqrt(np.power(m_lat - i_lat, 2) + np.power(m_lon - i_lon, 2))
        d2 = np.sqrt(np.power(f_lat - m_lat, 2) + np.power(f_lon - m_lon, 2))

        # Calculate the speeds of the 2 sectors calculated above.
        v1, v2 = 0, 0
        if m_time - i_time > 0:
            v1 = d1 / (m_time - i_time)
        if f_time - m_time:
            v2 = d2 / (f_time - m_time)

        return np.abs(v2 - v1)
