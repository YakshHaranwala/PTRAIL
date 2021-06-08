"""
    The conversions modules contains various available methods
    that can be used to convert given data into another format.

    @Author Yaksh J Haranwala, Salman Haidri
    @Date: 1st June, 2021
"""
from typing import Text


class Conversions:
    @staticmethod
    def convert_directions_to_degree_lat_lon(data, latitude: Text, longitude: Text):
        """
            Cite: PyMove
        """

        def decimal_degree_to_decimal(col):
            if col[latitude][-1:] == 'N':
                col[latitude] = float(col[latitude][:-1])
            else:
                col[latitude] = float(col[latitude][:-1]) * -1

            if col[longitude][-1:] == 'E':
                col[longitude] = float(col[longitude][:-1])
            else:
                col[longitude] = float(col[longitude][:-1]) * -1 + 360 if float(col[longitude][:-1]) * -1 < -180 \
                    else float(col[longitude][:-1]) * -1
            return col

        return data.apply(decimal_degree_to_decimal, axis=1)
