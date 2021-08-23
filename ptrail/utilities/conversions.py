"""
    The conversions modules contains various available methods
    that can be used to convert given data into another format.

    | Authors: Yaksh J Haranwala, Salman Haidri
"""
from typing import Text


class Conversions:
    @staticmethod
    def convert_directions_to_degree_lat_lon(data, latitude: Text, longitude: Text):
        """
            Convert the latitude and longitude format from degrees (NSEW)
            to float values. This is used for datasets like the Atlantic Hurricane dataset
            where the coordinates are not given as float values but are instead given as
            degrees.

            References
            ----------
                "Arina De Jesus Amador Monteiro Sanches. “Uma Arquitetura E Imple-menta ̧c ̃ao Do M ́odulo De
                Pr ́e-processamento Para Biblioteca Pymove”.Bachelor’s thesis. Universidade Federal Do Cear ́a, 2019"
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
