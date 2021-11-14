"""
    The visualization file contains several data visualization types
    like trajectory visualizer, radar maps, bar charts etc. It is to be noted
    that ipywidgets are used to make the visualizations in this module interactive.

    Warning
    -------
        The visualizations in this module are currently developed with a focus around the
        starkey.csv data as it has been developed as a side project by the developers. It
        will further be integrated into the library as a general class of visualizers in
        the time to come.

    | Authors: Yaksh J Haranwala, Salman Haidri
"""
import random

import folium

from ptrail.core.TrajectoryDF import PTRAILDataFrame
import ptrail.utilities.constants as const


class Visualization:
    @staticmethod
    def plot_folium_traj(dataset: PTRAILDataFrame, weight: float = 3, opacity: float = 0.8):
        """
            Use folium to plot the trajectory on a map.

            Parameters
            ----------
                dataset:

                weight: float
                    The weight of the trajectory line on the map.
                opacity: float
                    The opacity of the trajectory line on the map.

            Returns
            -------
                folium.folium.Map
                    The map with plotted trajectory.
        """
        sw = dataset[['lat', 'lon']].min().values.tolist()
        ne = dataset[['lat', 'lon']].max().values.tolist()

        # Create a map with the initial point.
        map_ = folium.Map(location=(dataset.latitude[0], dataset.longitude[0]))

        ids_ = list(dataset.traj_id.value_counts().keys())
        colors = ["#" + ''.join([random.choice('123456789BCDEF') for j in range(6)])
                  for i in range(len(ids_))]

        for i in range(len(ids_)):
            # First, filter out the smaller dataframe.
            small_df = dataset.reset_index().loc[dataset.reset_index()[const.TRAJECTORY_ID] == ids_[i],
                                                 [const.LAT, const.LONG]]

            # Then, create (lat, lon) pairs for the data points.
            locations = []
            for j in range(len(small_df)):
                locations.append((small_df['lat'].iloc[j], small_df['lon'].iloc[j]))

            # Create start and end markers for the trajectory.
            folium.Marker([small_df['lat'].iloc[0], small_df['lon'].iloc[0]],
                          color='green',
                          popup=f'Trajectory ID: {ids_[i]} \n'
                                f'Latitude: {locations[0][0]} \n'
                                f'Longitude: {locations[0][1]}',
                          marker_color='green',
                          icon=folium.Icon(icon_color='green', icon=None)).add_to(map_)

            folium.Marker([small_df['lat'].iloc[-1], small_df['lon'].iloc[-1]],
                          color='green',
                          popup=f'Trajectory ID: {ids_[i]} \n'
                                f'Latitude: {locations[-1][0]} \n'
                                f'Longitude: {locations[-1][1]}',
                          marker_color='red',
                          icon=folium.Icon(icon_color='red', icon=None)).add_to(map_)

            # Add trajectory to map.
            folium.PolyLine(locations,
                            color=colors[i],
                            weight=weight,
                            opacity=opacity).add_to(map_)

        map_.fit_bounds([sw, ne])
        return map_
