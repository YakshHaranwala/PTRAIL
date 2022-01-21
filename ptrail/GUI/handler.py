"""
    This class is used to connect the PTRAIL GUI to PTRAIL
    backend. All the GUI's functionalities are handled in this
    class.

    | Authors: Yaksh J Haranwala
"""
import io
import random
import folium
import pandas as pd
from PyQt5 import QtWidgets, QtWebEngineWidgets

from ptrail.GUI.Table import TableModel
from ptrail.GUI.InputDialog import InputDialog
from ptrail.core.TrajectoryDF import PTRAILDataFrame
import ptrail.utilities.constants as const


class GuiHandler:
    def __init__(self, filename, window):
        self._window = window
        self._data = None
        self._model = None
        self._table = None

        self.display_df(filename=filename)
        self.add_map()

    def display_df(self, filename):
        """
            Display the DataFrame on the DFPane of the GUI.

            Parameters
            ----------
                filename: str
                    The name of the file. This is obtained from the GUI.

            Returns
            -------
                None
        """
        # First, we clear out the DF pane area.
        # This is done in order to make sure that 2 dataframes
        # are not loaded simultaneously making the view cluttered.
        for i in reversed(range(self._window.DFPane.count())):
            item = self._window.DFPane.itemAt(i)
            if isinstance(item, QtWidgets.QTableView):
                item.widget().close()

            # remove the item from layout
            self._window.DFPane.removeItem(item)

        # Create the input dialog item.
        input_dialog = InputDialog(parent=self._window,
                                   labels=['Trajectory ID: ', 'DateTime: ', 'Latitude: ', 'Longitude: '],
                                   title='Enter the column names: ')

        # Get the input before displaying the dataframe.
        if input_dialog.exec():
            # Get the column names.
            col_names = input_dialog.getInputs()

            # Read the data into a PTRAIL datafram
            self._data = PTRAILDataFrame(data_set=pd.read_csv(filename),
                                         traj_id=col_names[0],
                                         datetime=col_names[1],
                                         latitude=col_names[2],
                                         longitude=col_names[3])
            # Set the table model and display the dataframe.
            self._table = QtWidgets.QTableView()

            # NOTE: whenever we update DFs, make sure to send the data after resetting
            #       index and setting inplace as False.
            self._model = TableModel(self._data.reset_index(inplace=False))
            self._table.setModel(self._model)
            self._window.DFPane.addWidget(self._table)

    def add_map(self, weight: float = 3, opacity: float = 0.8):
        """
            Use folium to plot the trajectory on a map.
            Parameters
            ----------
                weight: float
                    The weight of the trajectory line on the map.
                opacity: float
                    The opacity of the trajectory line on the map.
            Returns
            -------
                folium.folium.Map
                    The map with plotted trajectory.
        """
        sw = self._data[['lat', 'lon']].min().values.tolist()
        ne = self._data[['lat', 'lon']].max().values.tolist()

        # Create a map with the initial point.
        map_ = folium.Map(location=(self._data.latitude[0], self._data.longitude[0]),
                          zoom_start=13)

        ids_ = list(self._data.traj_id.value_counts().keys())
        colors = ["#" + ''.join([random.choice('123456789BCDEF') for j in range(6)])
                  for i in range(len(ids_))]

        for i in range(len(ids_)):
            # First, filter out the smaller dataframe.
            small_df = self._data.reset_index().loc[self._data.reset_index()[const.TRAJECTORY_ID] == ids_[i],
                                                    [const.LAT, const.LONG]]

            # # Then, create (lat, lon) pairs for the data points.
            locations = []
            for j in range(len(small_df)):
                locations.append((small_df['lat'].iloc[j], small_df['lon'].iloc[j]))
            #
            # Create popup text for the

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
            # folium.PolyLine(locations,
            #                 color=colors[i],
            #                 weight=weight,
            #                 opacity=opacity,).add_to(map_)

        map_.fit_bounds([sw, ne])

        data = io.BytesIO()
        map_.save(data, close_file=False)
        self._window.map.setHtml(data.getvalue().decode())
