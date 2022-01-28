"""
    This class is used to connect the PTRAIL GUI to PTRAIL
    backend. All the GUI's functionalities are handled in this
    class.

    | Authors: Yaksh J Haranwala, Salman Haidri
"""
import distutils
import random
import re
import sys
import time

import folium
import inspect
import pandas as pd

# GUI Imports.
from PyQt5 import QtWidgets, QtWebEngineWidgets
from ptrail.GUI.Table import TableModel
from ptrail.GUI.InputDialog import InputDialog
from ptrail.core.TrajectoryDF import PTRAILDataFrame

# Backend.
import ptrail.utilities.constants as const
from ptrail.features.kinematic_features import KinematicFeatures
from ptrail.features.temporal_features import TemporalFeatures
from ptrail.preprocessing.statistics import Statistics
from ptrail.preprocessing.filters import Filters
from ptrail.preprocessing.interpolation import Interpolation


class GuiHandler:
    def __init__(self, filename, window):
        self.traj_id_list = None

        self.map = None
        self._window = window
        self._map_data = None
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

            Raises
            ------
                AttributeError:
                    If the user gives incorrect column names, then we ask
                    the user to enter them again.
        """
        # self._window.statusBar.showMessage("Loading Dataset")
        try:
            self._window.statusBar.showMessage("Loading the Dataset...")
            # First, we clear out the DF pane area.
            # This is done in order to make sure that 2 dataframes
            # are not loaded simultaneously making the view cluttered.
            for i in reversed(range(self._window.DFPane.count())):
                item = self._window.DFPane.itemAt(i)
                if isinstance(item, QtWidgets.QTableView):
                    item.widget().close()

                # remove the item from layout
                self._window.DFPane.removeItem(item)

            col_names = self._get_input_params(labels=['Trajectory ID: ', 'DateTime: ', 'Latitude: ', 'Longitude: '],
                                               title="Enter Column Names")
            if col_names is not None and col_names[0] != '' and len(col_names) == 4:
                # Read the data into a PTRAIL dataframe.
                self._data = PTRAILDataFrame(data_set=pd.read_csv(filename),
                                             traj_id=col_names[0].strip(),
                                             datetime=col_names[1].strip(),
                                             latitude=col_names[2].strip(),
                                             longitude=col_names[3].strip())
                self._map_data = self._data
                # Set the table model and display the dataframe.
                self._table = QtWidgets.QTableView()

                # NOTE: whenever we update DFs, make sure to send the data after resetting
                #       index and setting inplace as False.
                self._model = TableModel(self._data.reset_index(inplace=False))
                self._table.setModel(self._model)
                self._window.DFPane.addWidget(self._table)
                self._window.run_stats_btn.setEnabled(True)
                self._window.statusBar.showMessage("Dataset Loaded Successfully.")
            else:
                self._window.open_file()

        except AttributeError or ValueError or TypeError:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setWindowTitle("Incorrect Column names")
            msg.setText("Incorrect Column names provided.\n"
                        "Please Enter the names again.")
            msg.exec()
            self.__init__(filename, self._window)

    def add_map(self):
        """
            Use folium to plot the trajectory on a map.

            Returns
            -------
                folium.folium.Map
                    The map with plotted trajectory.
        """
        # Get all the unique trajectory ids.
        ids_ = list(self._data.reset_index()['traj_id'].value_counts().keys())

        # Initiate the map placeholder.
        self.map = QtWebEngineWidgets.QWebEngineView()

        # Create the drop-down list for ID selection.
        self.traj_id_list = QtWidgets.QComboBox()
        self.traj_id_list.currentIndexChanged.connect(self.redraw_map)
        self.traj_id_list.addItems(ids_)

        # Add the drop-down and the map pane to the area.
        self._window.MapPane.addWidget(self.traj_id_list)
        self._window.MapPane.addWidget(self.map)

        # Actually draw the map.
        to_plot = self._map_data.reset_index().loc[self._map_data.reset_index()['traj_id']
                                                   == self.traj_id_list.currentText()]
        self._draw_map(to_plot)

    def _draw_map(self, to_plot):
        self.map.setHtml('')

        # This is the colorblind palette taken from seaborn.
        colors = ['#0173b2', '#de8f05', '#029e73', '#d55e00', '#cc78bc',
                  '#ca9161', '#fbafe4', '#949494', '#ece133', '#56b4e9']

        sw = to_plot[['lat', 'lon']].min().values.tolist()
        ne = to_plot[['lat', 'lon']].max().values.tolist()

        # Create a map with the initial point.
        map_ = folium.Map(location=(to_plot[const.LAT].iloc[0], to_plot[const.LONG].iloc[0]),
                          zoom_start=13, tiles='CartoDB positron')

        # Then, create (lat, lon) pairs for the data points.
        locations = []
        for j in range(len(to_plot)):
            locations.append((to_plot['lat'].iloc[j], to_plot['lon'].iloc[j]))

        # Create text frame.
        iframe = folium.IFrame(f'<font size="1px">Trajectory ID: {self.traj_id_list.currentText()} ' + '<br>' +
                               f'Latitude: {locations[0][0]}' + '<br>' +
                               f'Longitude: {locations[0][1]} </font>')

        # Create start and end markers for the trajectory.
        popup = folium.Popup(iframe, min_width=180, max_width=200, max_height=75)

        folium.Marker([to_plot['lat'].iloc[0], to_plot['lon'].iloc[0]],
                      color='green',
                      popup=popup,
                      marker_color='green',
                      icon=folium.Icon(icon_color='green', icon='play', prefix='fa')).add_to(map_)

        # Create text frame.
        iframe = folium.IFrame(f'<font size="1px">Trajectory ID: {self.traj_id_list.currentText()} ' + '<br>' +
                               f'Latitude: {locations[0][0]}' + '<br>' +
                               f'Longitude: {locations[0][1]} </font>')

        # Create start and end markers for the trajectory.
        popup = folium.Popup(iframe, min_width=180, max_width=200, max_height=75)

        folium.Marker([to_plot['lat'].iloc[-1], to_plot['lon'].iloc[-1]],
                      color='red',
                      popup=popup,
                      marker_color='red',
                      icon=folium.Icon(icon_color='red', icon='stop', prefix='fa')).add_to(map_)

        # Add trajectory to map.
        folium.PolyLine(locations,
                        color=colors[random.randint(0, len(colors)-1)],
                        ).add_to(map_)

        map_.fit_bounds([sw, ne])
        self.map.setHtml(map_.get_root().render())

    def redraw_map(self):
        to_plot = self._map_data.reset_index().loc[
            self._map_data.reset_index()['traj_id'] == self.traj_id_list.currentText()]
        self._draw_map(to_plot)

    def run_command(self):
        """
            When the user pushes the Run button, run the user's
            selected function on the dataset.

            Returns
            -------
                None
        """
        if self._window.feature_type.currentIndex() == 0:
            self._window.statusBar.showMessage("Running Filters ...")
            self._run_filters()
        elif self._window.feature_type.currentIndex() == 1:
            self._window.statusBar.showMessage("Running Interpolation ...")
            self._run_ip()
        elif self._window.feature_type.currentIndex() == 2:
            self._window.statusBar.showMessage("Running Kinematic Features ...")
            self._run_kinematic()
        elif self._window.feature_type.currentIndex() == 3:
            self._window.statusBar.showMessage("Running Statistics ...")
            self._run_stats()
        else:
            self._window.statusBar.showMessage("Running Temporal Features ...")
            self._run_temporal()

    def _run_ip(self):
        """
            Helper function for running the Interpolation commands
            from the GUI.

            Returns
            -------
                None
        """
        selected_function = self._window.listWidget.selectedItems()[0].text()

        if selected_function == 'Linear Interpolation':
            params = inspect.getfullargspec(Interpolation.interpolate_position).args
            params.remove('dataframe')
            params.remove('ip_type')
            args = self._get_input_params(params, title="Enter Parameters")

            if args:
                self._data = Interpolation.interpolate_position(dataframe=self._data,
                                                                ip_type='linear',
                                                                time_jump=float(args[0]))

        elif selected_function == 'Cubic Interpolation':
            params = inspect.getfullargspec(Interpolation.interpolate_position).args
            params.remove('dataframe')
            params.remove('ip_type')
            args = self._get_input_params(params, title="Enter Parameters")

            if args:
                self._data = Interpolation.interpolate_position(dataframe=self._data,
                                                                ip_type='cubic',
                                                                time_jump=float(args[0]))

        elif selected_function == 'Kinematic Interpolation':
            params = inspect.getfullargspec(Interpolation.interpolate_position).args
            params.remove('dataframe')
            params.remove('ip_type')
            args = self._get_input_params(params, title="Enter Parameters")

            if args:
                self._data = Interpolation.interpolate_position(dataframe=self._data,
                                                                ip_type='kinematic',
                                                                time_jump=float(args[0]))

        elif selected_function == 'Random-Walk Interpolation':
            params = inspect.getfullargspec(Interpolation.interpolate_position).args
            params.remove('dataframe')
            params.remove('ip_type')
            args = self._get_input_params(params, title="Enter Parameters")

            if args:
                self._data = Interpolation.interpolate_position(dataframe=self._data,
                                                                ip_type='random_walk',
                                                                time_jump=float(args[0]))

        self._window.statusBar.showMessage("Task Done ...")
        # Finally, update the GUI with the updated DF received from the
        # function results. DO NOT FORGET THE reset_index(inplace=False).
        self._model = TableModel(self._data.reset_index(inplace=False))
        self._table.setModel(self._model)

        # Also, update the map.
        self.redraw_map()

    def _run_kinematic(self):
        """
            Helper function for running the Kinematic commands
            from the GUI.

            Returns
            -------
                None
        """
        selected_function = self._window.listWidget.selectedItems()[0].text()

        # Based on the function selected by the user and whether it contains any
        # user given parameters, we will go ahead and run those features and update
        # the GUI with the results.

        if selected_function == 'All Kinematic Features':
            self._data = KinematicFeatures.generate_kinematic_features(self._data)
            self._window.statusBar.showMessage("Done ...")

        elif selected_function == 'Distance':
            self._data = KinematicFeatures.create_distance_column(self._data)

        elif selected_function == 'Distance from Start':
            self._data = KinematicFeatures.create_distance_from_start_column(self._data)

        elif selected_function == 'Point within Range':
            params = inspect.getfullargspec(KinematicFeatures.create_point_within_range_column).args
            params.remove('dataframe')
            args = self._get_input_params(params, title="Enter Parameters")

            # If the user provided the input params, then run the function, else
            # wait for the user to play their part.
            if args:
                # Use Regex to get all the digits from the coords input and convert
                # it to required tuple to feed into the method.
                temp = re.findall(r'\d+', args[0])
                coords = tuple(map(int, temp))

                dist_range = float(args[1])
                self._data = KinematicFeatures.create_point_within_range_column(self._data,
                                                                                coordinates=coords,
                                                                                dist_range=dist_range)

        elif selected_function == 'Distance from Co-ordinates':
            params = inspect.getfullargspec(KinematicFeatures.create_distance_from_point_column).args
            params.remove('dataframe')

            args = self._get_input_params(params, title="Enter Parameters")
            # If the user provided the input params, then run the function, else
            # wait for the user to play their part.
            if args:
                # Use Regex to get all the digits from the coords input and convert
                # it to required tuple to feed into the method.
                temp = re.findall(r'\d+', args[0])
                coords = tuple(map(int, temp))

                self._data = KinematicFeatures.create_distance_from_point_column(dataframe=self._data,
                                                                                 coordinates=coords, )

        elif selected_function == 'Speed':
            self._data = KinematicFeatures.create_speed_column(self._data)

        elif selected_function == 'Acceleration':
            self._data = KinematicFeatures.create_acceleration_column(self._data)

        elif selected_function == 'Jerk':
            self._data = KinematicFeatures.create_jerk_column(self._data)

        elif selected_function == 'Bearing':
            self._data = KinematicFeatures.create_bearing_column(self._data)

        elif selected_function == 'Bearing Rate':
            self._data = KinematicFeatures.create_bearing_rate_column(self._data)

        elif selected_function == 'Rate of Bearing Rate':
            self._data = KinematicFeatures.create_rate_of_br_column(self._data)

        # Finally, update the GUI with the updated DF received from the
        # function results. DO NOT FORGET THE reset_index(inplace=False).
        self._window.statusBar.showMessage("Task Done ...")
        self._model = TableModel(self._data.reset_index(inplace=False))
        self._table.setModel(self._model)

    def _run_temporal(self):
        """
            Helper function for running the Temporal commands
            from the GUI.

            Returns
            -------
                None
        """
        selected_function = self._window.listWidget.selectedItems()[0].text()

        if selected_function == "All Temporal Features":
            self._data = TemporalFeatures.generate_temporal_features(self._data)

        elif selected_function == "Date":
            self._data = TemporalFeatures.create_date_column(self._data)

        elif selected_function == "Time":
            self._data = TemporalFeatures.create_time_column(self._data)

        elif selected_function == "Day of the Week":
            self._data = TemporalFeatures.create_day_of_week_column(self._data)

        elif selected_function == "Weekend Indicator":
            self._data = TemporalFeatures.create_weekend_indicator_column(self._data)

        elif selected_function == "Time of Day":
            self._data = TemporalFeatures.create_time_of_day_column(self._data)

        # Finally, update the GUI with the updated DF received from the
        # function results. DO NOT FORGET THE reset_index(inplace=False).
        self._window.statusBar.showMessage("Task Done ...")
        self._model = TableModel(self._data.reset_index(inplace=False))
        self._table.setModel(self._model)

    def _run_filters(self):
        """
            Helper function for running the Filter commands
            from the GUI.

            Returns
            -------
                None
        """
        selected_function = self._window.listWidget.selectedItems()[0].text()

        if selected_function == 'Hampel Filter':
            params = inspect.getfullargspec(Filters.hampel_outlier_detection).args
            params.remove('dataframe')

            args = self._get_input_params(params, title="Enter Parameters")
            # If the user provided the input params, then run the function, else
            # wait for the user to play their part.
            if args:
                self._data = Filters.hampel_outlier_detection(dataframe=self._data,
                                                              column_name=args[0])

        elif selected_function == 'Remove Duplicates':
            self._data = Filters.remove_duplicates(self._data)

        elif selected_function == 'By Trajectory ID':
            params = inspect.getfullargspec(Filters.filter_by_traj_id).args
            params.remove('dataframe')

            args = self._get_input_params(params, title="Enter Parameters")
            # If the user provided the input params, then run the function, else
            # wait for the user to play their part.
            if args:
                self._data = Filters.filter_by_traj_id(dataframe=self._data,
                                                       traj_id=args[0])

        elif selected_function == 'By Bounding Box':
            params = inspect.getfullargspec(Filters.filter_by_bounding_box).args
            params.remove('dataframe')

            args = self._get_input_params(params, title="Enter Parameters")
            # If the user provided the input params, then run the function, else
            # wait for the user to play their part.
            if args:
                temp = re.findall("\d+\.\d+", args[0])
                coords = tuple(map(float, temp))
                self._data = Filters.filter_by_bounding_box(dataframe=self._data,
                                                            bounding_box=coords,
                                                            inside=(bool(distutils.util.strtobool(args[1]))))

        elif selected_function == 'By Date':
            params = inspect.getfullargspec(Filters.filter_by_date).args
            params.remove('dataframe')

            args = self._get_input_params(params, title="Enter Parameters")
            # If the user provided the input params, then run the function, else
            # wait for the user to play their part.
            if args:
                if args[0] == '':
                    self._data = Filters.filter_by_date(dataframe=self._data,
                                                        end_date=args[1])
                elif args[1] == '':
                    self._data = Filters.filter_by_date(dataframe=self._data,
                                                        start_date=args[0])
                else:
                    self._data = Filters.filter_by_date(dataframe=self._data,
                                                        start_date=args[0],
                                                        end_date=args[1])

        elif selected_function == 'By DateTime':
            params = inspect.getfullargspec(Filters.filter_by_datetime).args
            params.remove('dataframe')

            args = self._get_input_params(params, title="Enter Parameters")
            # If the user provided the input params, then run the function, else
            # wait for the user to play their part.
            if args:
                if args[0] == '':
                    self._data = Filters.filter_by_datetime(dataframe=self._data,
                                                            end_dateTime=args[1])
                elif args[1] == '':
                    self._data = Filters.filter_by_datetime(dataframe=self._data,
                                                            start_dateTime=args[0])
                else:
                    self._data = Filters.filter_by_datetime(dataframe=self._data,
                                                            start_dateTime=args[0],
                                                            end_dateTime=args[1])

        elif selected_function == 'By Maximum Speed':
            params = inspect.getfullargspec(Filters.filter_by_max_speed).args
            params.remove('dataframe')

            args = self._get_input_params(params, title="Enter Parameters")
            # If the user provided the input params, then run the function, else
            # wait for the user to play their part.
            if args:
                self._data = Filters.filter_by_max_speed(dataframe=self._data,
                                                         max_speed=float(args[0]))

        elif selected_function == 'By Minimum Speed':
            params = inspect.getfullargspec(Filters.filter_by_min_speed).args
            params.remove('dataframe')

            args = self._get_input_params(params, title="Enter Parameters")
            # If the user provided the input params, then run the function, else
            # wait for the user to play their part.
            if args:
                self._data = Filters.filter_by_min_speed(dataframe=self._data,
                                                         min_speed=float(args[0]))

        elif selected_function == 'By Minimum Consecutive Distance':
            params = inspect.getfullargspec(Filters.filter_by_min_consecutive_distance).args
            params.remove('dataframe')

            args = self._get_input_params(params, title="Enter Parameters")
            # If the user provided the input params, then run the function, else
            # wait for the user to play their part.
            if args:
                self._data = Filters.filter_by_min_consecutive_distance(dataframe=self._data,
                                                                        min_distance=float(args[0]))

        elif selected_function == 'By Maximum Consecutive Distance':
            params = inspect.getfullargspec(Filters.filter_by_max_consecutive_distance).args
            params.remove('dataframe')

            args = self._get_input_params(params, title="Enter Parameters")
            # If the user provided the input params, then run the function, else
            # wait for the user to play their part.
            if args:
                self._data = Filters.filter_by_max_consecutive_distance(dataframe=self._data,
                                                                        max_distance=float(args[0]))

        elif selected_function == 'By Maximum Distance and Speed':
            params = inspect.getfullargspec(Filters.filter_by_max_distance_and_speed).args
            params.remove('dataframe')

            args = self._get_input_params(params, title="Enter Parameters")
            # If the user provided the input params, then run the function, else
            # wait for the user to play their part.
            if args:
                self._data = Filters.filter_by_max_distance_and_speed(dataframe=self._data,
                                                                      max_distance=float(args[0]),
                                                                      max_speed=float(args[1]))

        elif selected_function == 'By Minimum Distance and Speed':
            params = inspect.getfullargspec(Filters.filter_by_min_distance_and_speed).args
            params.remove('dataframe')

            args = self._get_input_params(params, title="Enter Parameters")
            # If the user provided the input params, then run the function, else
            # wait for the user to play their part.
            if args:
                self._data = Filters.filter_by_min_distance_and_speed(dataframe=self._data,
                                                                      min_distance=float(args[0]),
                                                                      min_speed=float(args[1]))

        elif selected_function == 'Remove Outliers by Consecutive Distance':
            self._data = Filters.filter_outliers_by_consecutive_distance(dataframe=self._data)

        elif selected_function == 'Remove Outliers by Consecutive Speed':
            self._data = Filters.filter_outliers_by_consecutive_speed(dataframe=self._data)

        elif selected_function == 'Remove Trajectories with Less Points':
            params = inspect.getfullargspec(Filters.remove_trajectories_with_less_points).args
            params.remove('dataframe')

            args = self._get_input_params(params, title="Enter Parameters")
            # If the user provided the input params, then run the function, else
            # wait for the user to play their part.
            if args:
                self._data = Filters.remove_trajectories_with_less_points(dataframe=self._data,
                                                                          num_min_points=int(args[0]))

        # Finally, update the GUI with the updated DF received from the
        # function results. DO NOT FORGET THE reset_index(inplace=False).
        self._window.statusBar.showMessage("Task Done ...")
        self._model = TableModel(self._data.reset_index(inplace=False))
        self._table.setModel(self._model)

    def _run_stats(self):
        """
            Helper function for running the Statistical commands
            from the GUI.

            Returns
            -------
                None
        """
        selected_function = self._window.listWidget.selectedItems()[0].text()

        if selected_function == 'Segment Trajectories':
            params = inspect.getfullargspec(Statistics.segment_traj_by_days).args
            params.remove('dataframe')

            args = self._get_input_params(params, title="Enter Parameters")
            # If the user provided the input params, then run the function, else
            # wait for the user to play their part.
            if args:
                self._data = Statistics.segment_traj_by_days(dataframe=self._data,
                                                             num_days=int(args[0]))

        elif selected_function == 'Generate Kinematic Statistics':
            params = inspect.getfullargspec(Statistics.generate_kinematic_stats).args
            params.remove('dataframe')

            args = self._get_input_params(params, title="Enter Parameters")
            # If the user provided the input params, then run the function, else
            # wait for the user to play their part.
            if args:
                self._data = Statistics.generate_kinematic_stats(dataframe=self._data,
                                                                 target_col_name=args[0],
                                                                 segmented=bool(distutils.util.strtobool(args[1])))

        elif selected_function == 'Pivot Statistics DF':
            params = inspect.getfullargspec(Statistics.pivot_stats_df).args
            params.remove('dataframe')

            args = self._get_input_params(params, title="Enter Parameters")
            # If the user provided the input params, then run the function, else
            # wait for the user to play their part.
            if args:
                self._data = Statistics.pivot_stats_df(dataframe=self._data,
                                                       target_col_name=args[0],
                                                       segmented=bool(distutils.util.strtobool(args[1])))

        # Finally, update the GUI with the updated DF received from the
        # function results. DO NOT FORGET THE reset_index(inplace=False).
        self._window.statusBar.showMessage("Task Done ...")
        self._model = TableModel(self._data.reset_index(inplace=False))
        self._table.setModel(self._model)

    def _get_input_params(self, labels, title):
        """
            Take the input parameters for the function in question from
            the user.

            Parameters
            ----------
                labels: list
                    The name of the parameters.
                title: str
                    The title of the input dialog box.

            Returns
            -------
                list:
                    A list containing the user input.
        """
        input_dialog = InputDialog(parent=self._window,
                                   labels=labels,
                                   title=title)
        if input_dialog.exec_():
            args = input_dialog.getInputs()

            return args
