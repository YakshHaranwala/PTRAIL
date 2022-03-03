"""
    This class is used to connect the PTRAIL GUI to PTRAIL
    backend. All the GUI's functionalities are handled in this
    class.

    | Authors: Yaksh J Haranwala, Salman Haidri
"""
from distutils import util
import random
import re

import folium
import inspect
import pandas as pd

# GUI Imports.
from PyQt5 import QtWidgets, QtWebEngineWidgets, QtGui, QtCore
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

# Statistics imports.
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from sklearn.feature_selection import mutual_info_classif


class GuiHandler:
    def __init__(self, filename, window):
        self.generateStats = False
        self.generateFeatureImportanceBtn = False
        self.statCanvas = None
        self.statFigure = None

        self.featureCanvas = None
        self.featureFigure = None

        self.ax = None
        self.traj_id_list = None

        self.map = None
        self._window = window
        self._data = None
        self._map_data = None
        self._model = None
        self._table = None

        self.display_df(filename=filename)

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
                                               title="Enter Column Names",
                                               placeHolder=['Name of Identifier column', 'Name of Timestamp column',
                                                            'Name of Latitude column', 'Name of Longitude column'])
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
                self._window.add_df_controller()
                self._window.DFPane.addWidget(self._table)
                self._window.statusBar.showMessage("Dataset Loaded Successfully.")

                # Get all the unique trajectory ids.
                ids_ = list(self._data.reset_index()['traj_id'].value_counts().keys())

                # Initiate the map placeholder.
                self.map = QtWebEngineWidgets.QWebEngineView()

                # Create the drop-down list for ID selection.
                self.traj_id_list = QtWidgets.QComboBox()
                self.traj_id_list.setFont(QtGui.QFont('Tahoma', 12))
                self.traj_id_list.addItems(ids_)

                # Add the drop-down and the map pane to the area.
                self._window.MapPane.addWidget(self.traj_id_list)
                self._window.MapPane.addWidget(self.map)

                # Actually draw the map.
                to_plot = self._map_data.reset_index().loc[self._map_data.reset_index()['traj_id']
                                                           == self.traj_id_list.currentText()]
                self._window.open_btn.deleteLater()
                self._draw_map(to_plot)
                self.draw_stats()
                self.add_column_drop_widget()
                self._window.runStatsBtn.setEnabled(True)
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
                               f'Latitude: {locations[-1][0]}' + '<br>' +
                               f'Longitude: {locations[-1][1]} </font>')

        # Create start and end markers for the trajectory.
        popup = folium.Popup(iframe, min_width=180, max_width=200, max_height=75)

        folium.Marker([to_plot['lat'].iloc[-1], to_plot['lon'].iloc[-1]],
                      color='red',
                      popup=popup,
                      marker_color='red',
                      icon=folium.Icon(icon_color='red', icon='stop', prefix='fa')).add_to(map_)

        # Add trajectory to map.
        folium.PolyLine(locations,
                        color=colors[random.randint(0, len(colors) - 1)],
                        ).add_to(map_)

        map_.fit_bounds([sw, ne])
        self.traj_id_list.currentIndexChanged.connect(lambda: self.redraw_map())
        self.map.setHtml(map_.get_root().render())

    def redraw_map(self):
        """
            Redraw the map when the traj_id is changed from the DropDown list.
        """
        # Check whether the QComboBox is empty or not. If so, don't redraw.
        # NOTE: This is done specifically to handle the case of filtering where
        #       some trajectories might be filtered out, and we have to update the
        #       id selection list.
        if self.traj_id_list.currentText() and self.traj_id_list.currentText() != '' and len(self._data) > 0:
            to_plot = self._map_data.reset_index().loc[
                self._map_data.reset_index()['traj_id'] == self.traj_id_list.currentText()]
            self._draw_map(to_plot)
            if self.generateStats:
                self.redraw_stat()

    def draw_stats(self):
        """
            Handle the objects of the statistics pane from here.
        """
        # Create the Stat Selection Drop down button.
        self._window.selectStatDropdown = QtWidgets.QComboBox()
        self._window.selectStatDropdown.currentIndexChanged.connect(lambda: self.redraw_stat())
        self._window.selectStatDropdown.setFont(QtGui.QFont("Tahoma", 12))

        # Create the matplotlib figure and axis for stats.
        self.statFigure = plt.figure()
        self.statCanvas = FigureCanvas(self.statFigure)
        self.statCanvas.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        )

        # Create the matplotlib figure and axis for feature importance.
        self.featureFigure = plt.figure()
        self.featureCanvas = FigureCanvas(self.featureFigure)
        self.featureCanvas.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        )

        # Add the figure and canvas to a new layout and finally add the layout to the StatsPane.
        self.generateFeatureImportanceBtn = QtWidgets.QPushButton("Generate Feature Importance")
        self.generateFeatureImportanceBtn.clicked.connect(lambda: self.generate_feature_imp_plot())
        self.generateFeatureImportanceBtn.setFont(QtGui.QFont("Tahoma", 12))
        self.generateFeatureImportanceBtn.setEnabled(False)
        new_draw_layout = QtWidgets.QVBoxLayout()
        new_draw_layout.addWidget(self._window.selectStatDropdown)
        new_draw_layout.addWidget(self.statCanvas)
        new_draw_layout.addWidget(self.featureCanvas)
        new_draw_layout.addWidget(self.generateFeatureImportanceBtn)
        self._window.StatsPane.addLayout(new_draw_layout)

    def generate_feature_imp_plot(self):
        """
            Take the input from the user and draw the mutual info
            plot.
        """
        try:
            args = self._get_input_params(['Class-Label Column', 'Number of Features', 'Segment Based'],
                                          title="Enter Classification Target column name",
                                          placeHolder=['Classification target column name', 'Number Features Wanted',
                                                       'True/False'])
            if args:
                feat, importance = None, None
                # If the user wants mutual info for point-based data, use the if-block.
                # For segment-based data, the else block is used.
                if not bool(util.strtobool(args[2].strip())):
                    # Generate the features and drop duplicate cols.
                    df = KinematicFeatures.generate_kinematic_features(self._data).dropna()
                    df = df.loc[:, ~df.columns.duplicated()]
                    df = df[[
                        'lat', 'lon', 'Distance', 'Distance_from_start', 'Speed', 'Acceleration', 'Jerk',
                        'Bearing', 'Bearing_Rate', 'Rate_of_bearing_rate', args[0]
                    ]]
                    Y = df[args[0].strip()]
                    df = df.drop(columns=[args[0]])
                else:
                    df = Statistics.generate_kinematic_stats(self._data, args[0], False)
                    df = Statistics.pivot_stats_df(df, args[0], False)
                    df = df.loc[:, ~df.columns.duplicated()]
                    Y = df[[args[0].strip()]]
                    df = df.drop(columns=[args[0]])

                # Check whether the number of features that the user asked is lesser
                # than actual number of features.
                num_features = int(args[1]) if len(df.columns) > int(args[1]) else len(df.columns)

                # Predict the mutual info.
                importance = mutual_info_classif(df, Y)
                feat = pd.Series(importance, df.columns).sort_values(ascending=False)
                feat = feat[0: num_features]

                # Plot the bar plots to the tool.
                self.featureFigure.clear()
                ax = self.featureFigure.add_subplot(111)

                feat.plot(kind='barh', color='skyblue', ax=ax)
                ax.set_title("Classification Feature Importance")
                ax.set_xlabel("Mutual Info")

                self.featureFigure.tight_layout()
                self.featureCanvas.draw()
                self.generateFeatureImportanceBtn.setText("Refresh Importance")
        except:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setWindowTitle("Not a classification problem")
            msg.setText("Feature importance is only shown for datasets that have a "
                        "target column.")
            msg.exec()

    def redraw_stat(self):
        """
            Redraw the statistics plot when the user changes the option from
            the Dropdown menu.
        """
        selected_stat = self._window.selectStatDropdown.currentText()
        stat_data = self._data.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        colors = sns.color_palette('tab10')

        # Get the necessary stats out.
        percent_10 = stat_data[selected_stat]['10%']
        percent_25 = stat_data[selected_stat]['25%']
        percent_50 = stat_data[selected_stat]['50%']
        percent_75 = stat_data[selected_stat]['75%']
        percent_90 = stat_data[selected_stat]['90%']
        avg = stat_data[selected_stat]['mean']

        # Clear the figure and add an axes to it.
        self.statFigure.clear()
        ax = self.statFigure.add_subplot(111)

        # Attributes for the plot.
        text = ['10%', '25%', 'Median', '75%', '90%', 'Mean']
        horizontal_lines = [percent_10, percent_25, percent_50, percent_75, percent_90, avg]

        # Plot the line-plot and the stat lines.
        one_animal = self._data.reset_index().loc[self._data.reset_index()['traj_id']
                                                  == self.traj_id_list.currentText()]
        sns.lineplot(data=one_animal.reset_index(), x='DateTime', y=selected_stat, ax=ax, color='skyblue')
        for i in range(len(horizontal_lines)):
            ax.axhline(horizontal_lines[i], c=colors[i], linestyle='--', label=text[i])

        # Add the legend, rotate the ticks, set tight layout and draw it on the canvas.
        ax.legend(loc='upper right')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        self.statFigure.tight_layout()
        self.statCanvas.draw()

    def add_column_drop_widget(self):
        """
            Add a List Widget to drop columns from the dataset. This
            widget is added to the CommandPalette.

            Note
            ----
                It is to be noted that the following columns are mandatory for
                PTrailDataFrame:
                    | 1. traj_id
                    | 2. DateTime
                    | 3. lat
                    | 4. lon

                Hence, these columns are not presented as options for deletion.
        """
        # A layout for containing the drop column setup.
        small_layout = QtWidgets.QVBoxLayout()

        # Create the list widget and add the options.
        self._window.dropColumnWidget = QtWidgets.QListWidget()
        self._window.dropColumnWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self._window.dropColumnWidget.setFont(QtGui.QFont("Tahoma", 12))
        self._window.dropColumnWidget.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored))
        to_add = list(self._data.columns)
        to_add.remove('lat')
        to_add.remove('lon')
        self._window.dropColumnWidget.addItems(to_add)
        small_layout.addWidget(self._window.dropColumnWidget)

        # Add the button to drop the column.
        self._window.dropColumnBtn = QtWidgets.QPushButton("Drop Column(s)")
        self._window.dropColumnBtn.setFont(QtGui.QFont("Tahoma", 12))
        self._window.dropColumnBtn.clicked.connect(lambda: self.drop_col())
        small_layout.addWidget(self._window.dropColumnBtn)

        # Add the VBoxLayout to the Command Palette.
        self._window.CommandPalette.addLayout(small_layout)

    def run_command(self):
        """
            When the user pushes the Run button, run the user's
            selected function on the dataset.

            Returns
            -------
                None
        """
        if self._window.featureType.currentIndex() == 0:
            self._window.statusBar.showMessage("Running Filters ...")
            self._run_filters()
        elif self._window.featureType.currentIndex() == 1:
            self.generateStats = False
            self._window.statusBar.showMessage("Running Interpolation ...")
            self._run_ip()
        elif self._window.featureType.currentIndex() == 2:
            self.generateStats = True
            self._window.statusBar.showMessage("Running Kinematic Features ...")
            self._run_kinematic()
        elif self._window.featureType.currentIndex() == 3:
            self._window.statusBar.showMessage("Running Statistics ...")
            self._run_stats()
        else:
            self._window.statusBar.showMessage("Running Temporal Features ...")
            self._run_temporal()
        self.update_dropCol_options()

    def _run_ip(self):
        """
            Helper function for running the Interpolation commands
            from the GUI.

            Returns
            -------
                None
        """
        selected_function = self._window.featureListWidget.selectedItems()[0].text()

        if selected_function == 'Linear Interpolation':
            params = inspect.getfullargspec(Interpolation.interpolate_position).args
            params.remove('dataframe')
            params.remove('ip_type')
            args = self._get_input_params(params, title="Enter Parameters",
                                          placeHolder=['Sampling rate (in seconds)',
                                                       'Name of the column that contains class label (Leave empty if '
                                                       'none)'])

            if args:
                self._data = Interpolation.interpolate_position(dataframe=self._data,
                                                                ip_type='linear',
                                                                sampling_rate=float(args[0].strip()),
                                                                class_label_col=args[1].strip())

        elif selected_function == 'Cubic Interpolation':
            params = inspect.getfullargspec(Interpolation.interpolate_position).args
            params.remove('dataframe')
            params.remove('ip_type')
            args = self._get_input_params(params, title="Enter Parameters",
                                          placeHolder=['Sampling rate (in seconds)',
                                                       'Name of the column that contains class label (Leave empty if '
                                                       'none)']
                                          )

            if args:
                self._data = Interpolation.interpolate_position(dataframe=self._data,
                                                                ip_type='cubic',
                                                                sampling_rate=float(args[0].strip()),
                                                                class_label_col=args[1].strip())

        elif selected_function == 'Kinematic Interpolation':
            params = inspect.getfullargspec(Interpolation.interpolate_position).args
            params.remove('dataframe')
            params.remove('ip_type')
            args = self._get_input_params(params, title="Enter Parameters",
                                          placeHolder=['Sampling rate (in seconds)',
                                                       'Name of the column that contains class label (Leave empty if '
                                                       'none)'])

            if args:
                self._data = Interpolation.interpolate_position(dataframe=self._data,
                                                                ip_type='kinematic',
                                                                sampling_rate=float(args[0].strip()),
                                                                class_label_col=args[1].strip())

        elif selected_function == 'Random-Walk Interpolation':
            params = inspect.getfullargspec(Interpolation.interpolate_position).args
            params.remove('dataframe')
            params.remove('ip_type')
            args = self._get_input_params(params, title="Enter Parameters",
                                          placeHolder=['Sampling rate (in seconds)',
                                                       'Name of the column that contains class label (Leave empty if '
                                                       'none)']
                                          )

            if args:
                self._data = Interpolation.interpolate_position(dataframe=self._data,
                                                                ip_type='random_walk',
                                                                sampling_rate=float(args[0].strip()),
                                                                class_label_col=args[1].strip())

        self._window.statusBar.showMessage("Task Done ...")

        # Finally, update the GUI with the updated DF received from the
        # function results. DO NOT FORGET THE reset_index(inplace=False).
        self._model = TableModel(self._data.reset_index(inplace=False))
        self._table.setModel(self._model)

        # Also, update the map.
        self._map_data = self._data
        self._window.selectStatDropdown.blockSignals(True)
        self._window.selectStatDropdown.clear()
        self.statFigure.clear()
        self.statCanvas.draw()
        self._window.selectStatDropdown.blockSignals(False)
        self.redraw_map()

    def _run_kinematic(self):
        """
            Helper function for running the Kinematic commands
            from the GUI.

            Returns
            -------
                None
        """
        selected_function = self._window.featureListWidget.selectedItems()[0].text()

        # Based on the function selected by the user and whether it contains any
        # user given parameters, we will go ahead and run those features and update
        # the GUI with the results.

        if selected_function == 'All Kinematic Features':
            self._data = KinematicFeatures.generate_kinematic_features(self._data)
            self._window.statusBar.showMessage("Done ...")

            # Update the select-stats dropdown.
            self._window.selectStatDropdown.blockSignals(True)
            self._window.selectStatDropdown.clear()
            self._window.selectStatDropdown.addItems([
                'Distance', 'Distance_from_start', 'Speed', 'Acceleration', 'Jerk',
                'Bearing', 'Bearing_Rate', 'Rate_of_bearing_rate',
            ])
            self.generateFeatureImportanceBtn.setEnabled(True)
            self._window.selectStatDropdown.blockSignals(False)

        elif selected_function == 'Distance':
            self._data = KinematicFeatures.create_distance_column(self._data)
            self._window.selectStatDropdown.addItems(['Distance'])

        elif selected_function == 'Distance from Start':
            self._data = KinematicFeatures.create_distance_from_start_column(self._data)
            self._window.selectStatDropdown.addItems(['Distance_from_start'])

        elif selected_function == 'Point within Range':
            params = inspect.getfullargspec(KinematicFeatures.create_point_within_range_column).args
            params.remove('dataframe')
            args = self._get_input_params(params, title="Enter Parameters",
                                          placeHolder=['Coordinates (lat1, lon1, lat2, lon2)',
                                                       'Max distance from coordinates (metres)'])

            # If the user provided the input params, then run the function, else
            # wait for the user to play their part.
            if args:
                # Use Regex to get all the digits from the coords input and convert
                # it to required tuple to feed into the method.
                temp = re.findall(r'\d+', args[0].strip())
                coords = tuple(map(int, temp))

                dist_range = float(args[1].strip())
                self._data = KinematicFeatures.create_point_within_range_column(self._data,
                                                                                coordinates=coords,
                                                                                dist_range=dist_range)

        elif selected_function == 'Distance from Co-ordinates':
            params = inspect.getfullargspec(KinematicFeatures.create_distance_from_point_column).args
            params.remove('dataframe')

            args = self._get_input_params(params, title="Enter Parameters",
                                          placeHolder=['Coordinates (lat1, lon1, lat2, lon2)'])
            # If the user provided the input params, then run the function, else
            # wait for the user to play their part.
            if args:
                # Use Regex to get all the digits from the coords input and convert
                # it to required tuple to feed into the method.
                temp = re.findall(r'\d+', args[0].strip())
                coords = tuple(map(int, temp))

                self._data = KinematicFeatures.create_distance_from_point_column(dataframe=self._data,
                                                                                 coordinates=coords)

        elif selected_function == 'Speed':
            self._data = KinematicFeatures.create_speed_column(self._data)
            self._window.selectStatDropdown.addItems(['Speed'])

        elif selected_function == 'Acceleration':
            self._data = KinematicFeatures.create_acceleration_column(self._data)
            self._window.selectStatDropdown.addItems(['Acceleration'])

        elif selected_function == 'Jerk':
            self._data = KinematicFeatures.create_jerk_column(self._data)
            self._window.selectStatDropdown.addItems(['Jerk'])

        elif selected_function == 'Bearing':
            self._data = KinematicFeatures.create_bearing_column(self._data)
            self._window.selectStatDropdown.addItems(['Bearing'])

        elif selected_function == 'Bearing Rate':
            self._data = KinematicFeatures.create_bearing_rate_column(self._data)
            self._window.selectStatDropdown.addItems(['Bearing_Rate'])

        elif selected_function == 'Rate of Bearing Rate':
            self._data = KinematicFeatures.create_rate_of_br_column(self._data)
            self._window.selectStatDropdown.addItems(['Rate_of_bearing_rate'])

        # Redraw the stats panel.
        self.redraw_stat()

        # Finally, update the GUI with the updated DF received from the
        # function results. DO NOT FORGET THE reset_index(inplace=False).
        self._map_data = self._data
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
        selected_function = self._window.featureListWidget.selectedItems()[0].text()

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
        self._map_data = self._data
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
        selected_function = self._window.featureListWidget.selectedItems()[0].text()

        if selected_function == 'Hampel Filter':
            params = inspect.getfullargspec(Filters.hampel_outlier_detection).args
            params.remove('dataframe')

            args = self._get_input_params(params, title="Enter Parameters",
                                          placeHolder=['Filter by Metric (Enter Column Name)'])
            # If the user provided the input params, then run the function, else
            # wait for the user to play their part.
            if args:
                self._data = Filters.hampel_outlier_detection(dataframe=self._data,
                                                              column_name=args[0].strip())

        elif selected_function == 'Remove Duplicates':
            self._data = Filters.remove_duplicates(self._data)

        elif selected_function == 'By Trajectory ID':
            params = inspect.getfullargspec(Filters.filter_by_traj_id).args
            params.remove('dataframe')

            args = self._get_input_params(params, title="Enter Parameters",
                                          placeHolder=['Trajectory ID'])
            # If the user provided the input params, then run the function, else
            # wait for the user to play their part.
            if args:
                self._data = Filters.filter_by_traj_id(dataframe=self._data,
                                                       traj_id=args[0].strip())
        elif selected_function == 'By Bounding Box':
            params = inspect.getfullargspec(Filters.filter_by_bounding_box).args
            params.remove('dataframe')

            args = self._get_input_params(params, title="Enter Parameters",
                                          placeHolder=["lat1, lon1, lat2, lon2",
                                                       "Points inside the bounding box? (True/False)"])
            # If the user provided the input params, then run the function, else
            # wait for the user to play their part.
            if args:
                temp = re.findall("\d+\.\d+", args[0])
                coords = tuple(map(float, temp))
                self._data = Filters.filter_by_bounding_box(dataframe=self._data,
                                                            bounding_box=coords,
                                                            inside=(bool(util.strtobool(args[1].strip()))))

        elif selected_function == 'By Date':
            params = inspect.getfullargspec(Filters.filter_by_date).args
            params.remove('dataframe')

            args = self._get_input_params(params, title="Enter Parameters",
                                          placeHolder=["YYYY-MM-DD", "YYYY-MM-DD"])
            # If the user provided the input params, then run the function, else
            # wait for the user to play their part.
            if args:
                if args[0] == '':
                    self._data = Filters.filter_by_date(dataframe=self._data,
                                                        end_date=args[1].strip())
                elif args[1] == '':
                    self._data = Filters.filter_by_date(dataframe=self._data,
                                                        start_date=args[0].strip())
                else:
                    self._data = Filters.filter_by_date(dataframe=self._data,
                                                        start_date=args[0].strip(),
                                                        end_date=args[1].strip())

        elif selected_function == 'By DateTime':
            params = inspect.getfullargspec(Filters.filter_by_datetime).args
            params.remove('dataframe')

            args = self._get_input_params(params, title="Enter Parameters",
                                          placeHolder=['YYYY-MM-DD hh:mm:ss', 'YYYY-MM-DD hh:mm:ss'])
            # If the user provided the input params, then run the function, else
            # wait for the user to play their part.
            if args:
                if args[0] == '':
                    self._data = Filters.filter_by_datetime(dataframe=self._data,
                                                            end_dateTime=args[1].strip())
                elif args[1] == '':
                    self._data = Filters.filter_by_datetime(dataframe=self._data,
                                                            start_dateTime=args[0].strip())
                else:
                    self._data = Filters.filter_by_datetime(dataframe=self._data,
                                                            start_dateTime=args[0].strip(),
                                                            end_dateTime=args[1].strip())

        elif selected_function == 'By Maximum Speed':
            params = inspect.getfullargspec(Filters.filter_by_max_speed).args
            params.remove('dataframe')

            args = self._get_input_params(params, title="Enter Parameters",
                                          placeHolder=['Maximum speed between consecutive points (metres/second)'])
            # If the user provided the input params, then run the function, else
            # wait for the user to play their part.
            if args:
                self._data = Filters.filter_by_max_speed(dataframe=self._data,
                                                         max_speed=float(args[0].strip()))

        elif selected_function == 'By Minimum Speed':
            params = inspect.getfullargspec(Filters.filter_by_min_speed).args
            params.remove('dataframe')

            args = self._get_input_params(params, title="Enter Parameters",
                                          placeHolder=['Minimum speed between consecutive points (metres/second)'])
            # If the user provided the input params, then run the function, else
            # wait for the user to play their part.
            if args:
                self._data = Filters.filter_by_min_speed(dataframe=self._data,
                                                         min_speed=float(args[0].strip()))

        elif selected_function == 'By Minimum Consecutive Distance':
            params = inspect.getfullargspec(Filters.filter_by_min_consecutive_distance).args
            params.remove('dataframe')

            args = self._get_input_params(params, title="Enter Parameters",
                                          placeHolder=['Minimum Distance between 2 points (metres)'])
            # If the user provided the input params, then run the function, else
            # wait for the user to play their part.
            if args:
                self._data = Filters.filter_by_min_consecutive_distance(dataframe=self._data,
                                                                        min_distance=float(args[0].strip()))

        elif selected_function == 'By Maximum Consecutive Distance':
            params = inspect.getfullargspec(Filters.filter_by_max_consecutive_distance).args
            params.remove('dataframe')

            args = self._get_input_params(params, title="Enter Parameters",
                                          placeHolder=['Max Distance between 2 points (metres)'])
            # If the user provided the input params, then run the function, else
            # wait for the user to play their part.
            if args:
                self._data = Filters.filter_by_max_consecutive_distance(dataframe=self._data,
                                                                        max_distance=float(args[0].strip()))

        elif selected_function == 'By Maximum Distance and Speed':
            params = inspect.getfullargspec(Filters.filter_by_max_distance_and_speed).args
            params.remove('dataframe')

            args = self._get_input_params(params, title="Enter Parameters",
                                          placeHolder=['Max Distance (metres)', 'Max Speed (metres/second)'])
            # If the user provided the input params, then run the function, else
            # wait for the user to play their part.
            if args:
                self._data = Filters.filter_by_max_distance_and_speed(dataframe=self._data,
                                                                      max_distance=float(args[0].strip()),
                                                                      max_speed=float(args[1].strip()))

        elif selected_function == 'By Minimum Distance and Speed':
            params = inspect.getfullargspec(Filters.filter_by_min_distance_and_speed).args
            params.remove('dataframe')

            args = self._get_input_params(params, title="Enter Parameters",
                                          placeHolder=['Minimum Distance (metres)', 'Minimum Speed (metres/second)'])
            # If the user provided the input params, then run the function, else
            # wait for the user to play their part.
            if args:
                self._data = Filters.filter_by_min_distance_and_speed(dataframe=self._data,
                                                                      min_distance=float(args[0].strip()),
                                                                      min_speed=float(args[1].strip()))

        elif selected_function == 'Remove Outliers by Consecutive Distance':
            self._data = Filters.filter_outliers_by_consecutive_distance(dataframe=self._data)

        elif selected_function == 'Remove Outliers by Consecutive Speed':
            self._data = Filters.filter_outliers_by_consecutive_speed(dataframe=self._data)

        elif selected_function == 'Remove Trajectories with Less Points':
            params = inspect.getfullargspec(Filters.remove_trajectories_with_less_points).args
            params.remove('dataframe')

            args = self._get_input_params(params, title="Enter Parameters",
                                          placeHolder=['Number of Minimum points in the trajectory to avoid removal'])
            # If the user provided the input params, then run the function, else
            # wait for the user to play their part.
            if args:
                self._data = Filters.remove_trajectories_with_less_points(dataframe=self._data,
                                                                          num_min_points=int(args[0].strip()))

        # Update the traj_id list in case if some trajectories have been completely
        # removed.
        self._map_data = self._data
        self.traj_id_list.clear()
        self.traj_id_list.addItems(list(self._data.reset_index()['traj_id'].value_counts().keys()))

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
        selected_function = self._window.featureListWidget.selectedItems()[0].text()

        if selected_function == 'Segment Trajectories':
            params = inspect.getfullargspec(Statistics.segment_traj_by_days).args
            params.remove('dataframe')

            args = self._get_input_params(params, title="Enter Parameters",
                                          placeHolder=['Duration of each segment in days'])
            # If the user provided the input params, then run the function, else
            # wait for the user to play their part.
            if args:
                self._data = Statistics.segment_traj_by_days(dataframe=self._data,
                                                             num_days=int(args[0].strip()))
                self._map_data = self._data
        elif selected_function == 'Generate Kinematic Statistics':
            params = inspect.getfullargspec(Statistics.generate_kinematic_stats).args
            params.remove('dataframe')

            args = self._get_input_params(params, title="Enter Parameters",
                                          placeHolder=['Class-label column name', 'Is the trajectory segmented?'])
            # If the user provided the input params, then run the function, else
            # wait for the user to play their part.
            if args:
                self._data = Statistics.generate_kinematic_stats(dataframe=self._data,
                                                                 target_col_name=args[0].strip(),
                                                                 segmented=bool(util.strtobool(args[1].strip())))

        elif selected_function == 'Pivot Statistics DF':
            params = inspect.getfullargspec(Statistics.pivot_stats_df).args
            params.remove('dataframe')

            args = self._get_input_params(params, title="Enter Parameters",
                                          placeHolder=['Class-label column name', 'Is the trajectory segmented?'])
            # If the user provided the input params, then run the function, else
            # wait for the user to play their part.
            if args:
                self._data = Statistics.pivot_stats_df(dataframe=self._data,
                                                       target_col_name=args[0].strip(),
                                                       segmented=bool(util.strtobool(args[1].strip())))

        # Finally, update the GUI with the updated DF received from the
        # function results. DO NOT FORGET THE reset_index(inplace=False).
        self._window.statusBar.showMessage("Task Done ...")
        self._model = TableModel(self._data.reset_index(inplace=False))
        self._table.setModel(self._model)

    def _get_input_params(self, labels, title, placeHolder):
        """
            Take the input parameters for the function in question from
            the user.

            Parameters
            ----------
                labels: list
                    The name of the parameters.
                title: str
                    The title of the input dialog box.
                placeHolder: list
                    Default text for each QLineEdit.

            Returns
            -------
                list:
                    A list containing the user input.
        """
        input_dialog = InputDialog(parent=self._window,
                                   labels=labels,
                                   title=title,
                                   placeHolders=placeHolder)
        if input_dialog.exec_():
            args = input_dialog.getInputs()

            return args

    def update_dropCol_options(self):
        """
            Update the options in the QListWidget for dropping the columns.
        """
        try:
            self._window.dropColumnWidget.clear()
            toAdd = list(self._data.columns)
            toAdd.remove('lat')
            toAdd.remove('lon')
            self._window.dropColumnWidget.addItems(toAdd)
        except ValueError:
            self._window.dropColumnWidget.clear()
            toAdd = list(self._data.columns)
            self._window.dropColumnWidget.addItems(toAdd)

    def drop_col(self):
        """
            Drop the columns based on the user selection.
        """
        # Get the selected items and then create a list
        # of the column names to drop.
        items = self._window.dropColumnWidget.selectedItems()

        to_drop = list()
        for it in items:
            to_drop.append(it.text())

        # Now, whenever we drop a column related to one of the stats,
        # then we go ahead and remove that option from the stats selection
        # panel as well.
        all_stat_items = [self._window.selectStatDropdown.itemText(i)
                          for i in range(self._window.selectStatDropdown.count())]
        for val in to_drop:
            if val in all_stat_items:
                all_stat_items.remove(val)

        # Make sure to block this signal before clearing and unblock
        # after adding new options :).
        self._window.selectStatDropdown.blockSignals(True)
        self._window.selectStatDropdown.clear()
        self._window.selectStatDropdown.addItems(all_stat_items)
        self._window.selectStatDropdown.blockSignals(False)

        # Drop the column(s) selected by the user.
        self._data.drop(columns=to_drop, inplace=True)

        # Update the options in the drop-column selection list and display the table again.
        self.update_dropCol_options()
        self._model = TableModel(self._data.reset_index(inplace=False))
        self._table.setModel(self._model)
