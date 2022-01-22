"""
    This module contains the design of PTRAIL's GUI module.
    It is to be noted that this class does not handle the functionalities,
    it is rather handled by the handler class.

    | Author: Yaksh J Haranwala
"""

from PyQt5 import QtCore, QtWidgets, QtGui, QtWebEngineWidgets
import sys
from ptrail.GUI.handler import GuiHandler


class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, OuterWindow):
        super(Ui_MainWindow, self).__init__()
        # Main Window variables.
        self.vlayout = None
        self.centralwidget = None

        # MenuBar variables
        self.VersionInfoButton = None
        self.QuitButton = None
        self.OpenButton = None
        self.SaveButton = None
        self.menuAbout = None
        self.FileMenu = None
        self.MenuBar = None

        # MapPane variables.
        self.maplayoutmanager = None
        self.MapPane = None
        self.map = None

        # CommandPalette variables
        self.cmdlayoutmanager = None
        self.CommandPalette = None
        self.feature_type = None
        self.listWidget = None
        self.run_stats_btn = None

        # StatsPane variables.
        self.statslayoutmanager = None
        self.StatsPane = None

        # DFPane variable.
        self.dflayoutmanager = None
        self.DFPane = None
        self.OuterWindow = OuterWindow

        self.handler = None
        self.setupUi(OuterWindow)

    def setupUi(self, OuterWindow):
        """
            Set the main window of the GUI up and start the application.

            Parameters
            ----------
                OuterWindow: PyQt5.QtWidgets.QOuterWindow'
        """
        OuterWindow.setObjectName("OuterWindow")
        OuterWindow.resize(1125, 776)
        
        self.centralwidget = QtWidgets.QWidget(OuterWindow)
        self.vlayout = QtWidgets.QGridLayout(self.centralwidget)
        self.centralwidget.setObjectName("centralwidget")

        self.setup_menubar()
        self.setup_map_pane()
        self.setup_command_palette()
        self.setup_stats_palette()
        self.setup_df_pane()
        self.retranslateUi(OuterWindow)

        OuterWindow.setCentralWidget(self.centralwidget)
        QtCore.QMetaObject.connectSlotsByName(OuterWindow)

    def setup_df_pane(self):
        """
            Set up the pane that displays the dataframe.

            Returns
            -------
                None
        """
        # Create the layout manager and set the size of the pane.
        self.dflayoutmanager = QtWidgets.QWidget(self.centralwidget)
        self.dflayoutmanager.setObjectName("verticalLayoutWidget")
        self.DFPane = QtWidgets.QVBoxLayout(self.dflayoutmanager)
        self.DFPane.setContentsMargins(0, 0, 0, 0)
        self.DFPane.setObjectName("DFPane")
        self.vlayout.addWidget(self.dflayoutmanager, 3, 0)

    def setup_stats_palette(self):
        """
            Set up the pane that displays the statistics.

            Returns
            -------
                None
        """
        # Create the layout manager and set the size of the pane.
        self.statslayoutmanager = QtWidgets.QWidget(self.centralwidget)
        self.statslayoutmanager.setObjectName("gridLayoutWidget")
        self.statslayoutmanager.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.StatsPane = QtWidgets.QGridLayout(self.statslayoutmanager)
        self.StatsPane.setContentsMargins(0, 0, 0, 0)
        self.StatsPane.setObjectName("StatsPane")

        # Add a sample label to the pane.
        stat_label = QtWidgets.QLabel()
        stat_label.setText("Statistics ")
        self.StatsPane.addWidget(stat_label)

        self.vlayout.addWidget(self.statslayoutmanager, 0, 3)

    def setup_map_pane(self):
        """
            Set up the pane that displays the map.

            Returns
            -------
                None
        """
        # Add the layout manager and then finally set the size and shape.
        self.maplayoutmanager = QtWidgets.QWidget(self.centralwidget)
        self.maplayoutmanager.setObjectName("gridLayoutWidget_2")
        self.MapPane = QtWidgets.QGridLayout(self.maplayoutmanager)
        self.MapPane.setContentsMargins(0, 0, 0, 0)
        self.MapPane.setObjectName("MapPane")

        self.map = QtWebEngineWidgets.QWebEngineView()
        self.MapPane.addWidget(self.map)

        self.vlayout.addWidget(self.maplayoutmanager, 0, 1)

    def setup_command_palette(self):
        """
            Set up the pane that displays the command palette.

            Returns
            -------
                None
        """
        # Create the layout manager and set the size of the pane.
        self.cmdlayoutmanager = QtWidgets.QWidget(self.centralwidget)
        self.cmdlayoutmanager.setObjectName("verticalLayoutWidget_2")
        self.CommandPalette = QtWidgets.QVBoxLayout(self.cmdlayoutmanager)
        self.CommandPalette.setContentsMargins(0, 0, 0, 0)
        self.CommandPalette.setObjectName("CommandPalette")

        label = QtWidgets.QLabel()
        label.setText("Command Palette")
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setFont(QtGui.QFont('Times font', 14))
        self.CommandPalette.addWidget(label)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        self.CommandPalette.addWidget(line)

        # ---------------------------------------------------------------------------------- #
        # Declare the list containing the features and then individual lists of all features.
        feature_types = [
            'Kinematic Features', 'Temporal Features', 'Semantic Features', 'Interpolation', 'Statistics'
        ]

        # ------------------- Feature Selection List ---------------------- #
        self.feature_type = QtWidgets.QComboBox()
        self.feature_type.addItems(sorted(feature_types))
        self.feature_type.setFont(QtGui.QFont('Times font', 12))
        self.feature_type.currentIndexChanged.connect(self.add_tree_options)
        self.CommandPalette.addWidget(self.feature_type)

        # ------------------- Multi Selection Widget ----------------------#
        self.listWidget = QtWidgets.QListWidget()
        self.listWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        ip_features = [
            'Linear Interpolation', 'Cubic Interpolation', 'Kinematic Interpolation',
            'Random-Walk Interpolation'
        ]
        self.listWidget.addItems(ip_features)
        self.listWidget.setFont(QtGui.QFont("Times Font", 11))
        self.CommandPalette.addWidget(self.listWidget)

        # ------------- Add the run commands button. --------------------- #
        self.run_stats_btn = QtWidgets.QPushButton("Run")
        self.run_stats_btn.toggle()
        self.run_stats_btn.setFont(QtGui.QFont('Times font', 12))
        self.CommandPalette.addWidget(self.run_stats_btn)

        self.vlayout.addWidget(self.cmdlayoutmanager, 0, 0)

    def add_tree_options(self):
        ip_features = [
            'Linear Interpolation', 'Cubic Interpolation', 'Kinematic Interpolation',
            'Random-Walk Interpolation'
        ]

        kinematic_features = [
            'All Kinematic Features', 'Bounding Box', 'Start Location', 'End Location',
            'Distance', 'Distance from Start', 'Distance travelled by date and trajectory Id',
            'Point within Range', 'Distance from Co-ordinates',
            'Speed', 'Acceleration', 'Jerk', 'Bearing', 'Bearing Rate',
            'Rate of Bearing Rate', 'Distance Travelled by traj Id', 'Number of Unique locations',
        ]

        semantic_features = [
            'Visited Location', 'Visited POI', 'Trajectories Inside Polygon',
            'Trajectories Intersect Inside Polygon', 'Nearest POI'
        ]

        stat_features = [
            'Segment Trajectories', 'Generate Kinematic Statistics',
            'Pivot Statistics DF'
        ]

        temporal_features = [
            'All Temporal Features', 'Date', 'Time', 'Day of the Week',
            'Weekend Indicator', 'Time of Day', 'Trajectory Duration', 'Start Time(s)',
            'End Time(s)'
        ]

        self.listWidget.clear()

        # Change the options of the list as per the current
        # feature selection.
        if self.feature_type.currentIndex() == 0:
            self.listWidget.addItems(ip_features)
        elif self.feature_type.currentIndex() == 1:
            self.listWidget.addItems(kinematic_features)
        elif self.feature_type.currentIndex() == 2:
            self.listWidget.addItems(semantic_features)
        elif self.feature_type.currentIndex() == 3:
            self.listWidget.addItems(stat_features)
        else:
            self.listWidget.addItems(temporal_features)

        self.listWidget.setFont(QtGui.QFont('Times font', 12))

    def setup_menubar(self):
        """
            Create the menu bar of the window.

            Returns
            -------
                None
        """
        # Create the Menu Bar.
        self.MenuBar = QtWidgets.QMenuBar(self.OuterWindow)
        self.MenuBar.setObjectName("MenuBar")

        # Create the File Menu.
        self.FileMenu = QtWidgets.QMenu(self.MenuBar)
        self.FileMenu.setObjectName("FileMenu")

        # Create the save button of the File Menu.
        self.SaveButton = QtWidgets.QAction(self.OuterWindow)
        self.SaveButton.setObjectName("SaveButton")
        self.SaveButton.setShortcut("Ctrl+S")
        self.FileMenu.addAction(self.SaveButton)

        # Create the open button of the File Menu.
        self.OpenButton = QtWidgets.QAction(self.OuterWindow)
        self.OpenButton.triggered.connect(self.open_file)
        self.OpenButton.setObjectName("OpenButton")
        self.OpenButton.setShortcut("Ctrl+O" if sys.platform != 'darwin' else "cmd+O")
        self.FileMenu.addAction(self.OpenButton)

        self.FileMenu.addSeparator()
        # Create the quit button of the File Menu and add action listener.
        self.QuitButton = QtWidgets.QAction(self.OuterWindow)
        self.QuitButton.triggered.connect(QtWidgets.qApp.quit)
        self.QuitButton.setShortcut("Alt+F4" if sys.platform != 'darwin' else 'cmd+W')
        self.QuitButton.setObjectName("QuitButton")
        self.FileMenu.addAction(self.QuitButton)

        # Create the About Menu.
        self.menuAbout = QtWidgets.QMenu(self.MenuBar)
        self.menuAbout.setObjectName("menuAbout")

        # Add the Version Info Button to the About Menu and add the action listener.
        self.VersionInfoButton = QtWidgets.QAction(self.OuterWindow)
        self.VersionInfoButton.triggered.connect(self.version_button_clicked)
        self.VersionInfoButton.setShortcut("F1")
        self.VersionInfoButton.setObjectName("VersionInfoButton")
        self.menuAbout.addAction(self.VersionInfoButton)

        self.MenuBar.addAction(self.FileMenu.menuAction())
        self.MenuBar.addAction(self.menuAbout.menuAction())
        self.OuterWindow.setMenuBar(self.MenuBar)

    def open_file(self):
        """
            Open the file and load the dataframe to perform operations.

            Returns
            -------
                None
        """
        # Create the file selection dialog box.
        options = QtWidgets.QFileDialog.Options()

        # Limit the user to select only CSV files to load.
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "QFileDialog.getOpenFileName()", "",
                                                  ";csv Files (*.csv)", options=options)

        # If the user selects a file, load it into a pandas dataframe and print its statistics.
        if fileName:
            self.handler = GuiHandler(fileName, self)

    def version_button_clicked(self):
        """
            Show the version info of the application.

            Returns
            -------
                None
        """
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setWindowTitle("Version Info")
        msg.setText("Version: 0.6.3 Beta \n"
                    "Authors: Yaksh J Haranwala, Salman Haidri")
        msg.exec()

    def retranslateUi(self, OuterWindow):
        """ Auto Generated method by PyQt Designer."""
        _translate = QtCore.QCoreApplication.translate
        OuterWindow.setWindowTitle(_translate("OuterWindow", "PTRAIL - A Trajectory Preprocessing Tool"))
        self.FileMenu.setTitle(_translate("OuterWindow", "File"))
        self.menuAbout.setTitle(_translate("OuterWindow", "About"))
        self.OpenButton.setText(_translate("OuterWindow", "Open"))
        self.SaveButton.setText(_translate("OuterWindow", "Save"))
        self.QuitButton.setText(_translate("OuterWindow", "Quit"))
        self.VersionInfoButton.setText(_translate("OuterWindow", "Version Info"))
