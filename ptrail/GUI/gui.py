"""
    This module contains the design of PTRAIL's GUI module.
    It is to be noted that this class does not handle the functionalities,
    it is rather handled by the handler class.

    | Author: Yaksh J Haranwala
"""

from PyQt5 import QtCore, QtWidgets, QtGui
import sys
from ptrail.GUI.handler import GuiHandler


class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, OuterWindow):
        super(Ui_MainWindow, self).__init__()
        # Main Window variables.
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

        # CommandPalette variables
        self.cmdlayoutmanager = None
        self.CommandPalette = None
        self.kinematic_list = None
        self.temporal_list = None
        self.semantic_list = None
        self.ip_list = None
        self.stats_list = None
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
        self.centralwidget.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.centralwidget.setObjectName("centralwidget")

        self.setup_menubar()
        self.setup_map_pane()
        self.setup_command_palette()
        self.setup_stats_palette()
        self.setup_df_pane()
        self.retranslateUi(OuterWindow)

        OuterWindow.setMaximumSize(QtCore.QSize(1125, 776))
        OuterWindow.setMinimumSize(QtCore.QSize(1125, 776))
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
        self.dflayoutmanager.setGeometry(QtCore.QRect(0, 520, 761, 231))
        self.dflayoutmanager.setObjectName("verticalLayoutWidget")
        self.DFPane = QtWidgets.QVBoxLayout(self.dflayoutmanager)
        self.DFPane.setContentsMargins(0, 0, 0, 0)
        self.DFPane.setObjectName("DFPane")

        file_name = QtWidgets.QLabel()
        file_name.setText("The dataframe will be displayed here.")
        self.DFPane.addWidget(file_name)

    def setup_stats_palette(self):
        """
            Set up the pane that displays the statistics.

            Returns
            -------
                None
        """
        # Create the layout manager and set the size of the pane.
        self.statslayoutmanager = QtWidgets.QWidget(self.centralwidget)
        self.statslayoutmanager.setGeometry(QtCore.QRect(760, 0, 361, 751))
        self.statslayoutmanager.setObjectName("gridLayoutWidget")
        self.StatsPane = QtWidgets.QGridLayout(self.statslayoutmanager)
        self.StatsPane.setContentsMargins(0, 0, 0, 0)
        self.StatsPane.setObjectName("StatsPane")

        # Add a sample label to the pane.
        stat_label = QtWidgets.QLabel()
        stat_label.setText("Statistics ")
        self.StatsPane.addWidget(stat_label)

    def setup_command_palette(self):
        """
            Set up the pane that displays the command palette.

            Returns
            -------
                None
        """
        # Create the layout manager and set the size of the pane.
        self.cmdlayoutmanager = QtWidgets.QWidget(self.centralwidget)
        self.cmdlayoutmanager.setGeometry(QtCore.QRect(0, 0, 251, 521))
        self.cmdlayoutmanager.setObjectName("verticalLayoutWidget_2")
        self.CommandPalette = QtWidgets.QVBoxLayout(self.cmdlayoutmanager)
        self.CommandPalette.setContentsMargins(0, 0, 0, 0)
        self.CommandPalette.setObjectName("CommandPalette")

        self.command_palette_boxes()

    def setup_map_pane(self):
        """
            Set up the pane that displays the map.

            Returns
            -------
                None
        """
        # Add the layout manager and then finally set the size and shape.
        self.maplayoutmanager = QtWidgets.QWidget(self.centralwidget)
        self.maplayoutmanager.setGeometry(QtCore.QRect(250, 0, 511, 521))
        self.maplayoutmanager.setObjectName("gridLayoutWidget_2")
        self.MapPane = QtWidgets.QGridLayout(self.maplayoutmanager)
        self.MapPane.setContentsMargins(0, 0, 0, 0)
        self.MapPane.setObjectName("MapPane")

        label = QtWidgets.QLabel()
        label.setText("Map will go here.")
        self.MapPane.addWidget(label)

    def setup_menubar(self):
        """
            Create the menu bar of the window.

            Returns
            -------
                None
        """
        # Create the Menu Bar.
        self.MenuBar = QtWidgets.QMenuBar(self.OuterWindow)
        self.MenuBar.setGeometry(QtCore.QRect(0, 0, 1125, 23))
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

    def command_palette_boxes(self):
        label = QtWidgets.QLabel()
        label.setText("Command Palette")
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setFont(QtGui.QFont('Times font', 14))
        self.CommandPalette.addWidget(label)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        self.CommandPalette.addWidget(line)

        # ------------- Create the kinematic Features drop down menu. --------------------- #
        self.kinematic_list = QtWidgets.QComboBox()
        kinematic_features = [
            'All Kinematic Features', 'Bounding Box', 'Start Location', 'End Location',
            'Distance', 'Distance from Start', 'Distance travelled by date and trajectory Id',
            'Point within Range', 'Distance from Co-ordinates',
            'Speed', 'Acceleration', 'Jerk', 'Bearing', 'Bearing Rate',
            'Rate of Bearing Rate', 'Distance Travelled by traj Id', 'Number of Unique locations',
        ]
        self.kinematic_list.addItems(kinematic_features)
        label1 = QtWidgets.QLabel()
        label1.setText("Kinematic Features")
        label1.setAlignment(QtCore.Qt.AlignCenter)
        self.CommandPalette.addWidget(label1)
        self.CommandPalette.addWidget(self.kinematic_list)

        # ------------- Create the Temporal Features drop down menu. --------------------- #
        self.temporal_list = QtWidgets.QComboBox()
        temporal_features = [
            'All Temporal Features', 'Date', 'Time', 'Day of the Week',
            'Weekend Indicator', 'Time of Day', 'Trajectory Duration', 'Start Time(s)',
            'End Times'
        ]
        self.temporal_list.addItems(temporal_features)
        label2 = QtWidgets.QLabel()
        label2.setText("Temporal Features")
        label2.setAlignment(QtCore.Qt.AlignCenter)
        self.CommandPalette.addWidget(label2)
        self.CommandPalette.addWidget(self.temporal_list)

        # ------------- Create the Semantic Features drop down menu. --------------------- #
        self.semantic_list = QtWidgets.QComboBox()
        semantic_features = [
            'Visited Location', 'Visited POI', 'Trajectories Inside Polygon',
            'Trajectories Intersect Inside Polygon', 'Nearest POI'
        ]
        self.semantic_list.addItems(semantic_features)
        label3 = QtWidgets.QLabel()
        label3.setText("Semantic Features")
        label3.setAlignment(QtCore.Qt.AlignCenter)
        self.CommandPalette.addWidget(label3)
        self.CommandPalette.addWidget(self.semantic_list)

        # ------------- Create the Semantic Features drop down menu. --------------------- #
        self.ip_list = QtWidgets.QComboBox()
        ip_features = [
            'Linear Interpolation', 'Cubic Interpolation', 'Kinematic Interpolation',
            'Random-Walk Interpolation'
        ]
        self.ip_list.addItems(ip_features)
        label4 = QtWidgets.QLabel("Interpolation")
        label4.setAlignment(QtCore.Qt.AlignCenter)
        self.CommandPalette.addWidget(label4)
        self.CommandPalette.addWidget(self.ip_list)

        # ------------- Create the Semantic Features drop down menu. --------------------- #
        self.stats_list = QtWidgets.QComboBox()
        stat_features = [
            'Segment Trajectories', 'Generate Kinematic Statistics',
            'Pivot Statistics DF'
        ]
        self.stats_list.addItems(stat_features)
        label5 = QtWidgets.QLabel("Statistical Features")
        label5.setAlignment(QtCore.Qt.AlignCenter)
        self.CommandPalette.addWidget(label5)
        self.CommandPalette.addWidget(self.stats_list)

        # ------------- Add the run commands button. --------------------- #
        self.run_stats_btn = QtWidgets.QPushButton("Run Commands")
        self.run_stats_btn.toggle()
        self.CommandPalette.addWidget(self.run_stats_btn)
