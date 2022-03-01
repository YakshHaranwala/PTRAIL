"""
    This module contains the design of PTRAIL's GUI module.
    It is to be noted that this class does not handle the functionalities,
    it is rather handled by the handler class.

    | Author: Yaksh J Haranwala
"""
import sys

from PyQt5 import QtCore, QtWidgets, QtGui, QtWebEngineWidgets
from PyQt5.QtWidgets import QSizePolicy

from ptrail.GUI.handler import GuiHandler


# TODO: The application crashes when we cancel file upload and try to upload a different file.
class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, OuterWindow):
        super(Ui_MainWindow, self).__init__()
        # Main Window variables.
        self.statusBar = None
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
        self.open_btn = None

        # CommandPalette variables
        self.cmdlayoutmanager = None
        self.CommandPalette = None
        self.featureType = None
        self.featureListWidget = None
        self.dropColumnWidget = None
        self.runStatsBtn = None
        self.dropColumnBtn = None

        # StatsPane variables.
        self.statslayoutmanager = None
        self.StatsPane = None
        self.selectStatDropdown = None
        self.figure = None
        self.canvas = None

        # DFPane variable.
        self.dflayoutmanager = None
        self.DFPane = None
        self.OuterWindow = OuterWindow
        self.exportBtn = None
        self.dfController = None
        self.dfView = None

        # The GUI conductor.
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
        self.setup_command_palette()
        self.setup_stats_palette()
        self.setup_map_pane()
        self.setup_df_pane()
        self.setup_statusbar()
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
        self.vlayout.addWidget(self.dflayoutmanager, 3, 1, 1, 2)

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
        self.StatsPane = QtWidgets.QVBoxLayout(self.statslayoutmanager)
        self.StatsPane.setContentsMargins(0, 0, 0, 0)
        self.StatsPane.setObjectName("StatsPane")

        self.vlayout.addWidget(self.statslayoutmanager, 0, 3, 4, 1)

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

        self.open_btn = QtWidgets.QPushButton("Open File")
        self.open_btn.clicked.connect(self.open_file)
        self.open_btn.resize(150, 50)
        self.MapPane.addWidget(self.open_btn)

        self.vlayout.addWidget(self.maplayoutmanager, 0, 1, 3, 2)

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

        # ---------------------------------------------------------------------------------- #
        # Declare the list containing the features and then individual lists of all features.
        feature_types = [
            'Kinematic Features', 'Temporal Features', 'Filtering', 'Interpolation', 'Statistics'
        ]

        newVLayout = QtWidgets.QVBoxLayout()
        # ------------------- Feature Selection List ---------------------- #
        self.featureType = QtWidgets.QComboBox()
        self.featureType.addItems(sorted(feature_types))
        self.featureType.setFont(QtGui.QFont('Tahoma', 12))
        self.featureType.currentIndexChanged.connect(self.add_tree_options)
        newVLayout.addWidget(self.featureType)

        # ------------------- Multi Selection Widget ----------------------#
        self.featureListWidget = QtWidgets.QListWidget()
        self.featureListWidget.setSizePolicy(QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored))
        filt_features = [
            'Hampel Filter', 'Remove Duplicates', 'By Trajectory ID', 'By Bounding Box', 'By Date',
            'By DateTime', 'By Maximum Speed', 'By Minimum Speed', 'By Minimum Consecutive Distance',
            'By Maximum Consecutive Distance', 'By Maximum Distance and Speed', 'By Minimum Distance and Speed',
            'Remove Outliers by Consecutive Distance', 'Remove Outliers by Consecutive Speed',
            'Remove Trajectories with Less Points',
        ]

        self.featureListWidget.addItems(filt_features)
        self.featureListWidget.setFont(QtGui.QFont('Tahoma', 12))
        self.featureListWidget.setUniformItemSizes(True)
        self.featureListWidget.item(0).setSelected(True)
        newVLayout.addWidget(self.featureListWidget)

        # ------------- Add the run commands button. --------------------- #
        self.runStatsBtn = QtWidgets.QPushButton("Run")
        self.runStatsBtn.toggle()
        self.runStatsBtn.setFont(QtGui.QFont('Tahoma', 12))
        self.runStatsBtn.setEnabled(False)
        self.runStatsBtn.clicked.connect(lambda: self.handler.run_command())
        newVLayout.addWidget(self.runStatsBtn)
        self.CommandPalette.addLayout(newVLayout)

        self.vlayout.addWidget(self.cmdlayoutmanager, 0, 0, 4, 1)

    def add_tree_options(self):
        ip_features = [
            'Linear Interpolation', 'Cubic Interpolation', 'Kinematic Interpolation',
            'Random-Walk Interpolation'
        ]

        kinematic_features = [
            'All Kinematic Features', 'Distance', 'Distance from Start', 'Point within Range',
            'Distance from Co-ordinates', 'Speed', 'Acceleration', 'Jerk', 'Bearing', 'Bearing Rate',
            'Rate of Bearing Rate',
        ]

        filt_features = [
            'Hampel Filter', 'Remove Duplicates', 'By Trajectory ID', 'By Bounding Box', 'By Date',
            'By DateTime', 'By Maximum Speed', 'By Minimum Speed', 'By Minimum Consecutive Distance',
            'By Maximum Consecutive Distance', 'By Maximum Distance and Speed', 'By Minimum Distance and Speed',
            'Remove Outliers by Consecutive Distance', 'Remove Outliers by Consecutive Speed',
            'Remove Trajectories with Less Points',
        ]

        stat_features = [
            'Segment Trajectories', 'Generate Kinematic Statistics',
            'Pivot Statistics DF'
        ]

        temporal_features = [
            'All Temporal Features', 'Date', 'Time',
            'Day of the Week', 'Weekend Indicator', 'Time of Day',
        ]
        self.featureListWidget.clear()

        # Change the options of the list as per the current
        # feature selection.
        if self.featureType.currentIndex() == 0:
            self.featureListWidget.addItems(filt_features)
        elif self.featureType.currentIndex() == 1:
            self.featureListWidget.addItems(ip_features)
        elif self.featureType.currentIndex() == 2:
            self.featureListWidget.addItems(kinematic_features)
        elif self.featureType.currentIndex() == 3:
            self.featureListWidget.addItems(stat_features)
        else:
            self.featureListWidget.addItems(temporal_features)

        self.featureListWidget.setFont(QtGui.QFont('Tahoma', 12))
        self.featureListWidget.item(0).setSelected(True)
        self.featureListWidget.setUniformItemSizes(True)

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

        # Create the open button of the File Menu.
        self.OpenButton = QtWidgets.QAction(self.OuterWindow)
        self.OpenButton.triggered.connect(self.open_file)
        self.OpenButton.setObjectName("OpenButton")
        self.OpenButton.setShortcut("Ctrl+O" if sys.platform != 'darwin' else "cmd+O")
        self.FileMenu.addAction(self.OpenButton)

        # Create the save button of the File Menu.
        self.SaveButton = QtWidgets.QAction(self.OuterWindow)
        self.SaveButton.triggered.connect(self.save_file)
        self.SaveButton.setObjectName("SaveButton")
        self.SaveButton.setShortcut("Ctrl+S" if sys.platform != 'darwin' else "cmd+S")
        self.FileMenu.addAction(self.SaveButton)

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

    def retranslateUi(self, OuterWindow):
        """ Auto Generated method by PyQt Designer."""
        _translate = QtCore.QCoreApplication.translate
        OuterWindow.setWindowTitle(_translate("OuterWindow", "PTRAIL - A Trajectory Preprocessing Tool"))
        self.FileMenu.setTitle(_translate("OuterWindow", "File"))
        self.menuAbout.setTitle(_translate("OuterWindow", "About"))
        self.OpenButton.setText(_translate("OuterWindow", "Open"))
        self.SaveButton.setText(_translate("OuterWindow", "Export"))
        self.QuitButton.setText(_translate("OuterWindow", "Quit"))
        self.VersionInfoButton.setText(_translate("OuterWindow", "Version Info"))

    def add_df_controller(self):
        # Create a smaller box layout.
        self.dfController = QtWidgets.QHBoxLayout()

        # # Create a trajectory View selector.
        # self.dfView = QtWidgets.QComboBox()
        # self.dfView.addItems(['Point-Based View', 'Segment-Based View'])
        # self.dfController.addWidget(self.dfView)

        # Create an export button for the dataset.
        self.exportBtn = QtWidgets.QPushButton("Export Dataframe")
        self.exportBtn.clicked.connect(self.save_file)
        self.dfController.addWidget(self.exportBtn)

        # Add this mini panel to the main DF panel.
        self.DFPane.addLayout(self.dfController)

    def setup_statusbar(self):
        self.statusBar = QtWidgets.QStatusBar()
        self.statusBar.setStyleSheet("border :1px solid grey;")
        self.statusBar.setFont(QtGui.QFont('Tahoma', 12))
        self.OuterWindow.setStatusBar(self.statusBar)

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
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Load Dataset", "",
                                                            ";csv Files (*.csv)", options=options)

        # If the user selects a file, load it into a pandas dataframe and print its statistics.
        if fileName is not None and fileName != '':
            self.handler = GuiHandler(fileName, self)
        else:
            pass

    def save_file(self):
        """
            Save the dataframe to a .csv file.

            Returns
            -------
                None
        """
        # Create the file selection dialog box.
        options = QtWidgets.QFileDialog.Options()

        # Limit the user to select only CSV files to load.
        file, check = QtWidgets.QFileDialog.getSaveFileName(None, "Save File", "",
                                                            "Csv Files (*.csv)", options=options)

        # If the user has decided to save the file, save the dataframe to the
        # specified location and display the success message.
        if file and self.handler._data is not None:
            try:
                self.handler._data.to_csv(file + '.csv')
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Information)
                msg.setWindowTitle("Success")
                msg.setText("File Saved Successfully!")
                msg.exec()
            except PermissionError:
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Critical)
                msg.setWindowTitle("Error")
                msg.setText("File could not be saved. Please try again!")
                msg.exec()
        else:
            pass

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
