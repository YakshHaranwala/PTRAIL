from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
import pandas as pd


class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, MainWindow):
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

        # StatsPane variables.
        self.statslayoutmanager = None
        self.StatsPane = None

        # DFPane variable.
        self.dflayoutmanager = None
        self.DFPane = None

        self.setupUi(MainWindow)

    def setupUi(self, MainWindow):
        """
            Set the main window of the GUI up and start the application.

            Parameters
            ----------
                MainWindow: PyQt5.QtWidgets.QMainWindow'
        """
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1125, 776)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.centralwidget.setObjectName("centralwidget")

        self.setup_menubar()
        self.setup_map_pane()
        self.setup_command_palette()
        self.setup_stats_palette()
        self.setup_df_pane()
        self.retranslateUi(MainWindow)

        MainWindow.setMaximumSize(QtCore.QSize(1125, 776))
        MainWindow.setMinimumSize(QtCore.QSize(1125, 776))
        MainWindow.setCentralWidget(self.centralwidget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

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

        file_name = QLabel()
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

        # add a sample label.
        cmd_label = QtWidgets.QLabel()
        cmd_label.setText("Command Palette")
        self.CommandPalette.addWidget(cmd_label)

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

        # Add a frame to the pane and add a sample label to it.
        map_frame = QtWidgets.QFrame()
        map_frame.setGeometry(1, 1, 509, 519)
        map_frame.setFrameStyle(QtWidgets.QFrame.Panel)
        self.MapPane.addWidget(map_frame)

        label = QLabel()
        label.setText("Map will go here.")
        label.setParent(map_frame)

    def setup_menubar(self):
        """
            Create the menu bar of the window.

            Returns
            -------
                None
        """
        # Create the Menu Bar.
        self.MenuBar = QtWidgets.QMenuBar(MainWindow)
        self.MenuBar.setGeometry(QtCore.QRect(0, 0, 1125, 23))
        self.MenuBar.setObjectName("MenuBar")

        # Create the File Menu.
        self.FileMenu = QtWidgets.QMenu(self.MenuBar)
        self.FileMenu.setObjectName("FileMenu")

        # Create the save button of the File Menu.
        self.SaveButton = QtWidgets.QAction(MainWindow)
        self.SaveButton.setObjectName("SaveButton")
        self.SaveButton.setShortcut("Ctrl+S")
        self.FileMenu.addAction(self.SaveButton)

        # Create the open button of the File Menu.
        self.OpenButton = QtWidgets.QAction(MainWindow)
        self.OpenButton.triggered.connect(self.open_file)
        self.OpenButton.setObjectName("OpenButton")
        self.OpenButton.setShortcut("Ctrl+O" if sys.platform != 'darwin' else "cmd+O")
        self.FileMenu.addAction(self.OpenButton)

        self.FileMenu.addSeparator()
        # Create the quit button of the File Menu and add action listener.
        self.QuitButton = QtWidgets.QAction(MainWindow)
        self.QuitButton.triggered.connect(qApp.quit)
        self.QuitButton.setShortcut("Alt+F4" if sys.platform != 'darwin' else 'cmd+W')
        self.QuitButton.setObjectName("QuitButton")
        self.FileMenu.addAction(self.QuitButton)

        # Create the About Menu.
        self.menuAbout = QtWidgets.QMenu(self.MenuBar)
        self.menuAbout.setObjectName("menuAbout")

        # Add the Version Info Button to the About Menu and add the action listener.
        self.VersionInfoButton = QtWidgets.QAction(MainWindow)
        self.VersionInfoButton.triggered.connect(self.version_button_clicked)
        self.VersionInfoButton.setShortcut("F1")
        self.VersionInfoButton.setObjectName("VersionInfoButton")
        self.menuAbout.addAction(self.VersionInfoButton)

        self.MenuBar.addAction(self.FileMenu.menuAction())
        self.MenuBar.addAction(self.menuAbout.menuAction())
        MainWindow.setMenuBar(self.MenuBar)

    def open_file(self):
        """
            Open the file and load the dataframe to perform operations.

            Returns
            -------
                None
        """
        # Create the file selection dialog box.
        options = QFileDialog.Options()

        # Limit the user to select only CSV files to load.
        fileName, _ = QFileDialog.getOpenFileName(None, "QFileDialog.getOpenFileName()", "",
                                                  ";csv Files (*.csv)", options=options)

        # If the user selects a file, load it into a pandas dataframe and print its statistics.
        if fileName:
            df = pd.read_csv(fileName)
            print(df.describe())

    def version_button_clicked(self):
        """
            Show the version info of the application.

            Returns
            -------
                None
        """
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Version Info")
        msg.setText("Version: 0.6.3 Beta \n"
                    "Authors: Yaksh J Haranwala, Salman Haidri")
        msg.exec()

    def retranslateUi(self, MainWindow):
        """ Auto Generated method by PyQt Designer."""
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "PTRAIL - A Trajectory Preprocessing Tool"))
        self.FileMenu.setTitle(_translate("MainWindow", "File"))
        self.menuAbout.setTitle(_translate("MainWindow", "About"))
        self.OpenButton.setText(_translate("MainWindow", "Open"))
        self.SaveButton.setText(_translate("MainWindow", "Save"))
        self.QuitButton.setText(_translate("MainWindow", "Quit"))
        self.VersionInfoButton.setText(_translate("MainWindow", "Version Info"))


if __name__ == "__main__":
    import sys
    
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    print(type(MainWindow))

    ui = Ui_MainWindow(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
