"""
    This file launches PTRAIL's GUI module.

    | Author: Yaksh J Haranwala
"""
from PyQt5 import QtWidgets
import sys
from ptrail.GUI.gui import Ui_MainWindow

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow(MainWindow)
    MainWindow.showMaximized()
    sys.exit(app.exec_())
