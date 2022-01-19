"""
    This class is used to connect the PTRAIL GUI to PTRAIL
    backend. All the GUI's functionalities are handled in this
    class.

    | Authors: Yaksh J Haranwala, Salman Haidri
"""
import pandas as pd


class GuiHandler:
    def __init__(self, filename, window):
        self._window = window
        self._data = pd.read_csv(filename)
        print(self._data.describe())
        print(self._window)
