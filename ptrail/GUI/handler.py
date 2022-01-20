"""
    This class is used to connect the PTRAIL GUI to PTRAIL
    backend. All the GUI's functionalities are handled in this
    class.

    | Authors: Yaksh J Haranwala, Salman Haidri
"""
import pandas as pd
from PyQt5 import QtWidgets
from ptrail.GUI.Table import TableModel


class GuiHandler:
    def __init__(self, filename, window):
        self._window = window
        self._data = None
        self._model = None
        self._table = None

        self.display_df(filename=filename)

    def display_df(self, filename):
        # First, we clear out the DF pane area.
        # This is done in order to make sure that 2 dataframes
        # are not loaded simultaneously making the view cluttered.
        for i in reversed(range(self._window.DFPane.count())):
            item = self._window.DFPane.itemAt(i)
            if isinstance(item, QtWidgets.QTableView):
                item.widget().close()

            # remove the item from layout
            self._window.DFPane.removeItem(item)

        # Assign the dataframe.
        self._data = pd.read_csv(filename)

        # Set the table model and display the dataframe.
        self._table = QtWidgets.QTableView()
        self._model = TableModel(self._data)
        self._table.setModel(self._model)
        self._window.DFPane.addWidget(self._table)
