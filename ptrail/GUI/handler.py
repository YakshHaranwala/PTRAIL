"""
    This class is used to connect the PTRAIL GUI to PTRAIL
    backend. All the GUI's functionalities are handled in this
    class.

    | Authors: Yaksh J Haranwala, Salman Haidri
"""
import pandas as pd
from PyQt5 import QtWidgets

from ptrail.GUI.Table import TableModel
from ptrail.GUI.InputDialog import InputDialog

from ptrail.core.TrajectoryDF import PTRAILDataFrame


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

        # Create the input dialog item.
        input_dialog = InputDialog(parent=self._window,
                                   labels=['Trajectory ID: ', 'DateTime: ', 'Latitude: ', 'Longitude: '],
                                   title='Enter the column names: ')

        # Get the input before displaying the dataframe.
        if input_dialog.exec():
            col_names = input_dialog.getInputs()

            self._data = PTRAILDataFrame(data_set=pd.read_csv(filename),
                                         traj_id=col_names[0],
                                         datetime=col_names[1],
                                         latitude=col_names[2],
                                         longitude=col_names[3])
            # Set the table model and display the dataframe.
            self._table = QtWidgets.QTableView()
            self._model = TableModel(self._data.reset_index(inplace=False))
            self._table.setModel(self._model)
            self._window.DFPane.addWidget(self._table)
