"""
    This class is an abstraction that can be used to create
    input dialog boxes for virtually any number of inputs.

    | Authors: Yaksh J Haranwala
"""
from typing import List
from PyQt5 import QtWidgets


class InputDialog(QtWidgets.QDialog):
    def __init__(self, labels: List[str], title: str, placeHolders: List[str], parent=None,):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(560)

        buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok |
                                               QtWidgets.QDialogButtonBox.Cancel, self)
        layout = QtWidgets.QFormLayout(self)

        self.inputs = []
        for i in range(len(labels)):
            edit = QtWidgets.QLineEdit(self)
            edit.setPlaceholderText(placeHolders[i])
            self.inputs.append(edit)
            layout.addRow(labels[i], self.inputs[-1])

        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        return list(input.text() for input in self.inputs)