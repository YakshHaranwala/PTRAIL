"""
    This class is an abstraction that can be used to create
    input dialog boxes for virtually any number of inputs.

    | Authors: Yaksh J Haranwala
"""
from typing import List
from PyQt5 import QtWidgets


class InputDialog(QtWidgets.QDialog):
    def __init__(self, labels: List[str], title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)

        buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok |
                                               QtWidgets.QDialogButtonBox.Cancel, self)
        layout = QtWidgets.QFormLayout(self)

        self.inputs = []
        for lab in labels:
            self.inputs.append(QtWidgets.QLineEdit(self))
            layout.addRow(lab, self.inputs[-1])

        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        return list(input.text() for input in self.inputs)