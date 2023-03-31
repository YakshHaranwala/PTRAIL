from PyQt5.QtWidgets import QComboBox, QCompleter, QLineEdit
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt, QStringListModel, QTimer


class SearchableComboBox(QComboBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filtered_model = QStandardItemModel()
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.NoInsert)

        self.lineEdit().textChanged.connect(self.on_text_changed)

        self.completer = QCompleter(self)
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.completer.setCompletionMode(QCompleter.UnfilteredPopupCompletion)
        self.setCompleter(self.completer)

        self.timer = QTimer()
        self.timer.timeout.connect(self.search_items)
        self.timer.setSingleShot(True)

    def search_items(self):
        text = self.lineEdit().text()
        if not text:
            self.completer.setModel(None)
            self.hidePopup()
            return

        filtered_items = []
        for i in range(self.count()):
            item_text = self.itemText(i)
            if item_text.find(text) != -1:
                filtered_items.append(item_text)

        print(self.currentText())
        if not filtered_items:
            self.completer.setModel(None)
            self.hidePopup()
        else:
            self.filtered_model = QStringListModel(filtered_items)
            self.completer.setModel(self.filtered_model)
            self.completer.setCompletionPrefix(text)
            self.completer.popup().setCurrentIndex(self.completer.completionModel().index(0, 0))
            self.showPopup()

    def on_text_changed(self, text):
        self.timer.stop()
        self.timer.start(500)  # Set a delay of 500 ms before executing the search
