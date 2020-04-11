from PyQt5.QtCore import Qt, QRect
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QDesktopWidget
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QFont

WWIDTH = 1000
WHIGH = 800
MARGIN = 30


class CViewer(QWidget):
    def __init__(self, title='label classifier'):
        super().__init__()
        self.title = title
        self.desktop = QDesktopWidget()
        self.screen = self.desktop.availableGeometry()
        self.init_window()
        self.resize(WWIDTH, WHIGH)

    def init_window(self):
        self.setWindowTitle(self.title)
        self.init_widgets()

    def init_widgets(self):
        self.imW = WWIDTH // 2 - 1.5 * MARGIN
        self.imH = WHIGH - 3 * MARGIN
        LINE1 = MARGIN
        LINE2 = LINE1 * 2 + self.imW
        self.font_default = QFont("Monospace")
        self.font_default.setPointSize(14)
        self.font_default.setStyleHint(QFont.TypeWriter)

        self.font_2 = QFont("Monospace")
        self.font_2.setPointSize(14)
        self.font_2.setStyleHint(QFont.TypeWriter)

        self.font_3 = QFont("Monospace")
        self.font_3.setPointSize(72)
        self.font_3.setStyleHint(QFont.TypeWriter)
        self.font_3.setWeight(100)

        self.label_image = QLabel(self)

        self.label_image.setGeometry(QRect(LINE1, MARGIN * 2, MARGIN + self.imW, MARGIN + self.imH))
        self.label_image.setObjectName("label_image")

        self.red_mark = QLabel(self)
        self.red_mark.setFont(self.font_3)
        self.red_mark.setGeometry(QRect(LINE1 * 2, MARGIN * 2, WWIDTH - MARGIN, MARGIN + self.imH))
        self.red_mark.setStyleSheet("color: red;")

        self.label_selection_list = QLabel(self)
        self.label_selection_list.setFont(self.font_default)
        self.label_selection_list.setGeometry(QRect(LINE2, 30, 301, 311))
        self.label_selection_list.setObjectName("label_selection_list")

        self.label_status = QLabel(self)
        self.label_status.setGeometry(QRect(LINE1, 30, 800, 31))
        self.label_status.setObjectName("label_status")

        self.label = QLabel(self)
        self.label.setGeometry(QRect(LINE1, 380, 67, 17))
        self.label.setObjectName("label")

        self.Help_label = QLabel(self)
        self.Help_label.setFont(self.font_2)
        self.Help_label.setGeometry(QRect(LINE2, 420, 641, 200))
        self.Help_label.setTextFormat(Qt.PlainText)
        self.help_label_text = '0...9   select label ' \
                               '\na...... previous  ' \
                               '\nd...... follows   ' \
                               '\nspace.. new unlabeled image' \
                               '\ns...... store'
        self.Help_label.setText(self.help_label_text)
        self.Help_label.setObjectName("Help_label")


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    viewer = CViewer()
    app.exec()
