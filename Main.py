import sys
from PyQt5.QtWidgets import QApplication
from app.main_window import MainWindow
from PyQt5.QtGui import QIcon


def main():
    app = QApplication(sys.argv)
    icon = QIcon("./asset/icon.png")
    app.setWindowIcon(icon)
    window = MainWindow()
    window.setWindowTitle("Pixelgg")
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
