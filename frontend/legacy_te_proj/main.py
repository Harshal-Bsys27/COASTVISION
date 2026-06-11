# main.py
import sys

from PyQt5.QtWidgets import QApplication
from ui import Dashboard

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dash = Dashboard()
    dash.show()
    sys.exit(app.exec_())
# main.py
import sys
from PyQt5.QtWidgets import QApplication
from ui import Dashboard

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dash = Dashboard()
    dash.show()
    sys.exit(app.exec_())
