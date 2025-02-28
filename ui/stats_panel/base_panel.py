from PyQt5.QtWidgets import QWidget


class BaseStatsPanel(QWidget):
    """Base class for all statistics panels"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        """Setup the UI components - to be implemented by subclasses"""
        pass

    def update_stats(self, data_manager):
        """Update statistics - to be implemented by subclasses"""
        pass
