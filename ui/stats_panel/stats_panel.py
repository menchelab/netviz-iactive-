import logging
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTabWidget

from .main_stats_panel import MainStatsPanel


class NetworkStatsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)  # Cleaner look
        layout.addWidget(self.tab_widget)

        # Create main stats panel
        self.main_stats_panel = MainStatsPanel()

        # Add panel to tabs
        self.tab_widget.addTab(self.main_stats_panel, "Stats")

    def update_stats(self, data_manager):
        # Update main stats panel
        self.main_stats_panel.update_stats(data_manager)
