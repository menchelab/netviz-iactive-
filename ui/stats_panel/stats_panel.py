import logging
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTabWidget

from .main_stats_panel import MainStatsPanel
from .sankey_panel import SankeyPanel
from .layer_graph_panel import LayerGraphPanel
from .layer_influence_panel import LayerInfluencePanel
from .layer_communities_panel import LayerCommunitiesPanel
from .information_flow_panel import InformationFlowPanel

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

        # Create panels for each tab
        self.main_stats_panel = MainStatsPanel()
        self.sankey_panel = SankeyPanel()
        self.layer_graph_panel = LayerGraphPanel()
        self.layer_influence_panel = LayerInfluencePanel()
        self.layer_communities_panel = LayerCommunitiesPanel()
        self.information_flow_panel = InformationFlowPanel()

        # Add panels to tabs
        self.tab_widget.addTab(self.main_stats_panel, "Network Statistics")
        self.tab_widget.addTab(self.sankey_panel, "Sankey Diagram")
        self.tab_widget.addTab(self.layer_graph_panel, "L Graph")
        self.tab_widget.addTab(self.layer_influence_panel, "L Influence")
        self.tab_widget.addTab(self.layer_communities_panel, "L Communities")
        self.tab_widget.addTab(self.information_flow_panel, "Info Flow")

    def update_stats(self, data_manager):
        """Update statistics in all panels"""
        logger = logging.getLogger(__name__)
        
        # Update each panel
        self.main_stats_panel.update_stats(data_manager)
        self.sankey_panel.update_stats(data_manager)
        self.layer_graph_panel.update_stats(data_manager)
        self.layer_influence_panel.update_stats(data_manager)
        self.layer_communities_panel.update_stats(data_manager)
        self.information_flow_panel.update_stats(data_manager) 