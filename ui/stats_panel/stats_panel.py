import logging
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTabWidget

from .main_stats_panel import MainStatsPanel
from .sankey_panel import SankeyPanel
from .layer_graph_panel import LayerGraphPanel
from .layer_influence_panel import LayerInfluencePanel
from .layer_communities_panel import LayerCommunitiesPanel
from .information_flow_panel import InformationFlowPanel
from .layer_coupling_panel import LayerCouplingPanel
from .critical_structure_panel import CriticalStructurePanel
from .hyperbolic_embedding_panel import HyperbolicEmbeddingPanel

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
        
        # Create advanced analytics panels
        self.layer_coupling_panel = LayerCouplingPanel()
        self.critical_structure_panel = CriticalStructurePanel()
#        self.hyperbolic_embedding_panel = HyperbolicEmbeddingPanel()

        # Add panels to tabs
        self.tab_widget.addTab(self.main_stats_panel, "stats")
        self.tab_widget.addTab(self.sankey_panel, "test")
        self.tab_widget.addTab(self.layer_graph_panel, "L Graph")
        self.tab_widget.addTab(self.layer_influence_panel, "L Influence")
        self.tab_widget.addTab(self.layer_communities_panel, "L Communities")
        self.tab_widget.addTab(self.information_flow_panel, "test InfoFlow")
        
        # Add advanced analytics tabs
        self.tab_widget.addTab(self.layer_coupling_panel, "L Coupling")
        self.tab_widget.addTab(self.critical_structure_panel, "test critstruct")
#        self.tab_widget.addTab(self.hyperbolic_embedding_panel, "Hyperbolic View")

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
        
        # Update advanced analytics panels
        self.layer_coupling_panel.update_stats(data_manager)
        self.critical_structure_panel.update_stats(data_manager)
#        self.hyperbolic_embedding_panel.update_stats(data_manager) 