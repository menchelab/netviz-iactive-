from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QSplitter,
)
from PyQt5.QtCore import Qt
import logging
import numpy as np
import os

from ui.network_canvas import NetworkCanvas
from ui.stats_panel import NetworkStatsPanel
from ui.control_panel import ControlPanel
from data.data_loader import get_available_diseases, load_disease_data
from utils.color_utils import hex_to_rgba
from data.network_data_manager import NetworkDataManager
from ui.loader_panel import LoaderPanel


class MultilayerNetworkViz(QWidget):
    def __init__(
        self,
        node_positions=None,
        link_pairs=None,
        link_colors=None,
        node_ids=None,
        layers=None,
        node_clusters=None,
        unique_clusters=None,
        data_dir=None,
    ):
        super().__init__()
        logger = logging.getLogger(__name__)
        logger.info("Initializing visualization...")

        # Store the data directory
        self.data_dir = data_dir

        # Create data manager
        self.data_manager = NetworkDataManager(data_dir)

        # Create layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        self.setLayout(main_layout)

        # Create and add loader panel at the top
        self.loader_panel = LoaderPanel(data_dir=data_dir)
        self.loader_panel.load_button.clicked.connect(self.load_selected_disease)
        main_layout.addWidget(self.loader_panel)

        # Create the main content area with splitters
        self.splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.splitter, 1)

        # Create left panel for controls
        self.control_panel = ControlPanel(data_dir=data_dir)
        self.control_panel.setMinimumWidth(80)
        self.splitter.addWidget(self.control_panel)

        # Create canvas with data manager but don't show it yet
        logger.info("Creating canvas...")
        self.network_canvas = NetworkCanvas(data_manager=self.data_manager)
        self.network_canvas.canvas.native.hide()  # Hide canvas until data is loaded
        self.splitter.addWidget(self.network_canvas.canvas.native)

        # Create stats panel
        self.stats_panel = NetworkStatsPanel()
        self.stats_panel.setMinimumWidth(400)
        self.splitter.addWidget(self.stats_panel)

        # Set initial splitter sizes (control:canvas:stats = 1:3:2)
        self.splitter.setSizes([150, 540, 400])

        # Initialize data attributes
        self.node_positions = None
        self.link_pairs = None
        self.link_colors = None
        self.node_ids = None
        self.layers = None
        self.node_clusters = None
        self.unique_clusters = None
        self.node_origins = None
        self.unique_origins = None

        # Add a flag to prevent multiple updates
        self._updating_visibility = False

        # Minimum sizes for usability
        self.setMinimumWidth(900)
        self.setMinimumHeight(600)

        # If data is provided, load it
        if node_positions is not None:
            self.load_data(
                node_positions,
                link_pairs,
                link_colors,
                node_ids,
                layers,
                node_clusters,
                unique_clusters,
            )
        # Load first dataset if available
        elif self.data_dir and self.loader_panel.disease_combo.count() > 0:
            # Don't automatically load - wait for user to click load button
            pass

        logger.info("Visualization setup complete")
        self.setWindowTitle("DataDiVR - Multiplex")
        self.resize(1200, 768)
        self.show()

    def load_selected_disease(self):
        """Load the currently selected disease when load button is clicked"""
        disease_name = self.loader_panel.disease_combo.currentText()
        if not disease_name:
            return

        # Get ML layout preference from loader panel
        use_ml_layout = self.loader_panel.ml_layout_checkbox.isChecked()
        
        data = load_disease_data(self.data_dir, disease_name, use_ml_layout)
        if data:
            (node_positions, link_pairs, link_colors, node_ids, layers, 
             node_clusters, unique_clusters, node_colors, node_origins, 
             unique_origins, layer_colors) = data
            
            # Load the data into the visualization
            self.load_data(
                node_positions,
                link_pairs,
                link_colors,
                node_ids,
                layers,
                node_clusters,
                unique_clusters,
                node_colors,
                node_origins,
                unique_origins,
                layer_colors
            )
            # Show the canvas after data is loaded
            self.network_canvas.canvas.native.show()

    def load_disease(self, disease_name):
        """Load a disease dataset"""
        if not disease_name:
            return

        # Get ML layout preference from control panel
        use_ml_layout = self.control_panel.ml_layout_checkbox.isChecked()

        data = load_disease_data(self.data_dir, disease_name, use_ml_layout)
        if data:
            (
                node_positions,
                link_pairs,
                link_colors,
                node_ids,
                layers,
                node_clusters,
                unique_clusters,
                node_colors,
                node_origins,
                unique_origins,
                layer_colors,
            ) = data

            # Load the data into the visualization
            self.load_data(
                node_positions,
                link_pairs,
                link_colors,
                node_ids,
                layers,
                node_clusters,
                unique_clusters,
                node_colors,
                node_origins,
                unique_origins,
                layer_colors,
            )

    def load_data(
        self,
        node_positions,
        link_pairs,
        link_colors,
        node_ids=None,
        layers=None,
        node_clusters=None,
        unique_clusters=None,
        node_colors=None,
        node_origins=None,
        unique_origins=None,
        layer_colors=None,
    ):
        """Load network data into the visualization"""

        # Load data into the data manager
        self.data_manager.load_data(
            node_positions,
            link_pairs,
            link_colors,
            node_ids,
            layers,
            node_clusters,
            unique_clusters,
            node_colors,
            node_origins,
            unique_origins,
            layer_colors,
        )

        # Load data into the network canvas
        self.network_canvas.load_data()

        # Update the controls with layer and cluster colors
        self.control_panel.update_controls(
            layers,
            unique_clusters,
            unique_origins,
            self.update_visibility,
            layer_colors,
            self.data_manager.cluster_colors,
        )

        self.update_visibility()

    def update_visibility(self):
        """Update the visibility of nodes and edges based on control panel settings"""
        logger = logging.getLogger(__name__)

        # Prevent recursive calls
        if self._updating_visibility:
            return

        self._updating_visibility = True

        try:
            # Get visibility settings from control panel
            visible_layers = self.control_panel.get_visible_layers()
            visible_clusters = self.control_panel.get_visible_clusters()
            visible_origins = self.control_panel.get_visible_origins()
            show_intralayer = self.control_panel.show_intralayer_edges()
            show_nodes = self.control_panel.show_nodes()
            show_labels = self.control_panel.show_labels()
            show_stats_bars = self.control_panel.show_stats_bars()

            # Track if orthographic setting has changed
            orthographic = self.control_panel.use_orthographic_view()
            if (
                hasattr(self, "_previous_orthographic")
                and self._previous_orthographic == orthographic
            ):
                pass
            else:
                self.network_canvas.set_projection_mode(orthographic)
                self._previous_orthographic = orthographic
                logger.info(
                    f"Projection mode changed to {'orthographic' if orthographic else 'perspective'}"
                )

            logger.debug(
                f"Visibility settings: show_nodes={show_nodes}, show_labels={show_labels}, show_stats_bars={show_stats_bars}"
            )

            filter_changed = False

            if not hasattr(self, "_previous_filters"):
                self._previous_filters = {
                    "layers": set(visible_layers),
                    "clusters": set(visible_clusters),
                    "origins": set(visible_origins),
                }
                filter_changed = True
            else:
                # Compare with previous settings
                if set(visible_layers) != self._previous_filters["layers"]:
                    filter_changed = True
                if set(visible_clusters) != self._previous_filters["clusters"]:
                    filter_changed = True
                if set(visible_origins) != self._previous_filters["origins"]:
                    filter_changed = True

                # Update stored settings
                self._previous_filters["layers"] = set(visible_layers)
                self._previous_filters["clusters"] = set(visible_clusters)
                self._previous_filters["origins"] = set(visible_origins)

            # Only update data manager if filter settings have changed
            if filter_changed:
                # Update visibility in data manager
                node_mask, edge_mask = self.data_manager.update_visibility(
                    visible_layers, visible_clusters, visible_origins
                )

            # Update network canvas with visibility settings
            self.network_canvas.update_visibility(
                show_intralayer=show_intralayer,
                show_nodes=show_nodes,
                show_labels=show_labels,
                show_stats_bars=show_stats_bars,
            )

            # Only update statistics panel if filter settings have changed
            if filter_changed:
                logger.info("Filter settings changed, updating statistics...")
                self.stats_panel.update_stats(self.data_manager)
        finally:
            self._updating_visibility = False
