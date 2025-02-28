# Implementation for Layer Communities panel
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QCheckBox, QComboBox, QLabel
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.gridspec as gridspec

from charts.layer_communities import create_layer_communities_chart
from .base_panel import BaseStatsPanel


class LayerCommunitiesPanel(BaseStatsPanel):
    """Panel for Layer Communities visualization"""

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        controls_layout = QHBoxLayout()

        # Add checkbox to enable/disable visualization
        self.enable_checkbox = QCheckBox("Enable")
        self.enable_checkbox.setChecked(False)  # Disabled by default
        self.enable_checkbox.stateChanged.connect(self.on_state_changed)
        controls_layout.addWidget(self.enable_checkbox)

        # Add algorithm dropdown
        algorithm_label = QLabel("Algorithm:")
        controls_layout.addWidget(algorithm_label)

        self.algorithm_dropdown = QComboBox()
        self.algorithm_dropdown.addItems(
            [
                "Louvain",
                "Leiden",
                "Label Propagation",
                "Spectral",
                "Infomap",
                "Fluid",
                "AsyncLPA",
            ]
        )

        self.algorithm_dropdown.currentTextChanged.connect(self.on_algorithm_changed)
        controls_layout.addWidget(self.algorithm_dropdown)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Create figure for Layer Communities
        self.figure = Figure(figsize=(8, 10), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Initialize subplots with GridSpec for custom layout
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 7])
        self.heatmap_ax = self.figure.add_subplot(gs[0])
        self.network_ax = self.figure.add_subplot(gs[1])

        self._current_data = None

    def on_state_changed(self, state):
        if self._current_data and state:
            self.figure.clear()

            # Recreate the GridSpec layout
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 7])
            self.heatmap_ax = self.figure.add_subplot(gs[0])
            self.network_ax = self.figure.add_subplot(gs[1])

            (
                layer_connections,
                layers,
                medium_font,
                large_font,
                visible_layer_indices,
                layer_colors,
            ) = self._current_data

            # Redraw with current algorithm
            create_layer_communities_chart(
                self.heatmap_ax,
                self.network_ax,
                layer_connections,
                layers,
                medium_font,
                large_font,
                visible_layer_indices,
                layer_colors,
                algorithm=self.algorithm_dropdown.currentText(),
            )

            self.figure.tight_layout(pad=1.0)
            self.canvas.draw()
        elif not state:
            self.figure.clear()

            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 7])
            self.heatmap_ax = self.figure.add_subplot(gs[0])
            self.network_ax = self.figure.add_subplot(gs[1])

            self.heatmap_ax.text(
                0.5,
                0.5,
                "Community detection disabled",
                ha="center",
                va="center",
                fontsize=12,
            )
            self.heatmap_ax.axis("off")

            self.network_ax.text(
                0.5,
                0.5,
                "Community detection disabled",
                ha="center",
                va="center",
                fontsize=12,
            )
            self.network_ax.axis("off")

            self.canvas.draw()

    def on_algorithm_changed(self, algorithm):
        if self._current_data and self.enable_checkbox.isChecked():
            self.figure.clear()

            # Recreate the GridSpec layout
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 7])
            self.heatmap_ax = self.figure.add_subplot(gs[0])
            self.network_ax = self.figure.add_subplot(gs[1])

            # Unpack stored data
            (
                layer_connections,
                layers,
                medium_font,
                large_font,
                visible_layer_indices,
                layer_colors,
            ) = self._current_data

            # Redraw with new algorithm
            create_layer_communities_chart(
                self.heatmap_ax,
                self.network_ax,
                layer_connections,
                layers,
                medium_font,
                large_font,
                visible_layer_indices,
                layer_colors,
                algorithm=algorithm,
            )

            self.figure.tight_layout(pad=1.0)
            self.canvas.draw()

    def update_stats(self, data_manager):
        self.figure.clear()

        # Recreate the GridSpec layout
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 7])
        self.heatmap_ax = self.figure.add_subplot(gs[0])
        self.network_ax = self.figure.add_subplot(gs[1])

        # Get data from manager
        layers = data_manager.layers
        visible_layer_indices = data_manager.visible_layers
        layer_colors = data_manager.layer_colors

        # Get layer connections from data manager
        layer_connections = data_manager.get_layer_connections()

        # Define font sizes
        medium_font = {"fontsize": 7}
        large_font = {"fontsize": 9}

        # Store current data for later use
        self._current_data = (
            layer_connections,
            layers,
            medium_font,
            large_font,
            visible_layer_indices,
            layer_colors,
        )

        # Only create visualization if enabled
        if self.enable_checkbox.isChecked():
            create_layer_communities_chart(
                self.heatmap_ax,
                self.network_ax,
                layer_connections,
                layers,
                medium_font,
                large_font,
                visible_layer_indices,
                layer_colors,
                algorithm=self.algorithm_dropdown.currentText(),
            )
        else:
            self.heatmap_ax.text(
                0.5,
                0.5,
                "Community detection disabled",
                ha="center",
                va="center",
                fontsize=12,
            )
            self.heatmap_ax.axis("off")

            self.network_ax.text(
                0.5,
                0.5,
                "Community detection disabled",
                ha="center",
                va="center",
                fontsize=12,
            )
            self.network_ax.axis("off")

        self.figure.tight_layout(pad=1.0)

        self.canvas.draw()
