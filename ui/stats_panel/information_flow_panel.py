# Implementation for Information Flow panel
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QCheckBox, QComboBox, QLabel
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.gridspec as gridspec

from charts.information_flow import create_information_flow_chart
from .base_panel import BaseStatsPanel


class InformationFlowPanel(BaseStatsPanel):
    """Panel for Information Flow visualization"""

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Add controls
        controls_layout = QHBoxLayout()

        # Add checkbox to enable/disable visualization
        self.enable_checkbox = QCheckBox("Enable")
        self.enable_checkbox.setChecked(False)  # Disabled by default
        self.enable_checkbox.stateChanged.connect(self.on_state_changed)
        controls_layout.addWidget(self.enable_checkbox)

        # Add flow metric dropdown
        controls_layout.addWidget(QLabel("Flow Metric:"))
        self.flow_metric_dropdown = QComboBox()
        self.flow_metric_dropdown.addItems(
            ["Betweenness Centrality", "Flow Betweenness", "Information Centrality"]
        )
        self.flow_metric_dropdown.currentTextChanged.connect(self.on_metric_changed)
        controls_layout.addWidget(self.flow_metric_dropdown)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Create figure for Information Flow
        self.figure = Figure(figsize=(8, 10), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Initialize subplots with GridSpec for custom layout
        # First row takes 30% of height, second row takes 70%
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 7])
        self.flow_ax = self.figure.add_subplot(gs[0])  # Top row (30%)
        self.network_ax = self.figure.add_subplot(gs[1])  # Bottom row (70%)

        # Store current data
        self._current_data = None

    def on_state_changed(self, state):
        """Handle enable/disable state change"""
        if self._current_data and state:
            self.figure.clear()

            # Recreate the GridSpec layout
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 7])
            self.flow_ax = self.figure.add_subplot(gs[0])
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

            # Redraw with current metric
            create_information_flow_chart(
                self.flow_ax,
                self.network_ax,
                layer_connections,
                layers,
                medium_font,
                large_font,
                visible_layer_indices,
                layer_colors,
                metric=self.flow_metric_dropdown.currentText(),
            )

            self.figure.tight_layout(pad=1.0)
            self.canvas.draw()
        elif not state:
            # Clear the visualization when disabled
            self.figure.clear()

            # Recreate the GridSpec layout
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 7])
            self.flow_ax = self.figure.add_subplot(gs[0])
            self.network_ax = self.figure.add_subplot(gs[1])

            self.flow_ax.text(
                0.5,
                0.5,
                "Information flow disabled",
                ha="center",
                va="center",
                fontsize=12,
            )
            self.flow_ax.axis("off")

            self.network_ax.text(
                0.5,
                0.5,
                "Information flow disabled",
                ha="center",
                va="center",
                fontsize=12,
            )
            self.network_ax.axis("off")

            self.canvas.draw()

    def on_metric_changed(self, metric):
        """Handle flow metric change"""
        if self._current_data and self.enable_checkbox.isChecked():
            self.figure.clear()

            # Recreate the GridSpec layout
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 7])
            self.flow_ax = self.figure.add_subplot(gs[0])
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

            # Redraw with new metric
            create_information_flow_chart(
                self.flow_ax,
                self.network_ax,
                layer_connections,
                layers,
                medium_font,
                large_font,
                visible_layer_indices,
                layer_colors,
                metric=metric,
            )

            self.figure.tight_layout(pad=1.0)
            self.canvas.draw()

    def update_stats(self, data_manager):
        """Update the Information Flow visualization with current data"""
        # Clear figure
        self.figure.clear()

        # Recreate the GridSpec layout
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 7])
        self.flow_ax = self.figure.add_subplot(gs[0])
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
            create_information_flow_chart(
                self.flow_ax,
                self.network_ax,
                layer_connections,
                layers,
                medium_font,
                large_font,
                visible_layer_indices,
                layer_colors,
                metric=self.flow_metric_dropdown.currentText(),
            )
        else:
            self.flow_ax.text(
                0.5,
                0.5,
                "Information flow disabled",
                ha="center",
                va="center",
                fontsize=12,
            )
            self.flow_ax.axis("off")

            self.network_ax.text(
                0.5,
                0.5,
                "Information flow disabled",
                ha="center",
                va="center",
                fontsize=12,
            )
            self.network_ax.axis("off")

        # Apply tight layout
        self.figure.tight_layout(pad=1.0)

        # Draw canvas
        self.canvas.draw()
