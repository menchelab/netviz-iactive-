from PyQt5.QtWidgets import QWidget, QVBoxLayout, QCheckBox, QGroupBox, QFrame, QLabel, QComboBox, QSlider, QDoubleSpinBox, QHBoxLayout
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt
import logging
from data.data_loader import get_available_diseases


class ShiftClickCheckBox(QCheckBox):
    """Custom checkbox that can detect shift+click"""

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.parent_control_panel = None
        self.group_name = None

    def mousePressEvent(self, event):
        """Handle mouse press events, detecting shift key"""
        if event.button() == Qt.LeftButton and event.modifiers() & Qt.ShiftModifier:
            # Shift+click detected
            if self.parent_control_panel and self.group_name:
                self.parent_control_panel.handle_shift_click(self, self.group_name)
            else:
                # Fall back to normal behavior if parent or group not set
                super().mousePressEvent(event)
        else:
            # Normal click
            super().mousePressEvent(event)


class ControlPanel(QWidget):
    def __init__(self, parent=None, data_dir=None):
        super().__init__(parent)
        self.data_dir = data_dir

        # Initialize empty control lists
        self.layer_checkboxes = []
        self.cluster_checkboxes = {}
        self.origin_checkboxes = {}

        # Initialize default values for display settings
        self.intralayer_linewidth = 1.0
        self.interlayer_linewidth = 1.0
        self.intralayer_opacity = 0.5
        self.interlayer_opacity = 0.5
        self.node_size = 1.0
        self.node_opacity = 1.0

        # Setup UI
        self.setup_ui()

    def create_labeled_slider(self, label_text, min_val, max_val, default_val, step=0.1, is_float=True):
        """Helper function to create a labeled slider with value display"""
        container = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        
        # Label and value display in horizontal layout
        header_container = QWidget()
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        # Label
        label = QLabel(label_text)
        header_layout.addWidget(label)
        
        # Value display
        if is_float:
            value_display = QLabel(f"{default_val:.4f}")
        else:
            value_display = QLabel(str(default_val))
        header_layout.addWidget(value_display, alignment=Qt.AlignRight)
        
        header_container.setLayout(header_layout)
        layout.addWidget(header_container)
        
        # Slider
        slider = QSlider(Qt.Horizontal)
        if is_float:
            # For float values, we'll multiply by 100 to get integer steps
            slider.setMinimum(int(min_val * 100))
            slider.setMaximum(int(max_val * 100))
            slider.setValue(int(default_val * 100))
            slider.setTickInterval(int(step * 100))
        else:
            slider.setMinimum(min_val)
            slider.setMaximum(max_val)
            slider.setValue(default_val)
            slider.setTickInterval(step)
            
        slider.setTickPosition(QSlider.TicksBelow)
        layout.addWidget(slider)
        
        # Connect slider value changed to update label
        def update_value():
            if is_float:
                val = slider.value() / 100.0
                value_display.setText(f"{val:.2f}")
            else:
                val = slider.value()
                value_display.setText(str(val))
            return val
            
        slider.valueChanged.connect(update_value)
        
        container.setLayout(layout)
        return container, slider

    def setup_ui(self):
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Create containers for controls
        self.layer_group = QGroupBox("Layers")
        self.layer_group.setFlat(True)
        layer_layout = QVBoxLayout()
        layer_layout.setContentsMargins(5, 5, 5, 5)
        layer_layout.setSpacing(2)
        self.layer_group.setLayout(layer_layout)
        layout.addWidget(self.layer_group)

        self.cluster_group = QGroupBox("Clusters")
        self.cluster_group.setFlat(True)
        cluster_layout = QVBoxLayout()
        cluster_layout.setContentsMargins(5, 5, 5, 5)
        cluster_layout.setSpacing(2)
        self.cluster_group.setLayout(cluster_layout)
        layout.addWidget(self.cluster_group)

        self.origin_group = QGroupBox("Origins")
        self.origin_group.setFlat(True)
        origin_layout = QVBoxLayout()
        origin_layout.setContentsMargins(5, 5, 5, 5)
        origin_layout.setSpacing(2)
        self.origin_group.setLayout(origin_layout)
        layout.addWidget(self.origin_group)

        # Add a separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        # Add display options group
        self.display_group = QGroupBox("Display Options")
        self.display_group.setFlat(True)
        display_layout = QVBoxLayout()
        display_layout.setContentsMargins(5, 5, 5, 5)
        display_layout.setSpacing(2)

        # Add orthographic view checkbox
        self.orthographic_view_checkbox = QCheckBox("Orthographic View")
        self.orthographic_view_checkbox.setChecked(True)
        display_layout.addWidget(self.orthographic_view_checkbox)

        # Add intralayer edges checkbox
        self.intralayer_edges_checkbox = QCheckBox("Intralayer Edges")
        self.intralayer_edges_checkbox.setChecked(True)
        display_layout.addWidget(self.intralayer_edges_checkbox)

        # Add show nodes checkbox
        self.show_nodes_checkbox = QCheckBox("Show Nodes")
        self.show_nodes_checkbox.setChecked(True)
        display_layout.addWidget(self.show_nodes_checkbox)

        # Add show node labels checkbox
        self.show_labels_checkbox = QCheckBox("Show Node Labels")
        self.show_labels_checkbox.setChecked(False)
        display_layout.addWidget(self.show_labels_checkbox)

        # Add show stats bars checkbox
        self.show_stats_bars_checkbox = QCheckBox("Show Inter Stats Bars")
        self.show_stats_bars_checkbox.setChecked(False)
        display_layout.addWidget(self.show_stats_bars_checkbox)

        # Add line antialias checkbox
        self.line_antialias_checkbox = QCheckBox("Line Antialiasing")
        self.line_antialias_checkbox.setChecked(True)
        display_layout.addWidget(self.line_antialias_checkbox)

        # Add new sliders
        # Intralayer linewidth
        self.intralayer_width_container, self.intralayer_width_slider = self.create_labeled_slider(
            "Intralayer Width:", 0.0001, 2.0, self.intralayer_linewidth
        )
        display_layout.addWidget(self.intralayer_width_container)

        # Interlayer linewidth
        self.interlayer_width_container, self.interlayer_width_slider = self.create_labeled_slider(
            "Interlayer Width:", 0.0001, 2.0, self.interlayer_linewidth
        )
        display_layout.addWidget(self.interlayer_width_container)

        # Intralayer opacity
        self.intralayer_opacity_container, self.intralayer_opacity_slider = self.create_labeled_slider(
            "Intralayer Opacity:", 0.0001, 5.0, self.intralayer_opacity, 0.05
        )
        display_layout.addWidget(self.intralayer_opacity_container)

        # Interlayer opacity
        self.interlayer_opacity_container, self.interlayer_opacity_slider = self.create_labeled_slider(
            "Interlayer Opacity:", 0.0, 5.0, self.interlayer_opacity, 0.05
        )
        display_layout.addWidget(self.interlayer_opacity_container)

        # Node size
        self.node_size_container, self.node_size_slider = self.create_labeled_slider(
            "Node Size:", 0.1, 10.0, self.node_size
        )
        display_layout.addWidget(self.node_size_container)

        # Node opacity
        self.node_opacity_container, self.node_opacity_slider = self.create_labeled_slider(
            "Node Opacity:", 0.00, 1.0, self.node_opacity, 0.05
        )
        display_layout.addWidget(self.node_opacity_container)

        self.display_group.setLayout(display_layout)
        layout.addWidget(self.display_group)

        # Add stretch at the bottom to push controls to the top
        layout.addStretch(1)

        # Store references to layouts for later updates
        self.layer_layout = layer_layout
        self.cluster_layout = cluster_layout
        self.origin_layout = origin_layout
        self.display_layout = display_layout

    def handle_shift_click(self, checkbox, group_name):
        """Handle shift+click on a checkbox by making it the only checked one in its group"""
        logger = logging.getLogger(__name__)
        logger.info(f"Shift+click detected on {checkbox.text()} in group {group_name}")

        # Determine which group of checkboxes to modify
        if group_name == "layers":
            checkboxes = self.layer_checkboxes
        elif group_name == "clusters":
            checkboxes = list(self.cluster_checkboxes.values())
        elif group_name == "origins":
            checkboxes = list(self.origin_checkboxes.values())
        else:
            logger.warning(f"Unknown checkbox group: {group_name}")
            return

        # Block signals on all checkboxes
        for cb in checkboxes:
            cb.blockSignals(True)

        # Uncheck all checkboxes in the group
        for cb in checkboxes:
            if cb != checkbox:
                cb.setChecked(False)

        # Make sure the clicked checkbox is checked
        checkbox.setChecked(True)

        # Unblock signals on all checkboxes
        for cb in checkboxes:
            cb.blockSignals(False)

        # Manually trigger one update
        logger.info(f"Triggering single update after shift-click on {checkbox.text()}")
        checkbox.stateChanged.emit(checkbox.checkState())

    def update_controls(
        self,
        layers,
        unique_clusters,
        unique_origins,
        visibility_callback,
        layer_colors=None,
        cluster_colors=None,
    ):
        """Update the layer, cluster, and origin controls based on loaded data"""
        logger = logging.getLogger(__name__)
        logger.info("Updating controls...")

        # Clear existing controls
        self.clear_layout(self.layer_layout)
        self.clear_layout(self.cluster_layout)
        self.clear_layout(self.origin_layout)
        self.layer_checkboxes = []
        self.cluster_checkboxes = {}
        self.origin_checkboxes = {}

        # Create layer controls (in reversed order)
        for i, layer in enumerate(reversed(layers)):
            cb = ShiftClickCheckBox(f"{layer}")
            cb.parent_control_panel = self
            cb.group_name = "layers"
            cb.setChecked(True)
            cb.stateChanged.connect(visibility_callback)

            # Set checkbox text color if layer_colors are provided
            if layer_colors and layer in layer_colors:
                color = layer_colors[layer]

                # Convert to QColor
                qcolor = None
                if isinstance(color, str) and color.startswith("#"):
                    qcolor = QColor(color)
                elif isinstance(color, (list, tuple)) and len(color) >= 3:
                    r, g, b = color[:3]
                    if isinstance(r, float) and 0 <= r <= 1:
                        qcolor = QColor(int(r * 255), int(g * 255), int(b * 255))
                    else:
                        qcolor = QColor(r, g, b)

                if qcolor:
                    # Set text color using stylesheet
                    color_hex = qcolor.name()
                    cb.setStyleSheet(f"QCheckBox {{ color: {color_hex}; }}")

            # Add to layout
            self.layer_layout.addWidget(cb)

            # Store in original order for get_visible_layers()
            original_index = len(layers) - i - 1
            # Ensure the list has enough slots
            while len(self.layer_checkboxes) <= original_index:
                self.layer_checkboxes.append(None)
            self.layer_checkboxes[original_index] = cb

        # Create cluster controls
        for cluster in unique_clusters:
            cb = ShiftClickCheckBox(f"{cluster}")
            cb.parent_control_panel = self
            cb.group_name = "clusters"
            cb.setChecked(True)
            cb.stateChanged.connect(visibility_callback)

            # Set checkbox text color if cluster_colors are provided
            if cluster_colors and cluster in cluster_colors:
                color = cluster_colors[cluster]

                # Convert to QColor
                qcolor = None
                if isinstance(color, str) and color.startswith("#"):
                    # Handle both RGB (#RRGGBB) and RGBA (#RRGGBBAA) formats
                    if len(color) == 9:  # #RRGGBBAA format
                        r = int(color[1:3], 16)
                        g = int(color[3:5], 16)
                        b = int(color[5:7], 16)
                        a = int(color[7:9], 16)
                        qcolor = QColor(r, g, b, a)
                    else:  # #RRGGBB format
                        qcolor = QColor(color)
                elif isinstance(color, (list, tuple)) and len(color) >= 3:
                    if len(color) >= 4:  # RGBA
                        r, g, b, a = color[:4]
                        if isinstance(r, float) and 0 <= r <= 1:
                            qcolor = QColor(
                                int(r * 255), int(g * 255), int(b * 255), int(a * 255)
                            )
                        else:
                            qcolor = QColor(r, g, b, a)
                    else:  # RGB
                        r, g, b = color[:3]
                        if isinstance(r, float) and 0 <= r <= 1:
                            qcolor = QColor(int(r * 255), int(g * 255), int(b * 255))
                        else:
                            qcolor = QColor(r, g, b)

                if qcolor:
                    # Set text color using stylesheet
                    color_hex = qcolor.name()
                    cb.setStyleSheet(f"QCheckBox {{ color: {color_hex}; }}")

            self.cluster_layout.addWidget(cb)
            self.cluster_checkboxes[cluster] = cb

        # Create origin controls
        for origin in unique_origins:
            cb = ShiftClickCheckBox(f"{origin}")
            cb.parent_control_panel = self
            cb.group_name = "origins"
            cb.setChecked(True)
            cb.stateChanged.connect(visibility_callback)
            self.origin_layout.addWidget(cb)
            self.origin_checkboxes[origin] = cb

        # Connect orthographic view checkbox
        self.orthographic_view_checkbox.stateChanged.connect(visibility_callback)

        # Connect intralayer edges checkbox
        self.intralayer_edges_checkbox.stateChanged.connect(visibility_callback)

        # Connect show nodes checkbox
        self.show_nodes_checkbox.stateChanged.connect(visibility_callback)

        # Connect show labels checkbox
        self.show_labels_checkbox.stateChanged.connect(visibility_callback)

        # Connect show stats bars checkbox
        self.show_stats_bars_checkbox.stateChanged.connect(visibility_callback)

        # Connect line antialias checkbox
        self.line_antialias_checkbox.stateChanged.connect(visibility_callback)

        # Connect value changed signals for sliders
        self.intralayer_width_slider.valueChanged.connect(visibility_callback)
        self.interlayer_width_slider.valueChanged.connect(visibility_callback)
        self.intralayer_opacity_slider.valueChanged.connect(visibility_callback)
        self.interlayer_opacity_slider.valueChanged.connect(visibility_callback)
        self.node_size_slider.valueChanged.connect(visibility_callback)
        self.node_opacity_slider.valueChanged.connect(visibility_callback)

    def clear_layout(self, layout):
        """Clear all widgets from a layout"""
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def get_visible_layers(self):
        """Get indices of visible layers"""
        return [i for i, cb in enumerate(self.layer_checkboxes) if cb.isChecked()]

    def get_visible_clusters(self):
        """Get names of visible clusters"""
        return [
            cluster for cluster, cb in self.cluster_checkboxes.items() if cb.isChecked()
        ]

    def get_visible_origins(self):
        """Get names of visible origins"""
        return [
            origin for origin, cb in self.origin_checkboxes.items() if cb.isChecked()
        ]

    def show_intralayer_edges(self):
        """Check if intralayer edges should be shown"""
        return self.intralayer_edges_checkbox.isChecked()

    def show_nodes(self):
        """Check if nodes should be shown"""
        return self.show_nodes_checkbox.isChecked()

    def show_labels(self):
        """Check if node labels should be shown"""
        return self.show_labels_checkbox.isChecked()

    def show_stats_bars(self):
        """Check if node stats bars should be shown"""
        return self.show_stats_bars_checkbox.isChecked()

    def use_orthographic_view(self):
        """Check if orthographic view should be used"""
        return self.orthographic_view_checkbox.isChecked()

    def get_line_antialias(self):
        """Check if line antialiasing should be enabled"""
        return self.line_antialias_checkbox.isChecked()

    def get_intralayer_width(self):
        """Get intralayer line width"""
        return self.intralayer_width_slider.value() / 100.0

    def get_interlayer_width(self):
        """Get interlayer line width"""
        return self.interlayer_width_slider.value() / 100.0

    def get_intralayer_opacity(self):
        """Get intralayer line opacity"""
        return self.intralayer_opacity_slider.value() / 100.0

    def get_interlayer_opacity(self):
        """Get interlayer line opacity"""
        return self.interlayer_opacity_slider.value() / 100.0

    def get_node_size(self):
        """Get node size"""
        return self.node_size_slider.value() / 100.0

    def get_node_opacity(self):
        """Get node opacity"""
        return self.node_opacity_slider.value() / 100.0

    def create_disease_dropdown(self):
        """Create dropdown menu with available disease datasets"""
        combo = QComboBox()
        
        diseases = get_available_diseases(self.data_dir)
        
        for disease in diseases:
            combo.addItem(disease)
            
        return combo
