from PyQt5.QtWidgets import QWidget, QVBoxLayout, QCheckBox, QGroupBox, QFrame, QLabel
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt
import logging

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
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Initialize empty control lists
        self.layer_checkboxes = []
        self.cluster_checkboxes = {}
        self.origin_checkboxes = {}
        
        # Setup UI (which will create the intralayer_edges_checkbox)
        self.setup_ui()
        
        # Initialize the checkbox if it wasn't created in setup_ui
        if not hasattr(self, 'intralayer_edges_checkbox') or self.intralayer_edges_checkbox is None:
            self.intralayer_edges_checkbox = QCheckBox("Intralayer Edges")
            self.intralayer_edges_checkbox.setChecked(True)  # On by default
            self.display_layout.addWidget(self.intralayer_edges_checkbox)
        
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
        
        # Add intralayer edges checkbox
        self.intralayer_edges_checkbox = QCheckBox("Intralayer Edges")
        self.intralayer_edges_checkbox.setChecked(True)  # On by default
        display_layout.addWidget(self.intralayer_edges_checkbox)
        
        # Add show nodes checkbox
        self.show_nodes_checkbox = QCheckBox("Show Nodes")
        self.show_nodes_checkbox.setChecked(True)  # On by default
        display_layout.addWidget(self.show_nodes_checkbox)
        
        # Add show node labels checkbox
        self.show_labels_checkbox = QCheckBox("Show Node Labels")
        self.show_labels_checkbox.setChecked(False)  # Off by default
        self.show_labels_checkbox.stateChanged.connect(self.on_show_labels_changed)
        display_layout.addWidget(self.show_labels_checkbox)
        
        # Add top layer labels only checkbox
        self.bottom_labels_only_checkbox = QCheckBox("Bottom Layer Labels Only") # actually is top layer but well
        self.bottom_labels_only_checkbox.setChecked(True) 
        self.bottom_labels_only_checkbox.setEnabled(False)
        display_layout.addWidget(self.bottom_labels_only_checkbox)
        
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
    
    def update_controls(self, layers, unique_clusters, unique_origins, visibility_callback, layer_colors=None):
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
                if isinstance(color, str) and color.startswith('#'):
                    qcolor = QColor(color)
                elif isinstance(color, (list, tuple)) and len(color) >= 3:
                    r, g, b = color[:3]
                    if isinstance(r, float) and 0 <= r <= 1:
                        qcolor = QColor(int(r*255), int(g*255), int(b*255))
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
        
        # Connect intralayer edges checkbox
        self.intralayer_edges_checkbox.stateChanged.connect(visibility_callback)
        
        # Connect show nodes checkbox
        self.show_nodes_checkbox.stateChanged.connect(visibility_callback)
        
        # Connect show labels checkbox
        self.show_labels_checkbox.stateChanged.connect(visibility_callback)
        
        # Connect bottom layer labels only checkbox
        self.bottom_labels_only_checkbox.stateChanged.connect(visibility_callback)
    
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
        return [cluster for cluster, cb in self.cluster_checkboxes.items() if cb.isChecked()]
    
    def get_visible_origins(self):
        """Get names of visible origins"""
        return [origin for origin, cb in self.origin_checkboxes.items() if cb.isChecked()]
    
    def show_intralayer_edges(self):
        """Check if intralayer edges should be shown"""
        return self.intralayer_edges_checkbox.isChecked()
    
    def show_nodes(self):
        """Check if nodes should be shown"""
        return self.show_nodes_checkbox.isChecked()
    
    def show_labels(self):
        """Check if node labels should be shown"""
        return self.show_labels_checkbox.isChecked()
    
    def show_bottom_labels_only(self):
        """Check if only top layer labels should be shown"""
        return self.bottom_labels_only_checkbox.isChecked()

    def on_show_labels_changed(self, state):
        """Enable/disable the top layer labels checkbox based on show labels state"""
        self.bottom_labels_only_checkbox.setEnabled(state == Qt.Checked)
        
        # If show labels is unchecked, also uncheck the top layer labels checkbox
        if state == Qt.Unchecked:
            self.bottom_labels_only_checkbox.setChecked(False)