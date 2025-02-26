from PyQt5.QtWidgets import QWidget, QVBoxLayout, QCheckBox, QGroupBox, QFrame, QLabel
from PyQt5.QtGui import QColor
import logging

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
        
        self.display_group.setLayout(display_layout)
        layout.addWidget(self.display_group)
        
        # Add stretch at the bottom to push controls to the top
        layout.addStretch(1)
        
        # Store references to layouts for later updates
        self.layer_layout = layer_layout
        self.cluster_layout = cluster_layout
        self.origin_layout = origin_layout
        self.display_layout = display_layout
    
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
            cb = QCheckBox(f"{layer}")
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
                    logger.info(f"Set layer {layer} color to {color_hex}")
            
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
            cb = QCheckBox(f"{cluster}")
            cb.setChecked(True)
            cb.stateChanged.connect(visibility_callback)
            self.cluster_layout.addWidget(cb)
            self.cluster_checkboxes[cluster] = cb
        
        # Create origin controls
        for origin in unique_origins:
            cb = QCheckBox(f"{origin}")
            cb.setChecked(True)
            cb.stateChanged.connect(visibility_callback)
            self.origin_layout.addWidget(cb)
            self.origin_checkboxes[origin] = cb
        
        # Connect intralayer edges checkbox
        self.intralayer_edges_checkbox.stateChanged.connect(visibility_callback)
    
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