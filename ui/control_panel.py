from PyQt5.QtWidgets import QWidget, QVBoxLayout, QCheckBox, QGroupBox
import logging

class ControlPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
        # Initialize empty control lists
        self.layer_checkboxes = []
        self.cluster_checkboxes = {}
        self.origin_checkboxes = {}
        
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
        
        # Add stretch at the bottom to push controls to the top
        layout.addStretch(1)
        
        # Store references to layouts for later updates
        self.layer_layout = layer_layout
        self.cluster_layout = cluster_layout
        self.origin_layout = origin_layout
    
    def update_controls(self, layers, unique_clusters, unique_origins, visibility_callback):
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
        
        # Create layer controls
        for i, layer in enumerate(layers):
            cb = QCheckBox(f"{layer}")
            cb.setChecked(True)
            cb.stateChanged.connect(visibility_callback)
            self.layer_layout.addWidget(cb)
            self.layer_checkboxes.append(cb)
        
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