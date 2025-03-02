from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QCheckBox,
    QComboBox,
    QPushButton,
    QSlider,
    QLabel,
)
from PyQt5.QtCore import Qt
from data.data_loader import get_available_diseases
from utils.calc_layout import AVAILABLE_LAYOUTS_LOADER


class LoaderPanel(QWidget):
    def __init__(self, data_dir=None, parent=None):
        super().__init__(parent)
        self.data_dir = data_dir
        self.setup_ui()
        self.setFixedHeight(30)

    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)  # Reduced top/bottom padding
        layout.setSpacing(10)

        self.disease_combo = self.create_disease_dropdown()
        self.disease_combo.setMinimumWidth(200)  # Ensure dropdown has reasonable width
        layout.addWidget(self.disease_combo)

        self.ml_layout_checkbox = QCheckBox("ML Layout")
        self.ml_layout_checkbox.setChecked(False)
        layout.addWidget(self.ml_layout_checkbox)

        self.layout_combo = self.create_layout_dropdown()
        self.layout_combo.setMinimumWidth(150)  # Ensure dropdown has reasonable width
        layout.addWidget(self.layout_combo)

        self.z_offset_label = QLabel("Z(n+1)*= Auto")
        layout.addWidget(self.z_offset_label)

        self.z_offset_slider = QSlider(Qt.Horizontal)
        self.z_offset_slider.setMinimum(0)  # 0 = auto
        self.z_offset_slider.setMaximum(10)
        self.z_offset_slider.setValue(0)  # Default to auto
        self.z_offset_slider.setFixedWidth(100)
        self.z_offset_slider.valueChanged.connect(self._update_z_offset_label)
        layout.addWidget(self.z_offset_slider)

        self.load_button = QPushButton("Load Dataset")
        layout.addWidget(self.load_button)

        layout.addStretch(1)

        layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

    def _update_z_offset_label(self):
        value = self.z_offset_slider.value()
        if value == 0:
            self.z_offset_label.setText("Z Offset: Auto")
        else:
            self.z_offset_label.setText(f"Z Offset: {value / 10:.1f}")

    def get_z_offset(self):
        """Get the current z offset value from the slider"""
        value = self.z_offset_slider.value()
        if value == 0:
            return 0.0  # Auto mode
        return value / 10.0

    def create_disease_dropdown(self):
        combo = QComboBox()
        if self.data_dir:
            diseases = get_available_diseases(self.data_dir)
            for disease in diseases:
                combo.addItem(disease)
        return combo

    def create_layout_dropdown(self):
        combo = QComboBox()
        for layout in AVAILABLE_LAYOUTS_LOADER:
            combo.addItem(layout)

        default_index = AVAILABLE_LAYOUTS_LOADER.index("kamada_kawai")
        combo.setCurrentIndex(default_index)
        return combo

    def get_selected_disease(self):
        """Get the currently selected disease"""
        return self.disease_combo.currentText()
        
    def get_prefilter_options(self):
        """
        Get prefiltering options for large networks.
        
        Returns:
        --------
        dict or None
            Dictionary with prefiltering options if any are set, or None if no prefiltering is needed.
            Can include:
            - 'layers': list of layer names to include
            - 'clusters': list of cluster names to include
            - 'max_nodes': maximum number of nodes to include
        """
        # Check if prefiltering is enabled
        if hasattr(self, 'prefilter_checkbox') and self.prefilter_checkbox.isChecked():
            options = {}
            
            # Get selected layers if available
            if hasattr(self, 'layer_list') and self.layer_list.count() > 0:
                selected_layers = []
                for i in range(self.layer_list.count()):
                    item = self.layer_list.item(i)
                    if item.checkState() == Qt.Checked:
                        selected_layers.append(item.text())
                if selected_layers:
                    options['layers'] = selected_layers
            
            # Get selected clusters if available
            if hasattr(self, 'cluster_list') and self.cluster_list.count() > 0:
                selected_clusters = []
                for i in range(self.cluster_list.count()):
                    item = self.cluster_list.item(i)
                    if item.checkState() == Qt.Checked:
                        selected_clusters.append(item.text())
                if selected_clusters:
                    options['clusters'] = selected_clusters
            
            # Get max nodes if available
            if hasattr(self, 'max_nodes_spinbox') and self.max_nodes_spinbox.value() > 0:
                options['max_nodes'] = self.max_nodes_spinbox.value()
            
            return options if options else None
        
        return None
        
    def set_loading(self, is_loading):
        """Set the loading state of the panel"""
        if hasattr(self, 'load_button'):
            self.load_button.setEnabled(not is_loading)
            self.load_button.setText("Loading..." if is_loading else "Load Network")
