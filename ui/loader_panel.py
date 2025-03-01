from PyQt5.QtWidgets import QWidget, QHBoxLayout, QCheckBox, QComboBox, QPushButton, QSlider, QLabel
from PyQt5.QtCore import Qt
from data.data_loader import get_available_diseases
from utils.calc_layout import AVAILABLE_LAYOUTS_LOADER

class LoaderPanel(QWidget):
    def __init__(self, data_dir=None, parent=None):
        super().__init__(parent)
        self.data_dir = data_dir
        self.setup_ui()
        # Set fixed height for the panel
        self.setFixedHeight(35)  # Reduced from 50 to 35

    def setup_ui(self):
        # Create horizontal layout with minimal margins
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)  # Reduced top/bottom padding
        layout.setSpacing(10)

        # Create disease dropdown
        self.disease_combo = self.create_disease_dropdown()
        self.disease_combo.setMinimumWidth(200)  # Ensure dropdown has reasonable width
        layout.addWidget(self.disease_combo)

        # Create ML Layout checkbox
        self.ml_layout_checkbox = QCheckBox("ML Layout")
        self.ml_layout_checkbox.setChecked(False)
        layout.addWidget(self.ml_layout_checkbox)

        # Create layout algorithm dropdown
        self.layout_combo = self.create_layout_dropdown()
        self.layout_combo.setMinimumWidth(150)  # Ensure dropdown has reasonable width
        layout.addWidget(self.layout_combo)

        # Create z offset slider and label
        self.z_offset_label = QLabel("Z(n+1)*= Auto")
        layout.addWidget(self.z_offset_label)
        
        self.z_offset_slider = QSlider(Qt.Horizontal)
        self.z_offset_slider.setMinimum(0)  # 0 = auto
        self.z_offset_slider.setMaximum(10) 
        self.z_offset_slider.setValue(0)  # Default to auto
        self.z_offset_slider.setFixedWidth(100)
        self.z_offset_slider.valueChanged.connect(self._update_z_offset_label)
        layout.addWidget(self.z_offset_slider)

        # Create load button
        self.load_button = QPushButton("Load Dataset")
        layout.addWidget(self.load_button)
        
        # Add stretch to push everything to the left
        layout.addStretch(1)

        # Set the layout alignment
        layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

    def _update_z_offset_label(self):
        """Update the z offset label with the current value"""
        value = self.z_offset_slider.value()
        if value == 0:
            self.z_offset_label.setText("Z Offset: Auto")
        else:
            self.z_offset_label.setText(f"Z Offset: {value/10:.1f}")

    def get_z_offset(self):
        """Get the current z offset value from the slider"""
        value = self.z_offset_slider.value()
        if value == 0:
            return 0.0  # Auto mode
        return value / 10.0

    def create_disease_dropdown(self):
        """Create dropdown menu with available disease datasets"""
        combo = QComboBox()
        if self.data_dir:
            diseases = get_available_diseases(self.data_dir)
            for disease in diseases:
                combo.addItem(disease)
        return combo

    def create_layout_dropdown(self):
        """Create dropdown menu with available layout algorithms"""
        combo = QComboBox()
        for layout in AVAILABLE_LAYOUTS_LOADER:
            combo.addItem(layout)
        # Set default to cluster_centric
        default_index = AVAILABLE_LAYOUTS_LOADER.index("cluster_centric")
        combo.setCurrentIndex(default_index)
        return combo 