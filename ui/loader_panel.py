from PyQt5.QtWidgets import QWidget, QHBoxLayout, QCheckBox, QComboBox, QPushButton
from PyQt5.QtCore import Qt
from data.data_loader import get_available_diseases

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

        # Create load button
        self.load_button = QPushButton("Load Dataset")
        layout.addWidget(self.load_button)
        
        # Add stretch to push everything to the left
        layout.addStretch(1)

        # Set the layout alignment
        layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

    def create_disease_dropdown(self):
        """Create dropdown menu with available disease datasets"""
        combo = QComboBox()
        if self.data_dir:
            diseases = get_available_diseases(self.data_dir)
            for disease in diseases:
                combo.addItem(disease)
        return combo 