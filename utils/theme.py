from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

def enable_dark_mode(app: QApplication):

    dark_palette = app.palette()
    
    # Configure basic palette colors
    dark_palette.setColor(dark_palette.Background, Qt.black)
    dark_palette.setColor(dark_palette.Window, Qt.black)
    dark_palette.setColor(dark_palette.WindowText, Qt.white)
    dark_palette.setColor(dark_palette.Base, Qt.black)
    dark_palette.setColor(dark_palette.AlternateBase, Qt.black)
    dark_palette.setColor(dark_palette.ToolTipBase, Qt.black)  # Dark background for tooltips
    dark_palette.setColor(dark_palette.ToolTipText, Qt.white)  # White text for tooltips
    dark_palette.setColor(dark_palette.Text, Qt.white)
    dark_palette.setColor(dark_palette.Button, Qt.black)
    dark_palette.setColor(dark_palette.ButtonText, Qt.white)
    dark_palette.setColor(dark_palette.Link, Qt.blue)
    
    app.setPalette(dark_palette)

    # Modern dark style with improved scrollbars
    dark_style = """
    QWidget {
        background-color: #1e1e1e;
        color: #ffffff;
    }

    QPushButton {
        background-color: #3d3d3d;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 6px 12px;
        font-weight: 500;
    }

    QPushButton:hover {
        background-color: #4d4d4d;
    }

    QPushButton:pressed {
        background-color: #2d2d2d;
    }

    /* Tooltip styling */
    QToolTip {
        background-color: #2d2d2d;
        color: #ffffff;
        border: 1px solid #3d3d3d;
        border-radius: 4px;
        padding: 4px;
    }

    /* Modern Scrollbar Styling */
    QScrollBar:vertical {
        border: none;
        background-color: #2d2d2d;
        width: 10px;
        margin: 0px;
    }

    QScrollBar::handle:vertical {
        background-color: #5d5d5d;
        border-radius: 5px;
        min-height: 20px;
    }

    QScrollBar::handle:vertical:hover {
        background-color: #6d6d6d;
    }

    QScrollBar::add-line:vertical,
    QScrollBar::sub-line:vertical {
        height: 0px;
    }

    QScrollBar:horizontal {
        border: none;
        background-color: #2d2d2d;
        height: 10px;
        margin: 0px;
    }

    QScrollBar::handle:horizontal {
        background-color: #5d5d5d;
        border-radius: 5px;
        min-width: 20px;
    }

    QScrollBar::handle:horizontal:hover {
        background-color: #6d6d6d;
    }

    QScrollBar::add-line:horizontal,
    QScrollBar::sub-line:horizontal {
        width: 0px;
    }

    QComboBox {
        background-color: #3d3d3d;
        color: white;
        border: 1px solid #4d4d4d;
        border-radius: 4px;
        padding: 4px 8px;
    }

    QComboBox:hover {
        border: 1px solid #5d5d5d;
    }

    QLineEdit {
        background-color: #3d3d3d;
        color: white;
        border: 1px solid #4d4d4d;
        border-radius: 4px;
        padding: 4px 8px;
    }

    QLineEdit:focus {
        border: 1px solid #5d5d5d;
    }

    /* Modern Tab Widget Styling */
    QTabWidget {
        background-color: #1e1e1e;
        border: none;
    }

    QTabWidget::pane {
        border: 1px solid #3d3d3d;
        border-radius: 4px;
    }

    QTabBar::tab {
        background-color: #2d2d2d;
        color: white;
        padding: 8px 16px;
        margin: 0px;
        border: none;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
    }

    QTabBar::tab:selected {
        background-color: #3d3d3d;
    }

    QTabBar::tab:hover:!selected {
        background-color: #353535;
    }
    """
    app.setStyleSheet(dark_style) 