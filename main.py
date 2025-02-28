from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from ui.main_window import MultilayerNetworkViz
from utils.logging_setup import setup_logging
from datetime import datetime
def enable_dark_mode(app):
    # Set dark palette
    dark_palette = app.palette()
    
    dark_palette.setColor(dark_palette.Background, Qt.black)
    dark_palette.setColor(dark_palette.Window, Qt.black)
    dark_palette.setColor(dark_palette.WindowText, Qt.white)
    dark_palette.setColor(dark_palette.Base, Qt.black)
    dark_palette.setColor(dark_palette.AlternateBase, Qt.black)
    dark_palette.setColor(dark_palette.ToolTipBase, Qt.white)
    dark_palette.setColor(dark_palette.ToolTipText, Qt.white)
    dark_palette.setColor(dark_palette.Text, Qt.white)
    dark_palette.setColor(dark_palette.Button, Qt.black)
    dark_palette.setColor(dark_palette.ButtonText, Qt.white)
    dark_palette.setColor(dark_palette.Link, Qt.blue)
    
    app.setPalette(dark_palette)

    # Apply custom dark style (for extra customization)
    dark_style = """
    QWidget {
        background-color: #2e2e2e;
        color: white;
    }

    QPushButton {
        background-color: #444444;
        color: white;
        border: none;
        padding: 5px 10px;
    }

    QPushButton:hover {
        background-color: #666666;
    }

    QScrollBar {
        background-color: #444444;
    }

    QScrollBar::handle {
        background-color: #666666;
    }

    QComboBox {
        background-color: #444444;
        color: white;
        border: 1px solid #666666;
    }

    QLineEdit {
        background-color: #444444;
        color: white;
        border: 1px solid #666666;
    }

    /* Dark style for tab widget */
    QTabWidget {
        background-color: #333333;
        border: 1px solid #444444;
    }

    QTabWidget::tab-bar {
        alignment: center;
    }

    QTabBar::tab {
        background-color: #444444;
        color: white;
        padding: 5px;
        margin: 0px;
        border: 1px solid #666666;
    }

    QTabBar::tab:selected {
        background-color: #666666;
        border: 1px solid #888888;
    }

    QTabBar::tab:hover {
        background-color: #555555;
    }
    """
    app.setStyleSheet(dark_style)

def main():
    app = QApplication([])
    enable_dark_mode(app)

    logger = setup_logging()
    start_time = datetime.now()

    data_dir = "Multiplex_DataDiVR/Multiplex_Net_Files"

    main_widget = MultilayerNetworkViz(data_dir=data_dir)

    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    logger.info(f"Total setup time: {total_duration:.2f} seconds")

    logger.info("Starting viz loop")
    app.exec_()


if __name__ == "__main__":
    main()
