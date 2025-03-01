from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from ui.main_window import MultilayerNetworkViz
from utils.logging_setup import setup_logging
from utils.theme import enable_dark_mode
from datetime import datetime

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
