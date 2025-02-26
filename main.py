import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import MultilayerNetworkViz
from utils.logging_setup import setup_logging
from datetime import datetime

def main():
    app_instance = QApplication([])
    logger = setup_logging()
    start_time = datetime.now()
    
    # Set data directory
    data_dir = "Multiplex_DataDiVR/Multiplex_Net_Files"
    
    # Create visualization with just the data directory
    main_widget = MultilayerNetworkViz(data_dir=data_dir)
    
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    logger.info(f"Total setup time: {total_duration:.2f} seconds")
    
    logger.info("Starting viz loop")
    app_instance.exec_()

if __name__ == "__main__":
    main() 