# YOLOv8 Vehicle Detection (**It's not final version !!!**)

A comprehensive application for vehicle detection using YOLOv8, featuring dataset exploration, model training, and real-time detection with a Streamlit interface.

## Project Overview

This project implements a vehicle detection system using YOLOv8, one of the most advanced real-time object detection models. The system is designed to detect various types of vehicles including cars, buses, trucks, bicycles, and more from images and videos.

The application provides a user-friendly interface built with Streamlit that allows users to:
- Explore and visualize the dataset
- Learn about YOLOv8 architecture and theory
- Train custom YOLOv8 models with different parameters
- Track experiments with MLflow
- Perform real-time vehicle detection on images and videos

## Dataset

The project uses the Indian Vehicle Dataset collected by DataCluster Labs, containing over 50,000 high-resolution images of vehicles captured across 1,000+ urban and rural areas in India. The dataset includes the following vehicle classes:
- Auto
- Bicycle
- Bus
- Car
- Tempo
- Tractor
- Two-wheelers
- Trucks

## Project Structure

```
YOLOv8-Vehicle-Detection/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Project dependencies
├── LICENSE                 # MIT License
├── README.md               # Project documentation
├── src/                    # Source code
│   ├── data_t.py           # Dataset exploration module
│   ├── theory_t.py         # YOLOv8 theory module
│   ├── train_t.py          # Model training module
│   ├── demo_t.py           # Detection demo module
│   └── mlflow_t.py         # MLflow tracking module
├── utils/                  # Utility scripts
│   └── convert_to_xml.py   # Dataset conversion utility
├── models/                 # Pre-trained models
│   ├── yolov8n.pt
│   ├── yolov8s.pt
│   └── yolov8m.pt
├── yolov8_dataset/         # Dataset directory
│   ├── custom_dataset.yaml # Dataset configuration
│   ├── train/              # Training data
│   ├── val/                # Validation data
│   └── test/               # Test data
```

## Features

1. **Data Exploration**
   - Dataset statistics and visualization
   - Class distribution analysis
   - Sample image viewing with annotations

2. **YOLOv8 Theory**
   - Explanation of YOLOv8 architecture
   <!-- - Comparison with previous YOLO versions
   - Object detection concepts -->

3. **Model Training**
   - Custom training with configurable parameters
   - Support for different YOLOv8 model sizes (nano, small, medium)
   - Real-time training progress visualization
   - Experiment tracking with MLflow

4. **Vehicle Detection**
   - Real-time detection on images and videos
   - Adjustable confidence threshold
   - Detection results visualization
   - Detailed detection statistics

5. **MLflow Integration**
   - Experiment tracking and comparison
   - Parameter and metric logging
   - Model versioning
   - Performance visualization

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/lbnm203/YOLOv8-Vehicle-Detection.git
   cd YOLOv8-Vehicle-Detection
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download pre-trained YOLOv8 models:
   ```bash
   mkdir -p models
   
   # Download YOLOv8 nano
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -P models/
   
   # Download YOLOv8 small (optional)
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt -P models/
   
   # Download YOLOv8 medium (optional)
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt -P models/
   ```

5. Prepare the dataset:
   - Download the Indian Vehicle Dataset (https://www.kaggle.com/datasets/dataclusterlabs/indian-vehicle-dataset) or use your own dataset
   - Caution: The original format of this dataset is .xml, so run the `utils/convert_to_yaml.py` to convert it to .yaml format.
   - Organize it in the YOLOv8 format
   - Update the `yolov8_dataset/custom_dataset.yaml` file with your dataset paths

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Access the application in your web browser at `http://localhost:8501`

3. Navigate through the tabs to explore the dataset, learn about YOLOv8, train models, and perform vehicle detection

## Training Your Own Model

1. Navigate to the "Training" tab in the application
2. Select a base model (YOLOv8n, YOLOv8s, or YOLOv8m)
3. Configure training parameters (epochs, batch size, learning rate, etc.)
4. Click "Start Training" to begin the training process
5. Monitor training progress and results in real-time
6. View and compare experiments in the MLflow tab

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv8 implementation
- [DataCluster Labs](https://www.datacluster.ai/) for the Indian Vehicle Dataset
- [Streamlit](https://streamlit.io/) for the web application framework
- [MLflow](https://mlflow.org/) for experiment tracking

## Author

Nhat Minh Le Ba


