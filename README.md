# YOLOv8 Vehicle Detection

A user-friendly application for vehicle detection using YOLOv8, featuring dataset exploration, model training, and real-time detection with a Streamlit interface.

---

## 🚗 Project Overview

This project implements a vehicle detection system using YOLOv8, one of the most advanced real-time object detection models. The system can detect various types of vehicles (cars, buses, trucks, bicycles, etc.) from images and videos. The Streamlit interface allows you to:
- Explore and visualize the dataset
- Learn about YOLOv8 architecture
- Train custom YOLOv8 models
- Track experiments with MLflow
- Perform real-time vehicle detection

---

The project uses the Indian Vehicle Dataset collected by DataCluster Labs, containing over 50,000 high-resolution images of vehicles captured across 1,000+ urban and rural areas in India. The dataset includes the following vehicle classes:
- Auto
- Bicycle
- Bus
- Car
- Tempo
- Tractor
- Two-wheelers
- Trucks

---

- **Default**: [Indian Vehicle Dataset](https://www.kaggle.com/datasets/dataclusterlabs/indian-vehicle-dataset) (>50,000 images, 8 classes: Auto, Bicycle, Bus, Car, Tempo, Tractor, Two-wheelers, Trucks)
- **Custom**: You can use your own dataset in YOLO format.
- **Note**: The original dataset is in `.xml` format. Use the provided script to convert to YOLO format (see below).

---

## 🗂️ Project Structure

```
Vehicle-Detection-YOLOv8/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Project dependencies
├── LICENSE                 # MIT License
├── README.md               # Project documentation
├── packages.txt            # Additional package list (if needed)
├── src/                    # Source code
│   ├── data_t.py           # Dataset exploration module
│   ├── theory_t.py         # YOLOv8 theory module
│   ├── train_t.py          # Model training module
│   ├── demo_t.py           # Detection demo module
│   └── mlflow_t.py         # MLflow tracking module
├── utils/                  # Utility scripts
│   └── convert_to_yaml.py  # Dataset conversion utility
├── models/                 # Pre-trained models
│   ├── yolov8n.pt
│   ├── yolov8s.pt
│   └── yolov8m.pt
├── yolov8_dataset/         # Dataset directory
│   ├── custom_dataset.yaml # Dataset configuration
│   ├── train/              # Training data
│   ├── val/                # Validation data
│   └── test/               # Test data
├── runs/                   # YOLOv8 training runs/outputs
├── mlruns/                 # MLflow experiment tracking
└── .gitignore              # Git ignore file
```

---

## ✨ Features

- **Data Exploration**: Statistics, class distribution, sample images with annotations
- **YOLOv8 Theory**: Learn about the model architecture
- **Model Training**: Custom training, real-time progress, MLflow experiment tracking
- **Vehicle Detection**: Real-time detection on images/videos, adjustable confidence, result visualization
- **MLflow Integration**: Track and compare experiments, log parameters/metrics, model versioning

---

## ⚡ Quickstart

### 1. Prerequisites
- Python 3.8+
- (Recommended) CUDA-compatible GPU for training

### 2. Installation

```bash
# Clone the repository
$ git clone https://github.com/lbnm203/Vehicle-Detection-YOLOv8.git
$ cd Vehicle-Detection-YOLOv8

# (Optional) Create a virtual environment
$ python -m venv venv
# On Windows:
$ venv\Scripts\activate
# On macOS/Linux:
$ source venv/bin/activate

# Install dependencies
$ pip install -r requirements.txt
```

### 3. Download Pre-trained Models (Optional)

```bash
# Create models directory if not exists
$ mkdir -p models

# Download YOLOv8 nano
$ wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -P models/
# Download YOLOv8 small (optional)
$ wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt -P models/
# Download YOLOv8 medium (optional)
$ wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt -P models/
```
*Windows users*: If you do not have `wget`, please download the files manually and place them in the `models/` directory.

### 4. Prepare the Dataset
- Download the [Indian Vehicle Dataset](https://www.kaggle.com/datasets/dataclusterlabs/indian-vehicle-dataset) or use your own dataset.
- If your dataset is in `.xml` format, run the conversion script:
  ```bash
  python utils/convert_to_xml.py
  ```
- Make sure your data follows the YOLOv8 structure (see the `yolov8_dataset/` folder).
- Update the paths in `yolov8_dataset/custom_dataset.yaml` if needed.

---

## 🚀 Usage

### 1. Run the Streamlit Application
```bash
streamlit run app.py
```
Then open [http://localhost:8501](http://localhost:8501) in your browser.

Or try the online demo: [https://vehicle-detection-yolov8.streamlit.app](https://vehicle-detection-yolov8.streamlit.app)

### 2. Main Features
- Explore the dataset and view sample images
- Learn about YOLOv8
- Train your own model (optional)
- Perform vehicle detection on images or videos

---

## 🏋️‍♂️ Training Your Own Model

> **Note:** If you want to enable training, uncomment the relevant lines in `app.py`.

1. Go to the "Training" tab in the application
2. Select a base model (YOLOv8n, YOLOv8s, YOLOv8m)
3. Set training parameters (epochs, batch size, learning rate, etc.)
4. Click "Start Training"
5. Monitor training progress and results in real-time
6. View and compare experiments in the MLflow tab

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLOv8 implementation
- [DataCluster Labs](https://www.datacluster.ai/) - Indian Vehicle Dataset
- [Streamlit](https://streamlit.io/) - Web application framework
- [MLflow](https://mlflow.org/) - Experiment tracking

---

## 👤 Author

Nhat Minh Le Ba


