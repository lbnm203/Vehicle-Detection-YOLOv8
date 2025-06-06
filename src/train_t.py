from datetime import datetime
import streamlit as st
import matplotlib.pyplot as plt
import os
import yaml
import tempfile
from ultralytics import YOLO
import pandas as pd
import mlflow


def choose_model():
    st.write("### Chọn mô hình YOLOv8")

    model_option = st.selectbox(
        "Chọn model",
        options=["YOLOv8n", "YOLOv8s", "YOLOv8m"],
        index=0
    )

    # Map model option to path
    model_paths = {
        "YOLOv8n": "models/yolov8n.pt",
        "YOLOv8s": "models/yolov8s.pt",
        "YOLOv8m": "models/yolov8m.pt",
    }

    model_path = model_paths[model_option]

    return model_path, model_option


def download_model(model_name):
    """Download a pre-trained YOLOv8 model if it doesn't exist"""
    model_path = f"models/{model_name}.pt"

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(model_path):
        st.info(f"Downloading {model_name}.pt model...")
        try:
            # Use ultralytics download method
            from ultralytics.utils.downloads import download

            url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_name}.pt"
            download(url, dir="models/")

            if os.path.exists(model_path):
                st.success(f"✅ Downloaded {model_name}.pt successfully!")
                return True
            else:
                st.error(f"❌ Failed to download {model_name}.pt")
                return False
        except Exception as e:
            st.error(f"Error downloading model: {str(e)}")
            return False
    return True  # Model already exists


def train_model():
    st.header("Training Module")

    # Check if dataset exists
    yaml_path = './yolov8_dataset/custom_dataset.yaml'
    if not os.path.exists(yaml_path):
        st.warning(
            "⚠️ Dataset not detected. Please go to the Data tab first to download the dataset.")
        return

    # Chọn model
    model_path, model_name = choose_model()

    # Check if model exists
    if not os.path.exists(model_path):
        st.warning(f"⚠️ Model file not found: {model_path}")
        st.info("Attempting to download the pre-trained model...")

        # Extract model name from path (e.g., "yolov8n" from "models/yolov8n.pt")
        model_filename = os.path.basename(model_path)
        model_name_only = os.path.splitext(model_filename)[0]

        # Try to download the model
        with st.spinner(f"Downloading {model_name_only}.pt..."):
            if download_model(model_name_only):
                st.success(
                    f"✅ Model {model_name_only} downloaded successfully!")
            else:
                # Provide manual download instructions if automatic download fails
                with st.expander("Manual Download Instructions"):
                    st.code("""
    # Download YOLOv8 models
    mkdir -p models
    wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O models/yolov8n.pt
    wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt -O models/yolov8s.pt
    wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt -O models/yolov8m.pt
                    """, language="bash")
                return

    # Upload data config YAML
    # st.write("#### Cấu hình dữ liệu (data.yaml)")
    # yaml_file = st.file_uploader("Upload file data.yaml", type=['yaml', 'yml'])
    # yaml_file = os.path.join(os.path.dirname(
    #     os.path.abspath(__file__)), "yolov8_dataset", "custom_dataset.yaml")
    yaml_file = os.path.abspath("./yolov8_dataset/custom_dataset.yaml")

    # Thêm input cho đường dẫn gốc của dataset
    # st.write("#### Đường dẫn đến thư mục dataset")
    dataset_root = os.path.abspath("./yolov8_dataset")

    # if not os.path.exists(dataset_root):
    #     st.warning(
    #         f"Thư mục '{dataset_root}' không tồn tại. Vui lòng kiểm tra lại đường dẫn.")

    #     # Create directory option
    #     if st.button("Create directory structure"):
    #         try:
    #             # Create base directory
    #             os.makedirs(dataset_root, exist_ok=True)

    #             # Create subdirectories
    #             for split in ['train', 'val', 'test']:
    #                 for subdir in ['images', 'labels']:
    #                     os.makedirs(os.path.join(
    #                         dataset_root, split, subdir), exist_ok=True)

    #             st.success(f"Created directory structure at {dataset_root}")
    #         except Exception as e:
    #             st.error(f"Error creating directories: {str(e)}")

    # Tham số huấn luyện
    st.write("#### Tham số huấn luyện")
    epochs = st.slider("Số epochs", min_value=1, max_value=100, value=30)
    imgsz = st.slider("Kích thước ảnh", min_value=320,
                      max_value=1280, value=640)
    batch_size = st.slider("Batch size", min_value=1, max_value=32, value=16)
    lr = st.number_input("Learning rate", min_value=1e-6,
                         max_value=1.0, value=0.01, format="%.6f")

    # Display selected parameters
    st.write("### Thông số đã chọn:")

    # Create a dictionary of parameters
    params = {
        "Model": model_name,
        "Epochs": epochs,
        "Image Size": imgsz,
        "Batch Size": batch_size,
        "Learning Rate": lr,
        # "Dataset Path": dataset_root
    }

    params_df = pd.DataFrame(params.items(), columns=["Tham số", "Giá trị"])
    st.table(params_df)

    if os.path.exists(yaml_file):
        # Đọc nội dung YAML
        with open(yaml_file, 'r') as f:
            yaml_content = yaml.safe_load(f)

        # Kiểm tra xem YAML có chứa các khóa bắt buộc không
        required_keys = ['train', 'val']
        missing_keys = [
            key for key in required_keys if key not in yaml_content]

        if missing_keys:
            st.error(
                f"File YAML thiếu các khóa bắt buộc: {', '.join(missing_keys)}. Vui lòng đảm bảo file YAML có cả 'train' và 'val'.")

            # Hiển thị nội dung YAML hiện tại
            st.write("Nội dung YAML hiện tại:")
            st.code(yaml.dump(yaml_content), language="yaml")

            # Cung cấp mẫu YAML đúng
            st.write("Mẫu YAML đúng:")
            sample_yaml = {
                'path': dataset_root,
                'train': 'train/images',
                'val': 'val/images',
                'test': 'test/images',
                'names': {
                    0: 'auto',
                    1: 'bicycle',
                    # ... thêm các lớp khác
                }
            }
            st.code(yaml.dump(sample_yaml), language="yaml")

            return

        # Cập nhật đường dẫn tuyệt đối trong YAML
        yaml_content['path'] = dataset_root

        # Kiểm tra xem các thư mục có tồn tại không
        train_path = os.path.join(dataset_root, yaml_content['train'])
        val_path = os.path.join(dataset_root, yaml_content['val'])

        if not os.path.exists(train_path):
            st.error(f"Không tìm thấy thư mục train: {train_path}")
            return

        if not os.path.exists(val_path):
            st.error(f"Không tìm thấy thư mục validation: {val_path}")
            return

        # Lưu tạm file YAML với đường dẫn tuyệt đối
        temp_yaml = tempfile.NamedTemporaryFile(delete=False, suffix='.yaml')
        with open(temp_yaml.name, 'w') as f:
            yaml.dump(yaml_content, f)
        data_cfg = temp_yaml.name

        # # Hiển thị nội dung YAML đã cập nhật
        # st.write("#### Nội dung YAML đã cập nhật:")
        # st.code(yaml.dump(yaml_content), language="yaml")

        # Đặt tên run_name theo ý mình + ngày tháng năm giờ phút giây
        run_name = st.text_input(
            "Tên run", value=model_name) + f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if st.button("Bắt đầu huấn luyện 🔥"):
            with st.spinner("Đang huấn luyện mô hình..."):
                # Setup MLflow tracking
                mlflow.set_tracking_uri("./mlruns")

                # Bắt đầu run MLflow
                with mlflow.start_run(run_name=run_name) as run:
                    st.write(f"MLflow Run ID: {run.info.run_id}")
                    try:
                        # Log parameters
                        mlflow.log_param("model", model_name)
                        mlflow.log_param("epochs", epochs)
                        mlflow.log_param("batch_size", batch_size)
                        mlflow.log_param("learning_rate", lr)
                        mlflow.log_param("image_size", imgsz)
                        mlflow.log_param(
                            "data_config", os.path.basename(data_cfg))
                        mlflow.log_param("dataset_path", dataset_root)

                        # Khởi tạo model
                        model = YOLO(model_path)

                        # Train với run_name để tạo thư mục riêng
                        results = model.train(
                            data=data_cfg,
                            epochs=epochs,
                            imgsz=imgsz,
                            batch=batch_size,
                            lr0=lr,
                            project='runs/train',
                            name=run_name,
                            exist_ok=True
                        )

                        # Thư mục lưu kết quả (sử dụng run_name thay vì model_name)
                        save_dir = os.path.join('runs', 'train', run_name)

                        # Đọc file CSV chứa kết quả huấn luyện
                        results_csv = os.path.join(save_dir, "results.csv")
                        if os.path.exists(results_csv):
                            df = pd.read_csv(results_csv)
                            st.write("#### Training Metrics")
                            st.dataframe(df)

                            # Plot loss curves
                            fig, (ax1, ax2) = plt.subplots(
                                1, 2, figsize=(12, 5))

                            # Plot losses
                            loss_cols = [
                                col for col in df.columns if "loss" in col.lower()]
                            if loss_cols:
                                for col in loss_cols:
                                    if col in df.columns:
                                        ax1.plot(df.index, df[col], label=col)
                                ax1.set_xlabel("Epoch")
                                ax1.set_ylabel("Loss")
                                ax1.set_title("Training Losses")
                                ax1.legend()
                                ax1.grid(True)

                            # Plot metrics
                            metric_cols = [col for col in df.columns if any(
                                metric in col.lower() for metric in ['map', 'precision', 'recall'])]
                            if metric_cols:
                                for col in metric_cols:
                                    if col in df.columns:
                                        ax2.plot(df.index, df[col], label=col)
                                ax2.set_xlabel("Epoch")
                                ax2.set_ylabel("Metric")
                                ax2.set_title("Validation Metrics")
                                ax2.legend()
                                ax2.grid(True)

                            plt.tight_layout()
                            st.pyplot(fig)

                            # Log metrics per epoch to MLflow
                            for idx, row in df.iterrows():
                                for col in df.columns:
                                    val = row[col]
                                    if isinstance(val, (float, int)) and not pd.isna(val):
                                        # Sanitize metric name - replace invalid characters
                                        sanitized_col = col.replace("(", "_").replace(
                                            ")", "_").replace(" ", "_").replace("/", "_")
                                        mlflow.log_metric(
                                            sanitized_col, float(val), step=idx)
                        else:
                            st.warning(
                                f"Không tìm thấy file results.csv tại: {results_csv}")

                        # Log training artifacts
                        weights_dir = os.path.join(save_dir, "weights")
                        if os.path.exists(weights_dir):
                            # Log best model
                            best_model_path = os.path.join(
                                weights_dir, "best.pt")
                            if os.path.exists(best_model_path):
                                mlflow.log_artifact(
                                    best_model_path, artifact_path="models")
                                st.success(
                                    f"✅ Model saved and logged to MLflow: {best_model_path}")

                                # Save the path to session state for later use
                                st.session_state.trained_model_path = best_model_path
                            else:
                                st.warning("Could not find best.pt model file")

                        # Log confusion matrix if exists
                        confusion_matrix_path = os.path.join(
                            save_dir, "confusion_matrix.png")
                        if os.path.exists(confusion_matrix_path):
                            mlflow.log_artifact(
                                confusion_matrix_path, artifact_path="plots")
                            st.image(confusion_matrix_path,
                                     caption="Confusion Matrix")

                        # Log training curves if exist
                        results_png = os.path.join(save_dir, "results.png")
                        if os.path.exists(results_png):
                            mlflow.log_artifact(
                                results_png, artifact_path="plots")
                            st.image(results_png, caption="Training Results")

                        # Display final metrics
                        if hasattr(results, 'results_dict'):
                            final_metrics = results.results_dict
                            st.write("#### Final Training Results:")
                            for key, value in final_metrics.items():
                                if isinstance(value, (int, float)):
                                    # Sanitize metric name - replace invalid characters
                                    sanitized_key = key.replace("(", "_").replace(
                                        ")", "_").replace(" ", "_").replace("/", "_")
                                    mlflow.log_metric(
                                        f"final_{sanitized_key}", value)
                                    st.write(f"**{key}**: {value:.4f}")

                    except Exception as e:
                        st.error(f"Lỗi trong quá trình huấn luyện: {str(e)}")
                        st.exception(e)
                    finally:
                        # Xóa file tạm
                        try:
                            os.unlink(data_cfg)
                        except:
                            pass
    else:
        st.info("Vui lòng upload file data.yaml để bắt đầu huấn luyện.")
