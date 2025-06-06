from mlflow import MlflowClient
import mlflow
import streamlit as st
import cv2
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image
import tempfile
import matplotlib.pyplot as plt


def load_model(model_path):
    """Load a YOLOv8 model"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def process_image(model, image, conf_threshold=0.25):
    """Process an image with the model and return results"""
    try:
        results = model.predict(image, conf=conf_threshold)
        return results[0]  # Return the first result
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None


def draw_results(image, results, class_names):
    """Draw bounding boxes and labels on the image"""
    img = image.copy()

    if results is not None and results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(boxes, confs, cls_ids):
            x1, y1, x2, y2 = box.astype(int)
            # Check if cls_id is in range of class_names
            if 0 <= cls_id < len(class_names):
                class_name = class_names[cls_id]
            else:
                class_name = f"Unknown-{cls_id}"

            label = f"{class_name} {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label background
            text_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(
                img, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), (0, 255, 0), -1)

            # Draw label text
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return img


def demo_detection():
    st.header("Demo Vehicle Detection")

    # Model selection
    st.write("### Chọn mô hình")
    model_option = st.selectbox(
        "Chọn model",
        options=["YOLOv8n", "YOLOv8s", "YOLOv8m", "Mô hình đã huấn luyện"],
        index=0
    )

    # Map model option to path
    if model_option == "Mô hình đã huấn luyện":
        # Get MLflow runs to show trained models
        mlflow.set_tracking_uri("./mlruns")
        client = MlflowClient()

        # Get all experiments and their runs
        try:
            experiments = client.search_experiments()
            all_runs = []

            # Check if experiments list is not empty
            if experiments:
                for exp in experiments:
                    try:
                        runs = client.search_runs(
                            experiment_ids=[exp.experiment_id])
                        all_runs.extend(runs)
                    except Exception as e:
                        st.warning(
                            f"Error loading runs for experiment {exp.experiment_id}: {str(e)}")
                        continue
            else:
                st.info("No MLflow experiments found")
        except Exception as e:
            st.warning(f"Error loading MLflow experiments: {str(e)}")
            all_runs = []

        # Get run names and paths
        trained_models = []

        # First check for models in the traditional path (runs/train/*/weights/best.pt)
        model_dir = "runs/train"
        if os.path.exists(model_dir):
            for model_folder in os.listdir(model_dir):
                model_path = os.path.join(
                    model_dir, model_folder, "weights", "best.pt")
                if os.path.exists(model_path):
                    # Get creation time for sorting
                    creation_time = os.path.getctime(model_path)
                    trained_models.append({
                        'name': model_folder,
                        'path': model_path,
                        'creation_time': creation_time,
                        'size': os.path.getsize(model_path)
                    })

        # Sort models by creation time (newest first)
        trained_models.sort(key=lambda x: x['creation_time'], reverse=True)

        if trained_models:
            st.write("#### Danh sách model đã huấn luyện:")

            # Create a more detailed display
            model_info = []
            for i, model in enumerate(trained_models):
                import datetime
                creation_date = datetime.datetime.fromtimestamp(
                    model['creation_time']).strftime('%Y-%m-%d %H:%M:%S')
                size_mb = model['size'] / (1024 * 1024)
                model_info.append(
                    f"**{model['name']}** | Created: {creation_date} | Size: {size_mb:.1f}MB")

            # Display model selection with radio buttons for better visibility
            selected_model_name = st.radio(
                "Chọn model:",
                options=[model['name'] for model in trained_models],
                format_func=lambda x: next(
                    info for info in model_info if info.startswith(f"**{x}**"))
            )

            # Get the selected model path
            model_path = next(
                model['path'] for model in trained_models if model['name'] == selected_model_name)

            # Display additional info about selected model
            selected_model = next(
                model for model in trained_models if model['name'] == selected_model_name)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model Name", selected_model['name'])
            with col2:
                import datetime
                creation_date = datetime.datetime.fromtimestamp(
                    selected_model['creation_time']).strftime('%Y-%m-%d %H:%M')
                st.metric("Created", creation_date)
            with col3:
                size_mb = selected_model['size'] / (1024 * 1024)
                st.metric("Size", f"{size_mb:.1f} MB")

            # Check if results.csv exists for this model
            results_csv_path = os.path.join(
                "runs/train", selected_model['name'], "results.csv")
            if os.path.exists(results_csv_path):
                with st.expander("📊 View Training Results"):
                    import pandas as pd
                    df = pd.read_csv(results_csv_path)

                    # Show last few rows (final metrics)
                    st.write("**Final Training Metrics:**")
                    final_metrics = df.tail(1)

                    # Display key metrics in columns
                    if not final_metrics.empty:
                        metrics_cols = st.columns(4)
                        row = final_metrics.iloc[0]

                        # Define priority metrics to show
                        priority_metrics = [
                            'metrics/mAP50(B)', 'val/mAP50', 'mAP_0.5',
                            'metrics/mAP50-95(B)', 'val/mAP50-95', 'mAP_0.5:0.95',
                            'val/box_loss', 'train/box_loss',
                            'metrics/precision(B)', 'val/precision', 'precision',
                            'metrics/recall(B)', 'val/recall', 'recall'
                        ]

                        displayed_metrics = []

                        # First, try to display priority metrics
                        for metric in priority_metrics:
                            if metric in row and not pd.isna(row[metric]) and len(displayed_metrics) < 4:
                                col_idx = len(displayed_metrics)
                                with metrics_cols[col_idx]:
                                    # Clean up metric name for display
                                    clean_name = metric.replace(
                                        'metrics/', '').replace('(B)', '').replace('val/', 'Val ').replace('train/', 'Train ')
                                    st.metric(clean_name, f"{row[metric]:.4f}")
                                displayed_metrics.append(metric)

                        # If we still have space, show other numeric metrics
                        if len(displayed_metrics) < 4:
                            for col in df.columns:
                                if (col not in displayed_metrics and
                                    col in row and
                                    not pd.isna(row[col]) and
                                    isinstance(row[col], (int, float)) and
                                        len(displayed_metrics) < 4):
                                    col_idx = len(displayed_metrics)
                                    with metrics_cols[col_idx]:
                                        clean_name = col.replace(
                                            'metrics/', '').replace('(B)', '').replace('val/', 'Val ').replace('train/', 'Train ')
                                        st.metric(
                                            clean_name, f"{row[col]:.4f}")
                                    displayed_metrics.append(col)

                        if not displayed_metrics:
                            st.info(
                                "No suitable metrics found in the final epoch.")

                    # Show training curve
                    st.write("**Training Progress:**")

                    # Create separate plots for different metric types
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
                        2, 2, figsize=(15, 10))

                    # Define metric categories with more possible column names
                    loss_metrics = {
                        'Train Box Loss': ['train/box_loss', 'box_loss', 'train_box_loss'],
                        'Val Box Loss': ['val/box_loss', 'val_box_loss'],
                        'Train Cls Loss': ['train/cls_loss', 'cls_loss', 'train_cls_loss'],
                        'Val Cls Loss': ['val/cls_loss', 'val_cls_loss'],
                        'Train DFL Loss': ['train/dfl_loss', 'dfl_loss', 'train_dfl_loss'],
                        'Val DFL Loss': ['val/dfl_loss', 'val_dfl_loss']
                    }

                    map_metrics = {
                        'mAP50': ['metrics/mAP50(B)', 'val/mAP50', 'mAP_0.5', 'mAP50', 'mAP@0.5'],
                        'mAP50-95': ['metrics/mAP50-95(B)', 'val/mAP50-95', 'mAP_0.5:0.95', 'mAP50-95', 'mAP@0.5:0.95']
                    }

                    precision_metrics = {
                        'Precision': ['metrics/precision(B)', 'val/precision', 'precision'],
                    }

                    recall_metrics = {
                        'Recall': ['metrics/recall(B)', 'val/recall', 'recall', 'metrics/recall']
                    }

                    learning_rate_metrics = {
                        'LR PG0': ['lr/pg0', 'lr_pg0'],
                        'LR PG1': ['lr/pg1', 'lr_pg1'],
                        'LR PG2': ['lr/pg2', 'lr_pg2']
                    }

                    colors = ['blue', 'red', 'green',
                              'orange', 'purple', 'brown']

                    # Helper function to find and plot metrics
                    def plot_metrics_category(ax, metrics_dict, title, ylabel, subplot_name):
                        color_idx = 0
                        plotted = False
                        plotted_metrics = []

                        for metric_label, possible_names in metrics_dict.items():
                            for col_name in possible_names:
                                if col_name in df.columns and not df[col_name].isna().all():
                                    ax.plot(df.index, df[col_name],
                                            label=metric_label,
                                            color=colors[color_idx % len(colors)])
                                    plotted_metrics.append(
                                        f"{metric_label}: {col_name}")
                                    color_idx += 1
                                    plotted = True
                                    break

                        if plotted:
                            ax.set_xlabel('Epoch')
                            ax.set_ylabel(ylabel)
                            ax.set_title(title)
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            return plotted_metrics
                        else:
                            # Try to find any column that might be relevant based on keywords
                            relevant_cols = []
                            keywords = {
                                'loss': ['loss'] if 'Loss' in title else [],
                                'map': ['map', 'ap'] if 'mAP' in title else [],
                                'precision_recall': ['precision'] if 'Precision' in title else ['recall'] if 'Recall' in title else [],
                                'lr': ['lr'] if 'Learning' in title else []
                            }

                            for category, kws in keywords.items():
                                if kws:  # If this category has keywords
                                    for col in df.columns:
                                        if any(kw in col.lower() for kw in kws) and col.lower() not in ['epoch']:
                                            if col not in relevant_cols and not df[col].isna().all():
                                                relevant_cols.append(col)

                            if relevant_cols:
                                # st.write(
                                #     f"**{subplot_name} - Found relevant columns:** {relevant_cols}")
                                # Limit to 6 columns
                                for i, col in enumerate(relevant_cols[:6]):
                                    ax.plot(df.index, df[col],
                                            label=col,
                                            color=colors[i % len(colors)])
                                ax.set_xlabel('Epoch')
                                ax.set_ylabel(ylabel)
                                ax.set_title(title)
                                ax.legend()
                                ax.grid(True, alpha=0.3)
                                return [f"Auto-detected: {col}" for col in relevant_cols]
                            else:
                                ax.text(0.5, 0.5, f'No {subplot_name} Data Available',
                                        ha='center', va='center', transform=ax.transAxes)
                                ax.set_title(title)
                                return []

                    # Plot each category
                    loss_plotted = plot_metrics_category(
                        ax1, loss_metrics, 'Training and Validation Losses', 'Loss', 'Loss')
                    map_plotted = plot_metrics_category(
                        ax2, map_metrics, 'Mean Average Precision (mAP)', 'mAP', 'mAP')
                    precision_plotted = plot_metrics_category(
                        ax3, precision_metrics, 'Precision', 'Score', 'Precision')
                    recall_plotted = plot_metrics_category(
                        ax4, recall_metrics, 'Recall', 'Score', 'Recall')
                    # lr_plotted = plot_metrics_category(
                    #     ax4, learning_rate_metrics, 'Learning Rate Schedule', 'Learning Rate', 'Learning Rate')

                    # Show what was actually plotted
                    if any([loss_plotted, map_plotted, precision_plotted, recall_plotted]):
                        all_plotted = []
                        if loss_plotted:
                            all_plotted.extend(
                                [f"**Loss:** {m}" for m in loss_plotted])
                        if map_plotted:
                            all_plotted.extend(
                                [f"**mAP:** {m}" for m in map_plotted])
                        if precision_plotted:
                            all_plotted.extend(
                                [f"**Precision:** {m}" for m in precision_plotted])
                        if recall_plotted:
                            all_plotted.extend(
                                [f"**Recall:** {m}" for m in recall_plotted])
                        # if lr_plotted:
                        #     all_plotted.extend(
                        #         [f"**LR:** {m}" for m in lr_plotted])

                        # st.write("**Successfully plotted metrics:**")
                        # for metric in all_plotted:
                        #     st.write(f"- {metric}")

                    plt.tight_layout()
                    st.pyplot(fig)

        else:
            st.warning("Không tìm thấy model đã huấn luyện nào")
            st.info("💡 Tip: Hãy đi đến tab 'Training' để huấn luyện model trước")
            model_path = "models/yolov8n.pt"  # Default fallback

    else:
        model_paths = {
            "YOLOv8n": "models/yolov8n.pt",
            "YOLOv8s": "models/yolov8s.pt",
            "YOLOv8m": "models/yolov8m.pt",
        }
        model_path = model_paths[model_option]

    # Check if model exists
    if not os.path.exists(model_path):
        st.warning(f"Model không tồn tại tại đường dẫn: {model_path}")

    # Load class names
    try:
        yaml_path = "./yolov8_dataset/custom_dataset.yaml"
        if os.path.exists(yaml_path):
            import yaml
            with open(yaml_path, 'r') as f:
                yaml_content = yaml.safe_load(f)
                class_names = list(yaml_content['names'].values())
        else:
            # Default COCO class names
            class_names = ['auto', 'bicycle', 'bus', 'car',
                           'tempo', 'tractor', 'two_wheelers', 'vehicle_truck']
    except Exception as e:
        st.error(f"Error loading class names: {str(e)}")
        class_names = ['auto', 'bicycle', 'bus', 'car',
                       'tempo', 'tractor', 'two_wheelers', 'vehicle_truck']

    # Confidence threshold
    conf_threshold = st.slider(
        "Ngưỡng Confidence", min_value=0.1, max_value=1.0, value=0.25, step=0.05)

    # Input options
    st.write("### Chọn nguồn đầu vào")
    input_option = st.radio(
        "Chọn nguồn", ["Upload ảnh", "Upload video"])

    # Load model
    model = load_model(model_path)

    if model is None:
        st.error("Không thể tải model. Vui lòng kiểm tra đường dẫn.")
        return

    if input_option == "Upload ảnh":
        uploaded_file = st.file_uploader(
            "Chọn ảnh", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            image_np = np.array(image)

            # Display original image
            st.write("### Ảnh gốc")
            st.image(image, caption="Ảnh đầu vào", use_column_width=True)

            if st.button("Xử lý ảnh"):
                # Process image
                results = process_image(model, image_np, conf_threshold)

                if results is not None:
                    # Draw results
                    output_image = draw_results(image_np, results, class_names)

                    # Display output image
                    st.write("### Kết quả phát hiện")
                    st.image(output_image, caption="Kết quả",
                             use_column_width=True)

                # Display detection details
                if results.boxes is not None and len(results.boxes) > 0:
                    st.write("### Chi tiết phát hiện")

                    # Create a list of detections
                    detections = []
                    boxes = results.boxes.xyxy.cpu().numpy()
                    confs = results.boxes.conf.cpu().numpy()
                    cls_ids = results.boxes.cls.cpu().numpy().astype(int)

                    for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, cls_ids)):
                        # Check if cls_id is in range of class_names
                        if 0 <= cls_id < len(class_names):
                            class_name = class_names[cls_id]
                        else:
                            class_name = f"Unknown-{cls_id}"

                        detections.append({
                            "STT": i+1,
                            "Lớp": class_name,
                            "Confidence": f"{conf:.4f}",
                            "Tọa độ (x1,y1,x2,y2)": f"({int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])})"
                        })

                    # Display as table
                    import pandas as pd
                    df = pd.DataFrame(detections)
                    st.table(df)

    elif input_option == "Upload video":
        uploaded_file = st.file_uploader(
            "Chọn video", type=["mp4", "avi", "mov"])

        if uploaded_file is not None:
            # Save uploaded video to temp file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix='.mp4')
            temp_file.write(uploaded_file.read())
            video_path = temp_file.name

            # Open video
            cap = cv2.VideoCapture(video_path)

            # Get video info
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            st.write(
                f"Video FPS: {fps}, Resolution: {frame_width}x{frame_height}")

            # Process video
            st.write("### Xử lý video")

            # Create a placeholder for the processed video
            video_placeholder = st.empty()

            # Process first frame for preview
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process frame
                results = process_image(model, frame_rgb, conf_threshold)

                # Draw results
                output_frame = draw_results(frame_rgb, results, class_names)

                # Display frame
                video_placeholder.image(
                    output_frame, caption="Frame preview", use_column_width=True)

            # Option to process full video
            if st.button("Xử lý toàn bộ video"):
                # Create output video file
                output_path = "output_video.mp4"
                out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(
                    *'mp4v'), fps, (frame_width, frame_height))

                # Reset video capture
                cap.release()
                cap = cv2.VideoCapture(video_path)

                # Process each frame
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                progress_bar = st.progress(0)

                for i in range(frame_count):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Process frame
                    results = process_image(model, frame_rgb, conf_threshold)

                    # Draw results
                    output_frame = draw_results(
                        frame_rgb, results, class_names)

                    # Convert back to BGR for video writing
                    output_frame_bgr = cv2.cvtColor(
                        output_frame, cv2.COLOR_RGB2BGR)

                    # Write frame to output video
                    out.write(output_frame_bgr)

                    # Update progress
                    progress_bar.progress((i + 1) / frame_count)

                # Release resources
                cap.release()
                out.release()

                # Provide download link
                with open(output_path, 'rb') as f:
                    st.download_button(
                        label="Tải video đã xử lý",
                        data=f,
                        file_name="processed_video.mp4",
                        mime="video/mp4"
                    )

            # Clean up
            cap.release()
            os.unlink(video_path)
