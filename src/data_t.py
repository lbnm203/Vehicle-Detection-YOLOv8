import streamlit as st
import os
import yaml
import matplotlib.pyplot as plt
import random
from PIL import Image
import numpy as np
import matplotlib.patches as patches
import shutil
import gdown
import zipfile
import traceback


@st.cache_resource
def download_dataset_from_gdrive():
    try:
        st.info("Downloading dataset .zip from Google Drive...")

        # ✅ File ID từ link của bạn
        file_id = "1KNgMtAIjTXRCx-Q-xLb_dida3YDnRG5K"
        zip_output = "yolov8_dataset.zip"

        # Tải file .zip
        gdown.download(
            url=f"https://drive.google.com/uc?id={file_id}",
            output=zip_output,
            quiet=False
        )

        # Giải nén
        with zipfile.ZipFile(zip_output, 'r') as zip_ref:
            zip_ref.extractall(".")

        # Kiểm tra kết quả
        yaml_path = "yolov8_dataset/custom_dataset.yaml"
        if os.path.exists(yaml_path):
            st.success("✅ Dataset downloaded and extracted successfully!")

            # Verify that the directories contain files
            train_img_dir = "yolov8_dataset/train/images"
            train_label_dir = "yolov8_dataset/train/labels"

            if os.path.exists(train_img_dir) and os.path.exists(train_label_dir):
                img_count = len([f for f in os.listdir(
                    train_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
                label_count = len([f for f in os.listdir(
                    train_label_dir) if f.endswith('.txt')])

                st.info(
                    f"Found {img_count} images and {label_count} label files in the training set.")

                if img_count == 0 or label_count == 0:
                    st.warning(
                        "Dataset directories are empty. The download may be incomplete.")

            # Clean up zip file
            try:
                os.remove(zip_output)
            except:
                pass

            return True
        else:
            st.error("❌ `custom_dataset.yaml` not found after extraction.")
            return False

    except Exception as e:
        st.error(f"Error downloading dataset: {str(e)}")
        st.code(traceback.format_exc())

        return False


def data_description():
    st.write("### Tập Dữ Liệu Indian Vehicle Dataset")
    # Check if dataset exists locally
    yaml_path = './yolov8_dataset/custom_dataset.yaml'

    # Create base directory if it doesn't exist
    os.makedirs('./yolov8_dataset', exist_ok=True)

    if not os.path.exists(yaml_path):
        st.warning("Dataset not found locally.")

        # Create a more informative message for Streamlit web deployment
        st.info("""
        ### Dataset Options:
        
        1. ✅ Download automatically from **Google Drive**
        2. 📥 Or download manually from [Kaggle](https://www.kaggle.com/datasets/dataclusterlabs/indian-vehicle-dataset)
        
        **Note for Streamlit Web Deployment:** 
        If you're running this app on Streamlit Cloud, you'll need to download the dataset first.
        """)

        # Button to download from Google Drive
        if st.button("Download from Google Drive (.zip)"):
            success = download_dataset_from_gdrive()
            if success:
                st.rerun()  # Use st.rerun() instead of st.experimental_rerun()
            else:
                st.error(
                    "Failed to download dataset. Please try again or download manually.")
                return
        return

    # Display dataset information
    st.write("Tập dữ liệu này được thu thâp bởi DataCluster Labs. Bộ dữ liệu này là một tập hợp gồm hơn 50.000 hình ảnh xe gốc được chụp và thu thập từ hơn 1000 khu vực thành thị và nông thôn, trong đó mỗi hình ảnh đều được các chuyên gia về thị giác máy tính tại Datacluster Labs xem xét và xác minh thủ công")

    # Display dataset information
    st.write("#### Một số thông tin về tập dữ liệu:")
    st.write("- **Kích thước tập dữ liệu:** 50.000+ hình ảnh")
    st.write("- **Được chụp bởi:** Hơn 1000+ người đóng góp cộng đồng")
    st.write("- **Độ phân giải:** 100% hình ảnh là HD trở lên (1920x1080 trở lên)")
    st.write("- **Địa điểm:** Chụp ảnh tại hơn 1000 thành phố trên khắp Ấn Độ")
    st.write("- **Tính đa dạng:** Nhiều điều kiện ánh sáng khác nhau như ngày, đêm, khoảng cách khác nhau, điểm nhìn, ...")
    st.write(
        "- **Thiết bị sử dụng:** Chụp bằng điện thoại di động trong năm 2020-2022")
    st.write("- **Công dụng:** Phát hiện xe cộ, Phát hiện ô tô, Phát hiện xe xây dựng, Hệ thống tự lái, ...")
    st.write("- **Những lớp/phương tiện có trong tập dữ liệu:** auto, bicycle, bus, car, tempo, tractor, two_wheelers, vehicle_truck")
    st.write("---")

    try:
        # Read YAML file
        with open(yaml_path, 'r') as f:
            dataset_config = yaml.safe_load(f)

        # Get class names
        class_names = list(dataset_config['names'].values())

        # Get base directory
        base_dir = os.path.dirname(yaml_path)

        # Construct paths properly
        train_path = dataset_config.get('train', 'train/images')
        val_path = dataset_config.get('val', 'val/images')
        test_path = dataset_config.get('test', 'test/images')

        # Remove any leading ./ from paths
        train_path = train_path[2:] if train_path.startswith(
            './') else train_path
        val_path = val_path[2:] if val_path.startswith('./') else val_path
        test_path = test_path[2:] if test_path.startswith('./') else test_path

        # Construct absolute paths
        train_dir = os.path.normpath(os.path.join(base_dir, train_path))
        val_dir = os.path.normpath(os.path.join(base_dir, val_path))
        test_dir = os.path.normpath(os.path.join(base_dir, test_path))

        # Create directories if they don't exist
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Also create label directories
        train_label_dir = train_dir.replace('images', 'labels')
        val_label_dir = val_dir.replace('images', 'labels')
        test_label_dir = test_dir.replace('images', 'labels')

        os.makedirs(train_label_dir, exist_ok=True)
        os.makedirs(val_label_dir, exist_ok=True)
        os.makedirs(test_label_dir, exist_ok=True)

        # Check if directories have images
        train_images = [f for f in os.listdir(
            train_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        val_images = [f for f in os.listdir(
            val_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        test_images = [f for f in os.listdir(
            test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

        if not train_images or not val_images:
            st.warning(
                "Dataset directories exist but no images found. You need to download the dataset.")

            if st.button("Download Dataset Now"):
                success = download_dataset_from_gdrive()
                if success:
                    st.rerun()  # Use st.rerun() instead of st.experimental_rerun()
                else:
                    st.error("Failed to download dataset automatically.")
                    st.info(
                        "Please download manually from [Kaggle](https://www.kaggle.com/datasets/dataclusterlabs/indian-vehicle-dataset) and extract to the yolov8_dataset directory.")
                return
            return

        # Count images in each set
        train_count = len(train_images)
        val_count = len(val_images)
        test_count = len(test_images)

        # Plot image distribution
        st.write("#### Phân bố số lượng hình ảnh trong các tập dữ liệu")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(['Train', 'Val', 'Test'], [train_count, val_count, test_count],
               color=['blue', 'orange', 'green'])
        ax.set_title('Phân bố số lượng hình ảnh trong các tập dữ liệu')
        ax.set_xlabel('Tập dữ liệu')
        ax.set_ylabel('Số lượng hình ảnh')
        st.pyplot(fig)

        # Analyze dataset
        analyze_dataset(train_dir, class_names)
    except Exception as e:
        st.error(f"Error analyzing dataset: {str(e)}")
        st.info("Please make sure the dataset is properly downloaded and configured.")

        # Provide more detailed error information
        import traceback
        st.code(traceback.format_exc())

        # Offer to create a sample dataset structure
        if st.button("Create Sample Dataset Structure"):
            create_sample_dataset_structure()
            st.success(
                "Created sample dataset structure. Please download the actual dataset files.")
            st.rerun()  # Use st.rerun() instead of st.experimental_rerun()


def create_sample_dataset_structure():
    """Create a sample dataset structure with the necessary directories"""
    try:
        # Create base directory
        os.makedirs('./yolov8_dataset', exist_ok=True)

        # Create subdirectories
        for split in ['train', 'val', 'test']:
            for subdir in ['images', 'labels']:
                os.makedirs(
                    f'./yolov8_dataset/{split}/{subdir}', exist_ok=True)

        # Create a sample YAML file
        yaml_content = {
            'path': './yolov8_dataset',
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'names': {
                0: 'auto',
                1: 'bicycle',
                2: 'bus',
                3: 'car',
                4: 'tempo',
                5: 'tractor',
                6: 'two_wheelers',
                7: 'vehicle_truck'
            }
        }

        with open('./yolov8_dataset/custom_dataset.yaml', 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        return True
    except Exception as e:
        st.error(f"Error creating sample dataset structure: {str(e)}")
        return False


# b. Phân tích phân bố lớp (nếu có nhãn)
def get_class_distribution(label_dir, class_names):
    class_counts = [0] * len(class_names)
    if not os.path.exists(label_dir):
        return class_counts

    try:
        for label_file in os.listdir(label_dir):
            if label_file.endswith('.txt'):
                with open(os.path.join(label_dir, label_file), 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:  # Make sure line is not empty
                            try:
                                class_id = int(parts[0])
                                # Check if class_id is valid
                                if 0 <= class_id < len(class_names):
                                    class_counts[class_id] += 1
                            except (ValueError, IndexError):
                                # Skip invalid lines
                                continue
    except Exception as e:
        st.error(f"Error reading label files: {str(e)}")

    return class_counts


@st.cache_data
def analyze_dataset(train_dir, class_names):
    # Get paths for all sets
    val_dir = train_dir.replace('train', 'val')
    test_dir = train_dir.replace('train', 'test')

    train_label_dir = train_dir.replace('images', 'labels')
    val_label_dir = val_dir.replace('images', 'labels')
    test_label_dir = test_dir.replace('images', 'labels')

    # Check if label directories exist and have files
    if (not os.path.exists(train_label_dir) or not os.listdir(train_label_dir)) and \
       (not os.path.exists(val_label_dir) or not os.listdir(val_label_dir)) and \
       (not os.path.exists(test_label_dir) or not os.listdir(test_label_dir)):
        st.warning("Không tìm thấy nhãn trong các thư mục labels")
        st.info("Vui lòng tải dataset đầy đủ bao gồm cả thư mục labels.")

        # Create a placeholder chart with zeros
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(class_names, [0] * len(class_names), color='purple')
        ax.set_title(
            'Phân bố số lượng đối tượng theo lớp (Toàn bộ tập dữ liệu) - Không có dữ liệu')
        ax.set_xlabel('Lớp')
        ax.set_ylabel('Số lượng đối tượng')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    else:
        # Get class distribution for each set
        train_class_counts = get_class_distribution(
            train_label_dir, class_names)
        val_class_counts = get_class_distribution(val_label_dir, class_names)
        test_class_counts = get_class_distribution(test_label_dir, class_names)

        # Combine counts for total distribution
        total_class_counts = [
            t + v + ts for t, v, ts in zip(train_class_counts, val_class_counts, test_class_counts)]

        # Count images in each set

        st.write("#### Số lượng dữ liệu trong tập dữ liệu khi chia tập dữ liệu")
        st.table({
            "Train": sum(train_class_counts),
            "Val": sum(val_class_counts),
            "Test": sum(test_class_counts)
        })

        # Total of data train - val - test
        # st.write("#### Tổng số lượng đối tượng trong tập dữ liệu")
        st.info(
            f"Tổng số lượng đối tượng trong tập dữ liệu: {sum(total_class_counts)}")

        # Plot total class distribution
        st.write("#### Phân bố số lượng đối tượng theo lớp")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(class_names, total_class_counts, color='purple')
        ax.set_title(
            'Phân bố số lượng đối tượng theo lớp')
        ax.set_xlabel('Lớp')
        ax.set_ylabel('Số lượng đối tượng')
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # # Table for class distribution
        # st.write("#### Phân bố số lượng đối tượng theo lớp")
        # st.table(list(zip(class_names, total_class_counts)))

        # Table for number of data train - val - test (number of data in each set)
        # st.write("#### Số lượng dữ liệu trong tập dữ liệu")
        # st.table(list(zip(class_names, train_class_counts,
        #          val_class_counts, test_class_counts)))

        # # Also show train set distribution separately
        # st.write("#### Phân bố số lượng đối tượng theo lớp (Tập Train)")
        # fig, ax = plt.subplots(figsize=(10, 5))
        # ax.bar(class_names, train_class_counts, color='blue')
        # ax.set_title('Phân bố số lượng đối tượng theo lớp (Tập Train)')
        # ax.set_xlabel('Lớp')
        # ax.set_ylabel('Số lượng đối tượng')
        # plt.xticks(rotation=45)
        # st.pyplot(fig)

    # c. Hiển thị hình ảnh mẫu với bounding boxes (nếu có)
    display_sample_images_with_boxes(
        train_dir, train_label_dir, 'Train', class_names)

    # d. Phân tích phân bố giá trị pixel (histogram)
    # plot_pixel_distribution(train_dir)


@st.cache_data
def display_sample_images_with_boxes(image_dir, label_dir, title, class_names, num_samples=4):
    st.write("#### Một số mẫu với bounding boxes")

    # Check if image directory exists and has images
    if not os.path.exists(image_dir):
        st.warning(f"Không tìm thấy thư mục hình ảnh: {image_dir}")
        st.info("Vui lòng tải dataset đầy đủ bao gồm cả thư mục images.")
        return

    image_files = [f for f in os.listdir(
        image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not image_files:
        st.warning(f"Không tìm thấy hình ảnh trong thư mục {image_dir}")
        st.info("Vui lòng tải dataset đầy đủ bao gồm cả hình ảnh.")

        # Offer to download the dataset
        if st.button("Download Dataset (from display_sample)"):
            success = download_dataset_from_gdrive()
            if success:
                st.rerun()
        return

    # Create a 2x2 grid of sample images
    sample_files = random.sample(
        image_files, min(num_samples, len(image_files)))
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for i, img_file in enumerate(sample_files):
        if i >= num_samples:
            break

        # Đọc hình ảnh
        img_path = os.path.join(image_dir, img_file)
        try:
            img = Image.open(img_path)
            if hasattr(img, '_getexif') and img._getexif() is not None:
                orientation = img._getexif().get(274)
                if orientation == 3:
                    img = img.rotate(180, expand=True)
                elif orientation == 6:
                    img = img.rotate(270, expand=True)
                elif orientation == 8:
                    img = img.rotate(90, expand=True)
            img_array = np.array(img)
        except Exception as e:
            st.error(f"Error opening image {img_path}: {str(e)}")
            continue

        # Đọc nhãn (nếu có)
        label_path = os.path.join(label_dir, img_file.replace(
            '.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt'))
        boxes = []
        labels = []

        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:  # Make sure we have all required values
                            class_id = int(parts[0])
                            center_x, center_y, width, height = map(
                                float, parts[1:5])

                            # Chuyển đổi tọa độ YOLO sang tọa độ pixel
                            h, w = img_array.shape[:2]
                            x = (center_x - width / 2) * w
                            y = (center_y - height / 2) * h
                            box_w = width * w
                            box_h = height * h

                            # Check if class_id is valid
                            if 0 <= class_id < len(class_names):
                                boxes.append([x, y, box_w, box_h])
                                labels.append(class_names[class_id])
            except Exception as e:
                st.error(f"Error reading label file {label_path}: {str(e)}")

        # Vẽ hình ảnh và bounding boxes
        axes[i].imshow(img_array)
        for box, label in zip(boxes, labels):
            rect = patches.Rectangle(
                (box[0], box[1]), box[2], box[3], linewidth=2, edgecolor='r', facecolor='none')
            axes[i].add_patch(rect)
            axes[i].text(box[0], box[1], label, color='red',
                         fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        axes[i].set_title(f'{title} - Mẫu {i+1}')
        axes[i].axis('off')

    plt.tight_layout()
    st.pyplot(fig)


@st.cache_data
def plot_pixel_distribution(image_dir):
    if not os.path.exists(image_dir):
        st.warning(f"Không tìm thấy thư mục hình ảnh: {image_dir}")
        return

    image_files = [f for f in os.listdir(
        image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not image_files:
        st.warning(f"Không tìm thấy hình ảnh trong thư mục {image_dir}")
        return

    try:
        sample_img_path = os.path.join(image_dir, image_files[0])
        sample_img = Image.open(sample_img_path)
        img_array = np.array(sample_img)

        # Check if image is RGB (3 channels)
        if len(img_array.shape) < 3 or img_array.shape[2] < 3:
            st.warning(f"Hình ảnh {sample_img_path} không phải định dạng RGB")
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, color in enumerate(['red', 'green', 'blue']):
            ax.hist(img_array[:, :, i].ravel(), bins=256,
                    color=color, alpha=0.5, label=color.capitalize())
        ax.set_title('Phân bố giá trị pixel (RGB) - Tập Train')
        ax.set_xlabel('Giá trị pixel')
        ax.set_ylabel('Tần suất')
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error analyzing pixel distribution: {str(e)}")
        import traceback
    ax.legend()
