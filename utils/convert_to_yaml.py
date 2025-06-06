import os
import xml.etree.ElementTree as ET
import glob
import shutil
from sklearn.model_selection import train_test_split
import yaml
from tqdm import tqdm

def create_folder(folder_path):
    """Tạo thư mục nếu chưa tồn tại"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def get_classes(annotation_dir):
    """Lấy tất cả các lớp từ file XML"""
    class_names = set()
    
    # Lấy tất cả các file xml trong các thư mục con
    xml_files = []
    for root, dirs, files in os.walk(annotation_dir):
        for file in files:
            if file.endswith('.xml'):
                xml_files.append(os.path.join(root, file))
    
    for xml_file in tqdm(xml_files, desc="Scanning classes"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            class_names.add(class_name)
    
    return sorted(list(class_names))

def convert_coordinates(size, box):
    """Chuyển đổi tọa độ từ VOC (xmin, ymin, xmax, ymax) sang YOLO (x_center, y_center, width, height)"""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    
    x_min = float(box[0])
    y_min = float(box[1])
    x_max = float(box[2])
    y_max = float(box[3])
    
    # Tính toán tọa độ trung tâm và kích thước
    x_center = ((x_min + x_max) / 2.0) * dw
    y_center = ((y_min + y_max) / 2.0) * dh
    w = (x_max - x_min) * dw
    h = (y_max - y_min) * dh
    
    # Đảm bảo các giá trị nằm trong khoảng [0, 1]
    x_center = min(max(x_center, 0), 1)
    y_center = min(max(y_center, 0), 1)
    w = min(max(w, 0), 1)
    h = min(max(h, 0), 1)
    
    return x_center, y_center, w, h

def convert_xml_to_yolo(xml_file, output_dir, class_names):
    """Chuyển đổi một file XML sang định dạng YOLO"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Lấy kích thước hình ảnh
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        # Lấy tên file từ XML
        filename = root.find('filename').text
        base_filename = os.path.splitext(filename)[0]
        
        # Tạo file txt mới cho YOLO
        with open(os.path.join(output_dir, f"{base_filename}.txt"), 'w') as f:
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                
                # Lấy chỉ số lớp
                if class_name in class_names:
                    class_idx = class_names.index(class_name)
                else:
                    print(f"Warning: Class {class_name} not found in the predefined classes")
                    continue
                
                # Lấy tọa độ bounding box
                bbox = obj.find('bndbox')
                xmin = bbox.find('xmin').text
                ymin = bbox.find('ymin').text
                xmax = bbox.find('xmax').text
                ymax = bbox.find('ymax').text
                
                # Chuyển đổi tọa độ sang định dạng YOLO
                x_center, y_center, w, h = convert_coordinates((width, height), (xmin, ymin, xmax, ymax))
                
                # Ghi vào file
                f.write(f"{class_idx} {x_center} {y_center} {w} {h}\n")
        
        return True
    except Exception as e:
        print(f"Error processing {xml_file}: {e}")
        return False

def copy_images(image_dir, annotation_dir, output_dir):
    """Copy các file ảnh tương ứng với file annotation vào thư mục đích"""
    # Lấy danh sách file XML
    xml_files = []
    for root, dirs, files in os.walk(annotation_dir):
        for file in files:
            if file.endswith('.xml'):
                xml_files.append(os.path.join(root, file))
    
    # Tạo mapping từ tên file XML sang tên file ảnh
    image_map = {}
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.find('filename').text
        base_filename = os.path.splitext(filename)[0]
        image_map[base_filename] = filename
    
    # Tìm và copy file ảnh
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    copied_images = 0
    
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            file_base = os.path.splitext(file)[0]
            file_ext = os.path.splitext(file)[1].lower()
            
            if file_ext in image_extensions:
                # Kiểm tra xem file có trong danh sách annotation không
                if file_base in image_map or file in image_map.values():
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(output_dir, file)
                    shutil.copy2(src_path, dst_path)
                    copied_images += 1
    
    print(f"Copied {copied_images} images to {output_dir}")

def create_yaml(output_dir, class_names, dataset_name="custom_dataset"):
    """Tạo file YAML cấu hình cho YOLOv8"""
    yaml_content = {
        'path': '../', # Đường dẫn tương đối đến thư mục gốc của dataset
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    with open(os.path.join(output_dir, f"{dataset_name}.yaml"), 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

def main():
    # Cấu hình đường dẫn
    base_dir = "." # Thư mục hiện tại
    data_dir = os.path.join(base_dir, "data")
    annotation_dir = os.path.join(data_dir, "annotation")
    image_dir = os.path.join(data_dir, "image")
    
    # Thư mục đầu ra
    output_base = os.path.join(base_dir, "yolov8_dataset")
    create_folder(output_base)
    
    # Lấy danh sách các lớp
    class_names = get_classes(annotation_dir)
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # Tạo cấu trúc thư mục cho YOLOv8
    splits = ['train', 'val', 'test']
    for split in splits:
        for subdir in ['images', 'labels']:
            create_folder(os.path.join(output_base, split, subdir))
    
    # Chuyển đổi file XML sang YOLO và thu thập danh sách các file
    all_files = []
    for root, dirs, files in os.walk(annotation_dir):
        for file in files:
            if file.endswith('.xml'):
                xml_path = os.path.join(root, file)
                base_filename = os.path.splitext(file)[0]
                all_files.append(base_filename)
                
                # Tạo thư mục labels tạm thời
                temp_labels_dir = os.path.join(output_base, "labels_temp")
                create_folder(temp_labels_dir)
                
                # Chuyển đổi file
                convert_xml_to_yolo(xml_path, temp_labels_dir, class_names)
    
    # Phân chia dataset
    train_files, test_val_files = train_test_split(all_files, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(test_val_files, test_size=0.5, random_state=42)
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
    # Copy các file vào thư mục đích
    for base_filename in train_files:
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            src_img = glob.glob(os.path.join(image_dir, "**", f"{base_filename}{ext}"), recursive=True)
            if src_img:
                shutil.copy2(src_img[0], os.path.join(output_base, "train", "images", os.path.basename(src_img[0])))
                break
        src_label = os.path.join(output_base, "labels_temp", f"{base_filename}.txt")
        if os.path.exists(src_label):
            shutil.copy2(src_label, os.path.join(output_base, "train", "labels", f"{base_filename}.txt"))
    
    for base_filename in val_files:
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            src_img = glob.glob(os.path.join(image_dir, "**", f"{base_filename}{ext}"), recursive=True)
            if src_img:
                shutil.copy2(src_img[0], os.path.join(output_base, "val", "images", os.path.basename(src_img[0])))
                break
        src_label = os.path.join(output_base, "labels_temp", f"{base_filename}.txt")
        if os.path.exists(src_label):
            shutil.copy2(src_label, os.path.join(output_base, "val", "labels", f"{base_filename}.txt"))
    
    for base_filename in test_files:
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            src_img = glob.glob(os.path.join(image_dir, "**", f"{base_filename}{ext}"), recursive=True)
            if src_img:
                shutil.copy2(src_img[0], os.path.join(output_base, "test", "images", os.path.basename(src_img[0])))
                break
        src_label = os.path.join(output_base, "labels_temp", f"{base_filename}.txt")
        if os.path.exists(src_label):
            shutil.copy2(src_label, os.path.join(output_base, "test", "labels", f"{base_filename}.txt"))
    
    # Xóa thư mục labels tạm thời
    shutil.rmtree(os.path.join(output_base, "labels_temp"))
    
    # Tạo file YAML
    create_yaml(output_base, class_names)
    
    print(f"\nDone! Dataset ready for YOLOv8 training at {output_base}")
    
    # In lệnh đề xuất để huấn luyện
    print("\nSuggested command to train YOLOv8:")
    print(f"yolo task=detect mode=train model=yolov8n.pt data={output_base}/custom_dataset.yaml epochs=100 imgsz=640")

if __name__ == "__main__":
    main()