import streamlit as st
import os

from src.data_t import *
from src.theory_t import *
from src.demo_t import *
from src.mlflow_t import *

st.set_page_config(
    page_title="YOLOv8 Vehicle Detection",
    layout="centered"
)


def check_dataset_availability():
    """Check if dataset is available and provide download options if not"""
    yaml_path = 'yolov8_dataset/custom_dataset.yaml'
    with st.container(border=True):
        if not os.path.exists(yaml_path):
            st.warning("⚠️ Dataset not detected")
            st.info("""
            ### Dataset Options:
            
            1. ✅ Download automatically from **Google Drive**
            2. 📥 Or download manually from [Kaggle](https://www.kaggle.com/datasets/dataclusterlabs/indian-vehicle-dataset)
            """)

            # Nút tải từ Google Drive (ZIP)
            if st.button("Download from Google Drive (.zip)"):
                from src.data_t import download_dataset_from_gdrive
                success = download_dataset_from_gdrive()
                if success:
                    st.experimental_rerun()  # làm mới lại layout sau khi tải xong
        else:
            st.success("✅ Dataset detected")
            st.markdown(
                "📁 Dataset hosted on [Google Drive](https://drive.google.com/file/d/1KNgMtAIjTXRCx-Q-xLb_dida3YDnRG5K/view?usp=sharing)")


def main():
    st.title("YOLOv8 Vehicle Detection")

    data_tab, theory_tab, demo_tab, mlflow_tab = st.tabs(
        ["Data", "Theory", "Demo", "MLflow"]
    )

    with data_tab:
        # Check dataset availability
        check_dataset_availability()
        data_description()

    with theory_tab:
        yolov8_theory()

    # with train_tab:
    #     train_model()

    with demo_tab:
        demo_detection()

    with mlflow_tab:
        display_mlflow_runs()


if __name__ == "__main__":
    main()
