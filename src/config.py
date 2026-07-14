import os
BASE_DIR = "/content/drive/MyDrive/Colab Notebooks/Projeto Final/Projeto_Final_V1"
DATA_RAW = os.path.join(BASE_DIR, "data/raw/kc_house_data.csv")
DATA_PROCESSED = os.path.join(BASE_DIR, "data/processed/kc_house_data_clean.csv")
DATA_FINAL = os.path.join(BASE_DIR, "data/final/kc_house_data_final.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models/v1")
OUTPUT_FIGURES_DIR = os.path.join(BASE_DIR, "outputs/figures")
RANDOM_STATE = 42
TEST_SIZE = 0.20
