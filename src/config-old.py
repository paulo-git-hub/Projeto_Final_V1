import os

# ==========================================
# CAMINHOS E PARÂMETROS
# ==========================================
BASE_DIR = "/content/drive/MyDrive/Colab Notebooks/Projeto Final/Projeto_Final_V1"

# Dados
DATA_RAW = os.path.join(BASE_DIR, "data/raw/kc_house_data.csv")
DATA_PROCESSED = os.path.join(BASE_DIR, "data/processed/kc_house_data_clean.csv")
DATA_FINAL = os.path.join(BASE_DIR, "data/final/kc_house_data_final.csv")

# Modelos e Outputs
MODEL_DIR = os.path.join(BASE_DIR, "models/v1")
OUTPUT_FIGURES_DIR = os.path.join(BASE_DIR, "outputs/figures")

# Parâmetros Globais
RANDOM_STATE = 42
TEST_SIZE = 0.20

# ==========================================
# GARANTIA DE ESTRUTURA (RODA A CADA IMPORT)
# ==========================================
# Extraímos apenas as pastas dos caminhos dos arquivos
DIRETORIOS_NECESSARIOS = [
    os.path.dirname(DATA_RAW),       # Garante data/raw/
    os.path.dirname(DATA_PROCESSED), # Garante data/processed/
    os.path.dirname(DATA_FINAL),     # Garante data/final/
    MODEL_DIR,                       # Garante models/v1/
    OUTPUT_FIGURES_DIR,              # Garante outputs/figures/
    os.path.join(BASE_DIR, "notebooks") # Garante notebooks/
]

# Cria os diretórios caso o Colab ou o Drive não os encontrem
for pasta in DIRETORIOS_NECESSARIOS:
    os.makedirs(pasta, exist_ok=True)