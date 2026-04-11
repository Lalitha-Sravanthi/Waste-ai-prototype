from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"

DEFAULT_SOURCE_DATASET = Path(r"D:\Downloads\final_large_dataset.xls")
RAW_DATASET_PATH = RAW_DIR / "final_large_dataset.csv"
PROCESSED_DATASET_PATH = PROCESSED_DIR / "enriched_dataset.csv"
MODEL_PATH = MODELS_DIR / "peak_model.pkl"

PICKUP_THRESHOLD = 80.0
URGENT_THRESHOLD = 90.0
APPROVAL_COST_THRESHOLD = 120.0

