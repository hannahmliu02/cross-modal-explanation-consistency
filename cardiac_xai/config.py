import argparse
from pathlib import Path

BASE_DIR = Path(__file__).parent

DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_PACK = BASE_DIR / "data" / "pack" / "processed_data"  # pre-packaged npz data
CHECKPOINTS_DIR = BASE_DIR / "models" / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

LABEL_MAP = {
    0: "background",
    1: "left_ventricle",
    2: "right_ventricle",
    3: "left_atrium",
    4: "right_atrium",
    5: "myocardium",
    6: "aorta",
    7: "pulmonary_artery",
}

NUM_CLASSES = 8
IMAGE_SIZE = [256, 256]
TARGET_SPACING = [1.5, 1.5, 1.5]
VOLUME_SIZE = [256, 256, 64]

BATCH_SIZE = 16
LR = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 100
T_MAX = 100
PATIENCE = 15

IG_STEPS = 50
SMOOTHGRAD_SAMPLES = 50
SMOOTHGRAD_STD = 0.1
ATTRIBUTION_THRESHOLD = 0.5
MIN_LABEL_AREA = 100


def get_parser(description=""):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-raw", type=Path, default=DATA_RAW)
    parser.add_argument("--data-processed", type=Path, default=DATA_PROCESSED)
    parser.add_argument("--data-pack", type=Path, default=DATA_PACK)
    parser.add_argument("--checkpoints-dir", type=Path, default=CHECKPOINTS_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--device", type=str, default="mps")
    return parser
