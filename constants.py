from pathlib import Path
from platform import platform

BATCH_SIZE = 64
NUM_WORKERS = 16
DIRECTORY = 'P:/processed/'
MODEL_PATH = './model.pth'
MAX_IMAGES = 75
TOP_CROP_ORIGINAL = 460
BOT_CROP_ORIGINAL = 90
CENTER_CROP_ORIGINAL_X = 200
CENTER_CROP_ORIGINAL_Y = 200
FX = 0.2
FY = 0.2
top_crop = int(TOP_CROP_ORIGINAL * FY)
bot_crop = int(BOT_CROP_ORIGINAL * FY)
center_crop_x = int(CENTER_CROP_ORIGINAL_X * FX)
center_crop_y = int(CENTER_CROP_ORIGINAL_Y * FY)

directories = [
    'open',
    'closed'
]
classes = directories

if 'Windows' in platform():
    sorted_base_dir = Path('P:/sorted/')
    processed_base_dir = Path('P:/processed/')
    sorted_base_dir.mkdir(exist_ok=True)
    sort_directories = [sorted_base_dir / directory for directory in directories]
    train_directories = [processed_base_dir / "train" / f"{directory}" for directory in directories]
    test_directories = [processed_base_dir / "test" / f"{directory}"for directory in directories]
    sorted_base_dir = Path('P:/sorted/')
    sorted_base_dir.mkdir(exist_ok=True)
