BATCH_SIZE = 64
NUM_WORKERS = 10
DIRECTORY = '/mnt/p/processed/'
MODEL_PATH = './model.pth'

TOP_CROP_ORIGINAL = 460
BOT_CROP_ORIGINAL = 90
CENTER_CROP_ORIGINAL_X = 200
CENTER_CROP_ORIGINAL_Y = 200
FX = 0.1
FY = 0.1
top_crop = int(TOP_CROP_ORIGINAL * FY)
bot_crop = int(BOT_CROP_ORIGINAL * FY)
center_crop_x = int(CENTER_CROP_ORIGINAL_X * FX)
center_crop_y = int(CENTER_CROP_ORIGINAL_Y * FY)

directories = [
    'both_closed',
    'both_open',
    'honda_closed',
    'honda_open',
    'none_closed',
    'none_open',
    'truck_closed',
    'truck_open'
]
classes = directories
