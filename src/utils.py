import numpy as np
import os


BATCH_SIZE = 6
VIDEO_CATEGORIES = ["chop", "others", "gun", "chests", "shooting"]

SEQ_DATA_TYPE = 'image'  # image or feature
SEQ_LEN = 7
CNN_FEATURE_LEN = 1024


NB_EPOCH = 1000000
TRAIN_RATE = 0.8
VALID_RATE = 0.1
TEST_RATE = 0.1
#####################Preprocess#############################################

# 500ms
ANNOTATION_VIDEO_DURATION = '00:00:00.500'
ANNOTATION_VIDEO_ROOT = './dataset/annotated_video'
ANNOTATION_VIDEO_SCRIPT = './dataset/annotated_video_script'

SCRIPT_MERGE_VIDEO_PATH = './dataset/script/merged_videos.json'

DATA_EXTRACT_IMAGE_FOLDER = './dataset/extracted_image/'
DATA_EXTRACT_SEQ_FOLDER = './dataset/extracted_sequence/'
SCRIPT_FOLDER = './dataset/script'

SCRIPT_EXTRACT_SEQ_RAW_PATH = './dataset/script/extracted_seq_raw.json'
SCRIPT_EXTRACT_SEQ_SPLIT_PATH = './dataset/script/extracted_seq_split.json'

OUTPUT_CHECKPOINT_FOLDER = 'output/checkpoint/'
OUTPUT_LOG = 'output/logs'


############################################################################


def categories_to_np(categories):
    categories_map = {}
    y = [0 for i in VIDEO_CATEGORIES]
    for i, c in enumerate(VIDEO_CATEGORIES):
        categories_map[c] = i
    for item in categories:
        # if item == "chests":
        #     item = "others"
        # if item == "shooting":
        #     item = "others"
        # if item == "gun":
        #     item = "others"
        idx = categories_map[item]
        y[idx] = 1
    return np.array(y)


def video_path_to_image_folder(video_path):
    split_paths = video_path.split('/')
    if len(split_paths) < 2:
        raise ValueError("invalid video path")
    video_name = split_paths[len(split_paths) - 1]
    split_video_names = os.path.splitext(video_name)
    if len(split_video_names) < 2:
        raise ValueError("invalid video path")
    video_name_prefix = split_video_names[0]
    batch_name = split_paths[len(split_paths) - 2]
    return os.path.join(DATA_EXTRACT_SEQ_FOLDER, batch_name, video_name_prefix)


def video_path_to_seq(video_path):
    split_paths = video_path.split('/')
    if len(split_paths) < 2:
        raise ValueError("invalid video path")
    video_name = split_paths[len(split_paths) - 1]
    split_video_names = os.path.splitext(video_name)
    if len(split_video_names) < 2:
        raise ValueError("invalid video path")
    seq_name = split_video_names[0] + '.npy'
    batch_name = split_paths[len(split_paths) - 2]
    return os.path.join(DATA_EXTRACT_SEQ_FOLDER, batch_name, seq_name)


def list_to_string(str_list):
    result = ""
    for s in str_list:
        result += "_" + s
    return result


def create_folders():
    if not os.path.exists(ANNOTATION_VIDEO_ROOT):
        os.makedirs(ANNOTATION_VIDEO_ROOT)
    if not os.path.exists(ANNOTATION_VIDEO_SCRIPT):
        os.makedirs(ANNOTATION_VIDEO_SCRIPT)
    if not os.path.exists(OUTPUT_LOG):
        os.makedirs(OUTPUT_LOG)
    if not os.path.exists(SCRIPT_FOLDER):
        os.makedirs(SCRIPT_FOLDER)
    if not os.path.exists(DATA_EXTRACT_IMAGE_FOLDER):
        os.makedirs(DATA_EXTRACT_IMAGE_FOLDER)
    if not os.path.exists(DATA_EXTRACT_SEQ_FOLDER):
        os.makedirs(DATA_EXTRACT_SEQ_FOLDER)
    if not os.path.exists(OUTPUT_CHECKPOINT_FOLDER):
        os.makedirs(OUTPUT_CHECKPOINT_FOLDER)
    if not os.path.exists(OUTPUT_CHECKPOINT_FOLDER):
        os.makedirs(OUTPUT_CHECKPOINT_FOLDER)
    if not os.path.exists(OUTPUT_LOG):
        os.makedirs(OUTPUT_LOG)

if __name__ == '__main__':
    os.chdir('./..')
    create_folders()
