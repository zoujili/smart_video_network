import cv2
import glob
import json
import os
import os.path
import os.path
import os.path
import os.path
import sys
from src.network.cnn_extractor import Extractor
from subprocess import call

import numpy as np
from tqdm import tqdm

import src.utils as util


# Load Data #################################################################
def load_data_list(src, root_image_folder, root_seq_folder):
    result = []
    with open(src) as file:
        videos = json.load(file)
    video_data = videos['data']
    for item in video_data:
        video_path = item['path']
        if len(item['categories']) == 0:
            print("categories empty")
            continue
        split_paths = video_path.split('/')
        if len(split_paths) < 2:
            print("invalid video path")
            continue
        video_name = split_paths[len(split_paths) - 1]
        video_name_prefix = os.path.splitext(video_name)[0]
        batch_name = split_paths[len(split_paths) - 2]

        seq_name = video_name_prefix + '.npy'
        image_folder = os.path.join(root_image_folder, batch_name, video_name_prefix)
        seq_path = os.path.join(root_seq_folder, batch_name, seq_name)
        seq_folder = os.path.join(root_seq_folder, batch_name)

        data_item = {'path': video_path,
                     'image_folder': image_folder,
                     'seq_path': seq_path,
                     'seq_folder': seq_folder,
                     'categories': item['categories'],
                     }
        result.append(data_item)
    return result


############################################################################


# Extract Image ############################################################
def extract_images(data_list):
    pbar = tqdm(total=len(data_list))
    for item in data_list:
        video_path = item['path']
        image_folder = item['image_folder']
        if len(item['categories']) == 0:
            print("categories empty")
            pbar.update(1)
            continue
        if not check_already_extracted(image_folder):
            extract_images_from_video(video_path, image_folder)
        pbar.update(1)
    pbar.close()


def extract_images_from_video(video_path, image_folder):
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    call(["ffmpeg", "-i", video_path, os.path.join(image_folder, '%04d.png')])


def check_already_extracted(image_folder):
    """Check to see if we created the -0001 frame of this file."""
    return bool(os.path.exists(os.path.join(image_folder, '0001.png')))


############################################################################

# Build Sequence ###########################################################
def build_sequence(data_list, seq_length, seq_type='image', feature_extractor=None):
    pbar = tqdm(total=len(data_list))
    for item in data_list:
        path = item['seq_path']
        if os.path.isfile(path):
            pbar.update(1)
            continue
        frames = list_images_for_folder(item['image_folder'])
        frames = sample_list(frames, seq_length)
        sequence = []
        for image in frames:
            x_np = load_image(image)
            if seq_type == 'image':
                sequence.append(x_np)
            elif seq_type == 'feature':
                x_np = np.expand_dims(x_np, axis=0)
                features = feature_extractor.extract(x_np)
                sequence.append(features)
            else:
                print("Unknown sequence type.")
                sys.exit()

        seq_folder = item['seq_folder']
        if not os.path.exists(seq_folder):
            os.makedirs(seq_folder)
        # Save the sequence.
        np.save(path, sequence)
        pbar.update(1)
    pbar.close()


############################################################################

# Gen Train Map ############################################################
def gen_seq_file(data_list, dst):
    result = []
    for item in data_list:
        data_item = {'seq_path': item['seq_path'],
                     'categories': item['categories']}
        result.append(data_item)

    with open(dst, 'w') as f:
        json.dump(result, f)


def split_train_map(src, dst, total_categories, train_rate, valid_rate, test_rate, ):
    with open(src) as file:
        raw = json.load(file)

    categories_data = {}
    for item in raw:
        categories = item['categories']
        for c in categories:
            if c in categories_data:
                categories_data[c].append(item)
            else:
                categories_data[c] = [item]

    # cut data to train,valid,test
    rate = np.array([train_rate, valid_rate, test_rate])
    split_data = [[], [], []]
    sum_rate = np.cumsum(rate)

    for k, items in categories_data.items():
        for item in items:
            index = int(np.searchsorted(sum_rate, np.random.rand(1) * 1.0))
            split_data[index].append(item)

    train_map = {"train": split_data[0], "valid": split_data[1], "test": split_data[2]}

    print("Train map count:")
    count_by_categories(train_map["train"], total_categories)
    print("Valid map count:")
    count_by_categories(train_map["valid"], total_categories)
    print("Test map count:")
    count_by_categories(train_map["test"], total_categories)

    with open(dst, 'w') as f:
        json.dump(train_map, f)


##########################################################################


# Helper #################################################################
def load_image(path):
    x_np = cv2.imread(path)
    x_np = cv2.resize(x_np, dsize=(224, 224), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    return x_np


def list_images_for_folder(image_folder):
    """Given a sample row from the data file, get all the corresponding frame file names."""
    images = sorted(glob.glob(os.path.join(image_folder, '*.png')))
    return images


def sample_list(origin_list, size):
    """Given a list and a size, return a rescaled/samples list. For example,
    if we want a list of size 5 and we have a list of size 25, return a new
    list of size five which is every 5th element of the origin  list."""
    assert len(origin_list) >= size

    # Get the number to skip between iterations.
    skip = len(origin_list) // size

    # Build our new output.
    output = [origin_list[i] for i in range(0, len(origin_list), skip)]

    # Cut off the last one if needed.
    return output[:size]


def count_by_categories(data, video_categories):
    freq = {}
    for item in video_categories:
        freq[item] = 1
    data_key_list = []
    for item in data:
        data_key_list.append(item['seq_path'])
        cate_key = item['categories'][0]
        if cate_key in video_categories:
            freq[cate_key] += 1
    print(freq)
    data_key_set = list(set(data_key_list))
    print("set count: %s" % len(data_key_set))


############################################################################


def do():
    if util.SEQ_DATA_TYPE == 'image':
        cnn_extractor = None
    elif util.SEQ_DATA_TYPE == 'feature':
        cnn_extractor = Extractor('./output/cnn.hdf5', 8)

    meta = load_data_list(util.SCRIPT_MERGE_VIDEO_PATH, util.DATA_EXTRACT_IMAGE_FOLDER, util.DATA_EXTRACT_SEQ_FOLDER)
    extract_images(meta)
    build_sequence(meta, util.SEQ_LEN, util.SEQ_DATA_TYPE, cnn_extractor)
    gen_seq_file(meta, util.SCRIPT_EXTRACT_SEQ_RAW_PATH)
    split_train_map(util.SCRIPT_EXTRACT_SEQ_RAW_PATH,
                    util.SCRIPT_EXTRACT_SEQ_SPLIT_PATH,
                    util.VIDEO_CATEGORIES,
                    util.TRAIN_RATE,
                    util.VALID_RATE,
                    util.TEST_RATE, )


if __name__ == '__main__':
    os.chdir('./..')
    do()
    print("success")
