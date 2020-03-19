import datetime
import json
import os
import random
import src.utils as util
from subprocess import call

TIME_FORMAT = '%H:%M:%S.%f'
FILE_FORMAT = '%05d.mp4'


def str_to_time(str):
    return datetime.datetime.strptime(str, '%H:%M:%S.%f')


def exec_cmd(cmd):
    r = os.popen(cmd)
    t = r.read().strip()
    r.close()
    return t


def get_video_len(path):
    return exec_cmd("ffmpeg -i {} 2>&1 | grep 'Duration' | cut -d ' ' -f 4 | sed s/,//".format(path))


def split_video(path, dst, start_time, duration):
    call(["ffmpeg", "-i", path, "-ss", start_time.strftime(TIME_FORMAT), "-t", duration, "-strict", "-2", dst])


def gen_random_milliseconds():
    return datetime.timedelta(milliseconds=random.randint(5, 10) * 100)


def gen_sub_video_path(path, count):
    batch_name = gen_video_batch_name(path)
    return os.path.join(util.ANNOTATION_VIDEO_ROOT, batch_name, FILE_FORMAT) % count


def gen_video_batch_name(path):
    split_paths = path.split('/')
    return split_paths[len(split_paths) - 2]


def gen_annotation_file_name(path):
    return gen_video_batch_name(path) + ".json"


def gen_videos(path, duration):
    start_str = '00:00:00.000'
    start_time = str_to_time(start_str)
    video_time = str_to_time(get_video_len(path))

    count = 1
    folders = os.path.split(path)
    folder = folders[0]

    data = []
    while start_time < video_time:
        dst = os.path.join(folder, FILE_FORMAT) % count
        split_video(path, dst, start_time, duration)
        start_time = start_time + gen_random_milliseconds()
        count = count + 1
        item = {"path": gen_sub_video_path(path, count), "categories": []}
        data.append(item)

        print(start_time)
        print(count)

    json_data = {"data": data}
    with open(os.path.join(folder, gen_annotation_file_name(path)), 'w') as f:
        json.dump(json_data, f)


if __name__ == '__main__':
    video_path = '/home/ubuntu/fornate_video/batch2/test.mp4'
    duration = util.ANNOTATION_VIDEO_DURATION
    gen_videos(video_path, duration)




