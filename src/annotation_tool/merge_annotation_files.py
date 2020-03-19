import glob
import json
import os
import src.utils as util
import os

def merge_annotated_video(src, dst):
    files = glob.glob(os.path.join(src, '*.json'))
    data_list = []
    count = 1
    for file in files:
        with open(file) as f:
            file_data = json.load(f)
            data = file_data["data"]
            for item in data:
                if len(item['categories']) > 0:
                    data_list.append(item)
                    count = count + 1

    print(count)
    json_data = {"data": data_list}
    with open(dst, 'w') as f:
        json.dump(json_data, f)


if __name__ == '__main__':
    os.chdir('./../../')
    merge_annotated_video(util.ANNOTATION_VIDEO_SCRIPT, util.SCRIPT_MERGE_VIDEO_PATH)
