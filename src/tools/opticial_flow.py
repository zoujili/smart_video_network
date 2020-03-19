import numpy as np
import cv2

from skimage.transform import resize

def image_seq_to_optical_flow(image_list):
    result = []
    for i in range(len(image_list) - 1):
        mask = np.zeros_like(image_list[i])
        gray = cv2.cvtColor(image_list[i], cv2.COLOR_BGR2GRAY)
        next = cv2.cvtColor(image_list[i + 1], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gray, next, None, pyr_scale=0.5, levels=5, winsize=11, iterations=5,
                                            poly_n=5, poly_sigma=1.1, flags=0)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mask[..., 0] = angle * 180 / np.pi / 2
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        mask[..., 1] = 255
        # result.append(mask)
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
        dense_flow = cv2.addWeighted(image_list[i + 1], 1, rgb, 2, 0)
        # cv2.imshow(str(i),dense_flow)
        result.append(dense_flow)
    return result
#
# path = '00006_others.npy'
# data = np.load(path)
# a = image_seq_to_optical_flow(data)
# cv2.waitKey(0)

#
# path = '/home/ubuntu/jili/ai_project/video_to_seq_rnn/data/np_image_seq//2019-12-20-13-55-17_jili/00145_chests.npy'
# data = np.load(path)
#
# i = 0
# for item in data:
#     i = i + 1
#     cv2.imshow(str(i), item)
#
#
# cv2.waitKey(0)