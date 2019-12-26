from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from logger import setup_logger
from model import BiSeNet
from mtcnn.detect_face import MTCNN

import cv2
import numpy as np
import os
import time

import torch
import os.path as osp
from PIL import Image
import torchvision.transforms as transforms


def main():
    image_size = 512

    cap = cv2.VideoCapture(0)
    while True:
        ret, image = cap.read()
        height, width, _ = image.shape
        if not ret:
            break

        input = cv2.resize(image, (image_size, image_size))
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        # input = input.astype(np.float32) / 256.0
        # input = np.expand_dims(input, 0)

        # segmentation
        cp = '79999_iter.pth'

        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cpu()
        save_pth = osp.join('./', cp)
        net.load_state_dict(torch.load(save_pth, map_location=torch.device('cpu')))
        net.eval()

        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        with torch.no_grad():
            img = to_tensor(input)
            img = torch.unsqueeze(img, 0)
            img = img.cpu()
            st = time.time()
            out = net(img)[0]
            print("elapse: ", time.time() - st)
            parsing_anno = out.squeeze(0).cpu().numpy().argmax(0)
            stride = 1

            # Colors for all 20 parts
            part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                            [255, 0, 85], [255, 0, 170],
                            [0, 255, 0], [85, 255, 0], [170, 255, 0],
                            [0, 255, 85], [0, 255, 170],
                            [0, 0, 255], [85, 0, 255], [170, 0, 255],
                            [0, 85, 255], [0, 170, 255],
                            [255, 255, 0], [255, 255, 85], [255, 255, 170],
                            [255, 0, 255], [255, 85, 255], [255, 170, 255],
                            [0, 255, 255], [85, 255, 255], [170, 255, 255]]

            im = np.array(image)
            vis_im = im.copy().astype(np.uint8)
            vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
            vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
            vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

            num_of_class = np.max(vis_parsing_anno)

            for pi in range(1, num_of_class + 1):
                index = np.where(vis_parsing_anno == pi)
                vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

            vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
            input = cv2.addWeighted(cv2.cvtColor(input, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

        cv2.imshow('0', input)
        if cv2.waitKey(10) == 27:
            break


if __name__ == '__main__':
    main()
