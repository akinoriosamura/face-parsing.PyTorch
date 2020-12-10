#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from model import BiSeNet

import glob
import os
import os.path as osp
import numpy as np
import sys
import torch
import torchvision.transforms as transforms
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'RetinaFace'))
from retina_detector import RetinaDetector


def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
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

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # return vis_im

def evaluate(respth='./res/test_growing', dspth='./data', cp='model_final_diss.pth'):

    label_dir = osp.join(respth, "label")
    img_dir = osp.join(respth, "img")
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    # net.cuda()
    net.cpu()
    save_pth = osp.join('.', cp)
    net.load_state_dict(torch.load(save_pth, map_location=torch.device('cpu')))
    net.eval()

    detector = RetinaDetector()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        print("total len ", len(os.listdir(dspth)))
        files = glob.glob(dspth+"/*")
        for id, image_path in enumerate(files):
            if image_path[-3:] not in ["jpg", "JPG" ,"png", "PNG"]:
                continue

            # img = cv2.imread(osp.join(dspth, image_path), cv2.IMREAD_COLOR)
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            #import pdb; pdb.set_trace()
            if img is None:
                print("img is None")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite("ori_sample.jpg", img)
            # add padding 20% of image
            ori_height, ori_width, _ = img.shape[:3]
            # top, bottom, left, right
            pad_h = int(ori_height * 0.2)
            pad_w = int(ori_width * 0.2)
            padded_img = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE)
            cv2.imwrite("pad_sample.jpg", padded_img)
            # detector bbs
            print("==== start detector ====")
            bbs = detector.detect(padded_img)
            if bbs is not None:
                print("bbs; ", bbs.shape)
                for bb_id, bb in enumerate(bbs):

                    # bb: (x, y, x+w, y+h)
                    print("==== start alignment ====")
                    bb_x = bb[0]
                    bb_y = bb[1]
                    bb_w = bb[2] - bb[0]
                    bb_h = bb[3] - bb[1]
                    # 顔枠の1.4倍でcropするための拡大幅
                    scaled_w = int(0.2 * bb_w)
                    scaled_h = int(0.2 * bb_h)
                    new_bb_x1 = max(0, int(bb_x - (scaled_w / 2)))
                    new_bb_x2 = min(padded_img.shape[1], int(new_bb_x1 + bb_w + scaled_w))
                    new_bb_y1 = max(0, int(bb_y - (scaled_h / 2)))
                    new_bb_y2 = min(padded_img.shape[0], int(new_bb_y1 + bb_h + scaled_h))
                    croped_image = padded_img[new_bb_y1: new_bb_y2, new_bb_x1: new_bb_x2]
                    # croped_image = padded_img[int(bb_y): int(bb_y+bb_h), int(bb_x): int(bb_x+bb_w)]
                    try:
                        cv2.imwrite("croped_sample.jpg", croped_image)
                    except:
                        print(croped_image)
                        import pdb; pdb.set_trace()

                    # image = cv2.resize(croped_image, (512, 512), interpolation=cv2.INTER_LINEAR)
                    image = cv2.resize(croped_image, (512, 512), interpolation=cv2.INTER_LINEAR)
                    tensor_img = to_tensor(image)
                    tensor_img = torch.unsqueeze(tensor_img, 0)
                    # tensor_img = tensor_img.cuda()
                    tensor_img = tensor_img.cpu()
                    import time
                    start = time.time()
                    out = net(tensor_img)[0]
                    elapsed_time = time.time() - start
                    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

                    parsing = out.squeeze(0).cpu().numpy().argmax(0)
                    # print(parsing)
                    print(np.unique(parsing))

                    # save image
                    base_img_p = osp.basename((image_path))
                    save_img_p = osp.join(img_dir, base_img_p)[:-4] + "_" + str(bb_id) + ".jpg"
                    print(save_img_p)
                    cv2.imwrite(save_img_p, image)
                    save_label_p = osp.join(label_dir, base_img_p)[:-4] + "_" + str(bb_id) + ".jpg"
                    print(save_label_p)
                    vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=save_label_p)

            if id % 100 == 0:
                print("num; ", id)



if __name__ == "__main__":
    evaluate(respth='./res/test_hair_seg_512', dspth='./data/200424_hair_seg', cp='79999_iter.pth')


