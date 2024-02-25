# -*- coding: utf-8 -*-

import sys
import os
import cv2
import requests
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import glob
import tqdm

import matplotlib.pyplot as plt
from PIL import Image

sys.path.append('.')
import model.sefm_model import SeFM



def get_args_parser():
    parser = argparse.ArgumentParser('Open-World In-context Inference Demo', add_help=False)
    parser.add_argument('--ckpt_dir', type=str, help='dir to ckpt',
                        default='checkpoints/latest.pth')
    return parser.parse_args()


def prepare_model(chkpt_dir):
    # build model
    model = SeFM()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cuda:0')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    model.eval()
    return model



if __name__ == '__main__':
    args = get_args_parser()

    ckpt_dir = args.ckpt_dir
    model = args.model
    epoch = args.epoch

    ckpt_file = 'checkpoint-{}.pth'.format(epoch)
    assert ckpt_dir[-1] != "/"

    ckpt_path = ckpt_dir
    SeFM = prepare_model(ckpt_path, model)
    print('Model loaded.')

    device = torch.device("cuda")
    SeFM.to(device)

    img2_path = "test/1.png"
    tgt2_path = "test/2.png"
    img_path = "test/3.png"
    img_name = os.path.basename(img_path)
    out_path = "test/results.png"

    res = 256

    img2 = cv2.imread(img2_path)
    img2 = cv2.resize(img2, (res, res))

    tgt2 = cv2.imread(tgt2_path)
    tgt2 = cv2.resize(tgt2, (res, res))

    img = cv2.imread(img_path)
    img = cv2.resize(img, (res, res))


    tgt = img # we use query to init decoupled image for easy reconstruction

    input = np.zeros((2 * res, 2 * res, 3))
    input[:res, :res] = img2
    input[:res, res:] = tgt2
    input[res:, :res] = img
    input[:res, :res] = tgt

    torch.manual_seed(2)

    input = np.transpose(input, (2, 0, 1)).astype(np.float32) / 255.0
    input = torch.tensor(input)
    input = image.unsqueeze(0).cuda()

    decoupled = SeFM(input)

    decoupled = decoupled.cpu().data[0] * 255.
    decoupled = np.clip(decoupled, 0, 255)
    decoupled = np.ascontiguousarray(np.transpose(decoupled, (1, 2, 0)))
    cv2.imwrite(out_path, decoupled)

