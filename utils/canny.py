import os
import numpy as np
import os.path as osp
import cv2
import torch
from PIL import Image
from skimage.feature import canny

from utils.RCFmodels import RCF
from torchvision import transforms
from skimage.color import rgb2gray

def image_transforms(load_size):

    return transforms.Compose([
        transforms.Resize(size=load_size, interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def tensor_to_image():

    return transforms.ToPILImage()


def image_to_tensor():

    return transforms.ToTensor()


def tensor_to_gray_tensor(image):
    gray_image = rgb2gray(np.array(tensor_to_image()(image)))
    gray_image = image_to_tensor()(Image.fromarray(gray_image))
    return gray_image

# def image_to_edge(image, sigma):
#
#     gray_image = rgb2gray(np.array(tensor_to_image()(image)))
#     edge = image_to_tensor()(Image.fromarray(canny(gray_image, sigma=sigma)))
#     gray_image = image_to_tensor()(Image.fromarray(gray_image))
#     print(edge.shape)
#     print(gray_image.shape)
#
#     return edge, gray_image

def image_to_edge(model, image, load_size, save_dir="./temp"):
    # model = RCF().cuda()
    model.eval()
    # checkpoint = torch.load("./utils/bsds500_pascal_model.pth")
    # model.load_state_dict(checkpoint)

    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    scale = [0.5, 1, 1.5]

    in_ = image.numpy().transpose((1, 2, 0))
    _, H, W = image.shape
    ms_fuse = np.zeros((H, W), np.float32)
    for k in range(len(scale)):
        im_ = cv2.resize(in_, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
        im_ = im_.transpose((2, 0, 1))
        results = model(torch.unsqueeze(torch.from_numpy(im_).cuda(), 0))
        fuse_res = torch.squeeze(results[-1].detach()).cpu().numpy()
        fuse_res = cv2.resize(fuse_res, (W, H), interpolation=cv2.INTER_LINEAR)
        ms_fuse += fuse_res
    ms_fuse = ms_fuse / len(scale)

    ms_fuse = ((1 - ms_fuse) * 255).astype(np.uint8)
    # temp_root = osp.join(save_dir, 'temp.png')
    # cv2.imwrite(temp_root, ms_fuse)
    #
    # img = Image.open(temp_root)
    img = torch.tensor(ms_fuse)
    img = tensor_to_image()(img)
    image_files_transforms = image_transforms(load_size)
    img = image_files_transforms(img.convert('RGB'))

    gray_image = rgb2gray(np.array(tensor_to_image()(image)))
    gray_image = image_to_tensor()(Image.fromarray(gray_image))
    edge = rgb2gray(np.array(tensor_to_image()(img)))
    edge = image_to_tensor()(Image.fromarray(edge))
    #os.remove(temp_root)

    return edge, gray_image
