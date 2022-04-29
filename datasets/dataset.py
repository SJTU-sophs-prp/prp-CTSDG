import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.canny import image_to_edge, tensor_to_gray_tensor
from datasets.transform import mask_transforms, image_transforms
from datasets.folder import make_dataset
from utils.RCFmodels import RCF


class ImageDataset(Dataset):

    def __init__(self, image_root, mask_root, edge_root, load_size, sigma=2., mode='test'):
        super(ImageDataset, self).__init__()

        self.image_files = make_dataset(dir=image_root)
        self.mask_files = make_dataset(dir=mask_root)

        self.number_image = len(self.image_files)
        self.number_mask = len(self.mask_files)

        self.sigma = sigma
        self.mode = mode

        self.load_size = load_size

        self.image_files_transforms = image_transforms(load_size)
        self.mask_files_transforms = mask_transforms(load_size)

        # Newly added for additional edge
        self.image = []
        self.edge = []
        self.gray_image = []

        pbar = range(self.number_image)
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.01)

        # Newly added for RCF edge
        self.RCF = RCF().cuda()
        checkpoint = torch.load("./utils/bsds500_pascal_model.pth")
        self.RCF.load_state_dict(checkpoint)

        print("img_number: {}".format(self.number_image))
        for i in pbar:
            image = Image.open(self.image_files[i])
            image = self.image_files_transforms(image.convert('RGB'))
            edge, gray_image = image_to_edge(self.RCF, image, load_size)
            self.image.append(image)
            self.edge.append(edge)
            self.gray_image.append(gray_image)
        print("Conversion Finished!")


        # Newly added for PDC edge
        # edge_root = "./datasets/train128pdc"
        # self.edge_files = make_dataset(dir=edge_root)
        # self.number_edge = len(self.edge_files)
        # if self.number_image == self.number_edge:
        #     print("img_number: {}".format(self.number_image))
        #     for i in pbar:
        #         image = Image.open(self.image_files[i])
        #         image = self.image_files_transforms(image.convert('RGB'))
        #         edge = Image.open(self.edge_files[i])
        #         edge = self.image_files_transforms(edge.convert('RGB'))
        #         edge = tensor_to_gray_tensor(edge)
        #         gray_image = tensor_to_gray_tensor(image)
        #
        #         self.image.append(image)
        #         self.edge.append(edge)
        #         self.gray_image.append(gray_image)
        #     print("Conversion Finished!")
        # else:
        #     print("img_number != edge_number!")




    def __getitem__(self, index):

        # image = Image.open(self.image_files[index % self.number_image])
        # image = self.image_files_transforms(image.convert('RGB'))

        if self.mode == 'train':
            mask = Image.open(self.mask_files[random.randint(0, self.number_mask - 1)])
        else:
            mask = Image.open(self.mask_files[index % self.number_mask])

        mask = self.mask_files_transforms(mask)

        threshold = 0.5
        ones = mask >= threshold
        zeros = mask < threshold

        mask.masked_fill_(ones, 1.0)
        mask.masked_fill_(zeros, 0.0)

        mask = 1 - mask

        # edge, gray_image = image_to_edge(image, load_size=self.load_size)
        # edge, gray_image = image_to_edge(image, sigma=self.sigma)

        idx = index % self.number_image
        image = self.image[idx]
        edge = self.edge[idx]
        gray_image = self.gray_image[idx]

        return image, mask, edge, gray_image

    def __len__(self):

        return self.number_image


def create_image_dataset(opts):
    image_dataset = ImageDataset(
        opts.image_root,
        opts.mask_root,
        opts.edge_root,
        opts.load_size,
        opts.sigma,
        opts.mode
    )

    return image_dataset
