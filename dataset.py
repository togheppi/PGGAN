# Custom dataset
import torch
from PIL import Image
import torch.utils.data as data
import os
import random
import pickle

class DatasetFromFolder(data.Dataset):
    def __init__(self, data_root, dataset, subfolder='train', transform=None, resize_scale=None, crop_size=None, fliplr=False):
        super(DatasetFromFolder, self).__init__()
        self.input_path = os.path.join(data_root, dataset, subfolder)
        self.image_filenames = [x for x in sorted(os.listdir(self.input_path))]
        self.transform = transform
        self.resize_scale = resize_scale
        self.crop_size = crop_size
        self.fliplr = fliplr

        tag_fn = os.path.join(data_root, dataset) + '/' + dataset + '_label_v2.pkl'
        with open(tag_fn, 'rb') as fp:
            self.tags = pickle.load(fp)



    def __getitem__(self, index):
        # Load Image
        img_fn = os.path.join(self.input_path, self.image_filenames[index])
        img = Image.open(img_fn).convert('RGB')

        tag = torch.from_numpy(self.tags[index]).type(torch.FloatTensor)

        # preprocessing
        if self.resize_scale:
            img = img.resize((self.resize_scale, self.resize_scale), Image.BILINEAR)

        if self.crop_size:
            x = random.randint(0, self.resize_scale - self.crop_size + 1)
            y = random.randint(0, self.resize_scale - self.crop_size + 1)
            img = img.crop((x, y, x + self.crop_size, y + self.crop_size))
        if self.fliplr:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

        if self.transform is not None:
            img = self.transform(img)

        return img, tag

    def __len__(self):
        return len(self.image_filenames)
