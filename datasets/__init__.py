from .data import *
from torchvision import transforms as T 
from torch.utils.data import Dataset
from PIL import Image
import os.path as osp


def get_transform(height=480, width=640, tokyo=False):
    '''
    In the original SFRS implementation, it's Resize(max(height, width)) when tokyo=True
    Here we use min to reduce time at the cost of a little performance drop.

    '''
    transform = [T.Resize(min(height, width) if tokyo else (height, width)),
                T.ToTensor(),
                T.Normalize(mean=[0.48501960784313836, 0.4579568627450961, 0.4076039215686255],
                           std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])]
    return T.Compose(transform)


class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        else:
            return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, x, y = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if (self.transform is not None):
            img = self.transform(img)

        return img, index
