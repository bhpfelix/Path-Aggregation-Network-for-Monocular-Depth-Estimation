import torch.utils.data as data
import numpy as np
from PIL import Image
from scipy.misc import imread
from path import Path
from constants import *
from torchvision.transforms import Resize, Compose, ToPILImage, ToTensor, RandomHorizontalFlip

def load_as_float(path, single_channel=False):
    img = imread(path, flatten=single_channel).astype(np.float32)
#     if single_channel:
#         return np.expand_dims(img,axis=2)
    return img



class KittiDataset(data.Dataset):
    # Compose([ToPILImage(), Resize(TRAIN_IMG_SIZE), ToTensor()])
    def __init__(self, root='/disk2/data/', seed=None, train=True, transform=Compose([Resize(TRAIN_IMG_SIZE), RandomHorizontalFlip(), ToTensor()]), target_transform=None):
        np.random.seed(seed)
        self.root = Path(root)
        img_dir = self.root/'kitti_train_images.txt' if train else self.root/'kitti_val_images.txt'
        depth_dir = self.root/'kitti_train_depth_maps.txt' if train else self.root/'kitti_val_depth_maps.txt'
        # intr_dir = self.root/'kitti_train_intrinsics.txt' if train else self.root/'kitti_val_intrinsics.txt'
        self.img_l_paths = [d[:-1] for d in open(img_dir) if 'image_02' in d]
        self.depth_l_paths = [d[:-1] for d in open(depth_dir) if 'image_02' in d]

        # at least 20 frames between 2 examples
        del_idxs = []
        depth_idxs = []
        cur = 0
        for i in range(1,len(self.img_l_paths)):
            idx = int(self.img_l_paths[i][-7:-4])
            cur_idx = int(self.img_l_paths[cur][-7:-4])
            if abs(idx-cur_idx) < 3:
                del_idxs += [i]
            else:
                cur = i
        self.img_l_paths = np.delete(self.img_l_paths, del_idxs)
        self.depth_l_paths = np.delete(self.img_l_paths, depth_idxs)

        self.length = len(self.img_l_paths)
        self.transform = transform
            
    def __getitem__(self, index):
        img = Image.open(self.img_l_paths[index]) # load_as_float(self.img_l_paths[index])
        depth = Image.open(self.depth_l_paths[index])

#         print(img_l.shape, depth_l.shape)
        return [self.transform(img), self.transform(depth)]

    def __len__(self):
        return self.length

if __name__ == '__main__':
    # Testing
    dataset = KittiDataset()
    print(len(dataset))
    for item in dataset[0]:
        print(item.size())
