import random
import numpy as np
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
from vit_prisma.dataloaders.induction import draw_circle, draw_line, draw_x, draw_diagonal

class PolygenicInductionDataset(Dataset):
    def __init__(self, train_or_test, dir_path = '../data/polygenic_induction', use_metadata=False, transform=None):

        self.dir_path = dir_path
        self.cache_path = f'{dir_path}/all_{train_or_test}.npz'

        self.use_metadata = use_metadata
        self.transform = transform

        if not os.path.exists(self.cache_path):
            print("Generating and saving new polygenic induction dataset...")
            self._generate_and_cache()

        print("Loading induction dataset from cache...", self.cache_path)
        self._load_from_cache()
        # self._normalize_and_to_tensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx][np.newaxis, :, :]
        label = self.labels[idx]
        # meta_data = self.metadata[idx]
        image = torch.from_numpy(image).float()
        if self.transform:
            image = self.transform(image)
        return image, label

    def _load_from_cache(self):
        loaded = np.load(self.cache_path, allow_pickle=True, mmap_mode='r' )
        self.images = loaded['images']
        if self.use_metadata:
            self.metadata = loaded['metadata']
        self.labels = loaded['labels']

    def _generate_and_cache(self):
        generate_dataset(self.dir_path)

    # def _normalize_and_to_tensor(self):
    #     self.images = 2 * torch.tensor(self.images, dtype=torch.float32) / 255.0 - 1 # normalize [-1,]
    #     self.labels = torch.tensor(self.labels, dtype=torch.int64)

## Functionality to generate dataset ##

def plot_four_objects(A, B, C, D, Ax, Ay, Bx, By, Cx, Cy, Dx, Dy, vertical=False):  

    image = np.zeros((64, 64))
    
    A(image, Ax, Ay, im_size=64)
    B(image, Bx, By, im_size=64)
    C(image, Cx, Cy, im_size=64)
    D(image, Dx, Dy, im_size=64)

    if vertical:
        image = image.T
    return image

def generate_dataset(dir_path = '../data/polygenic_induction'):
    draw_functions = [draw_circle, draw_line, draw_x, draw_diagonal]
    padding = 4
    offset = 7
    max_shape_size = 5  

    images = []
    metadata = []
    labels = [] 

    max_a = 64 - 3 * offset - 2 * (padding + max_shape_size)
    max_b = 64 - padding - max_shape_size

    arrangements = {
        'A A A A' : 0,
        'A B A B' : 1,
        'A B B A' : 2,
        'A A B B' : 3,
        'A B B B' : 4,
        'A A A B' : 5,
    }

    for vertical in [True, False]:
        for a in range(padding + max_shape_size, max_a):
            for b in range(padding + max_shape_size, max_b):
                Ax = a
                Ay = b
                Bx = Ax + offset
                By = Ay
                Cx = Bx + offset
                Cy = Ay
                Dx = Cx + offset
                Dy = By

                for A in draw_functions:
                    for B in draw_functions:
                        for arrangement in arrangements.keys():

                            text_label = arrangement
                            l = arrangements[arrangement] + (0 if vertical else 6)
                            arrangement = arrangement.split()

                            ta = locals()[arrangement[0]]
                            tb = locals()[arrangement[1]]
                            tc = locals()[arrangement[2]]
                            td = locals()[arrangement[3]]

                            if len(set(text_label.split())) != len(set([ta, tb, tc, td])):
                                continue

                            img = plot_four_objects(ta, tb, tc, td, Ax, Ay, Bx, By, Cx, Cy, Dx, Dy, vertical=vertical)

                            if ta == tb == tc == td:
                                all_same = True
                            else:
                                all_same = False

                            images.append(img)

                            m = {
                                "Ax": Ax,
                                "Ay": Ay,
                                "Bx": Bx,
                                "By": By,
                                "Cx": Cx,
                                "Cy": Cy,
                                "Dx": Dx,
                                "Dy": Dy,
                                "A": ta.__name__,
                                "B": tb.__name__,
                                "C": tc.__name__,
                                "D": td.__name__,
                                "Same": all_same,
                                "Vertical": vertical,
                                "pattern": text_label,
                            }

                            metadata.append(m)
                            labels.append(l)

    path = f'{dir_path}/induction_dataset.npz'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    np.savez(path, images=images, metadata=metadata, labels=labels)

    print("Balancing dataset...")
    create_balanced_dataset(dir_path)


def create_balanced_dataset(dir_path='../data/polygenic_induction'):
    
    # Load the dataset
    dataset = np.load(f'{dir_path}/induction_dataset.npz', mmap_mode='r', allow_pickle=True)
    images, metadata, labels = dataset['images'], dataset['metadata'], dataset['labels']

    # Create categories based on the conditions
    categories = {}

    for label in range(0, 12):
        categories[label] = [i for i, entry in enumerate(labels) if entry == label]

    sample_size = min(len(c) for c in categories.values())

    # Sample from each category and save as separate npz files
    for key, val in categories.items():
        indices = random.sample(val, sample_size)
        np.savez(f'{dir_path}/{key}.npz', 
                 images=images[indices], 
                 metadata=metadata[indices], 
                 labels=labels[indices])

    # Split each npz file into train/test datasets
    all_train_data, all_test_data = [], []
    for key in categories.keys():
        data = np.load(f'{dir_path}/{key}.npz', allow_pickle=True)
        train, test = train_test_split(list(zip(data['images'], data['metadata'], data['labels'])), test_size=0.1, random_state=42)
        all_train_data.extend(train)
        all_test_data.extend(test)

    # Shuffle and save combined train/test datasets
    random.shuffle(all_train_data)
    random.shuffle(all_test_data)

    train_images, train_metadata, train_labels = zip(*all_train_data)
    test_images, test_metadata, test_labels = zip(*all_test_data)

    np.savez(f'{dir_path}/all_train.npz', images=np.array(train_images), metadata=np.array(train_metadata), labels=np.array(train_labels))
    np.savez(f'{dir_path}/all_test.npz', images=np.array(test_images), metadata=np.array(test_metadata), labels=np.array(test_labels))
    print("Saved total train and test.")
