import random
import numpy as np
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch

class InductionDataset(Dataset):
    def __init__(self, train_or_test, dir_path = '../data/induction', use_metadata=False, transform=None):

        self.dir_path = dir_path
        self.cache_path = f'{dir_path}/all_{train_or_test}.npz'

        self.use_metadata = use_metadata
        self.transform = transform

        if not os.path.exists(self.cache_path):
            print("Generating and saving new induction dataset...")
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

def draw_circle(image, center_row, center_col, radius=2, im_size=32):
        """Draw a circle on the given image."""
        for r in range(center_row-radius, center_row+radius+1):
            for c in range(center_col-radius, center_col+radius+1):
                if (r - center_row)**2 + (c - center_col)**2 <= radius**2 and 0 <= r < im_size and 0 <= c < im_size:
                    image[r, c] = 1
        return image

def draw_line(image, center_row, center_col, line_length=4, im_size=32):
    for i in range(-line_length // 2, line_length // 2 + 1):
        if 0 <= center_row + i < im_size and 0 <= center_col < im_size:
            image[center_row + i, center_col] = 1
    return image

def draw_x(image, center_row, center_col, x_length=5, im_size=32):
    # Drawing the X centered around the center_col
# Drawing the X centered around the start_col
    for i in range(x_length):
        image[center_row - x_length // 2 + i, center_col - x_length // 2 + i] = 1
        image[center_row - x_length // 2 + i, center_col + x_length // 2 - i] = 1
    return image


def draw_diagonal(image, center_row, center_col, line_length=4, im_size=32):
    for i in range(-line_length // 2, line_length // 2 + 1):
        if 0 <= center_row + i < im_size and 0 <= center_col + i < im_size:
            image[center_row + i, center_col + i] = 1

    return image

def plot_two_objects(A, B, Ax, Ay, Bx, By, vertical=False):

    image = np.zeros((32, 32))

    # List of available drawing functions
    draw_functions = [draw_circle, draw_line, draw_x, draw_diagonal]

    # Call the first chosen function
    A(image, Ax, Ay)

    # Call the second chosen function right next to the first
    B(image, Bx, By)

    if vertical:
        image = image.T

    return image

def generate_dataset(dir_path = '../data/induction'):

    # generate one of each image combo, make sure spacing makes sense
    draw_functions = [draw_circle, draw_line, draw_x, draw_diagonal]
    padding = 4
    offset = 7

    images = []
    metadata = []
    labels = [] # from 0 to 3

    for vertical in [True, False]:
        for a in range(padding, 32 - padding):
            for b in range(padding, 32 - padding - offset):
                Ax = a
                Ay = b
                Bx = Ax
                By = Ay + offset

                # Example of how to use it:
                for A in draw_functions:
                    for B in draw_functions:
                        img = plot_two_objects(A, B, Ax, Ay, Bx, By, vertical=vertical)

                        if A == B:
                            same = True
                        else:
                             same = False

                        images.append(img)
                        m = {
                            "Ax": Ax,
                            "Ay": Ay,
                            "Bx": Bx,
                            "By": By,
                            "A": A.__name__,
                            "B": B.__name__,
                            "Same": same,
                            "Vertical": vertical
                        }
                        metadata.append(m)

                        if vertical and same:
                            l = 0
                        elif vertical and not same:
                            l = 1
                        elif not vertical and same:
                            l = 2
                        elif not vertical and not same:
                            l = 3

                        labels.append(l)
    
    path = f'{dir_path}/induction_dataset.npz'
    print(f"Saving raw dataset to {path}...")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    np.savez(path, images=images, metadata=metadata, labels=labels)

    print("Balancing dataset...")
    create_balanced_dataset(dir_path)


def create_balanced_dataset(dir_path='../data/induction'):
    
    # Load the dataset
    dataset = np.load(f'{dir_path}/induction_dataset.npz', mmap_mode='r', allow_pickle=True)
    images, metadata, labels = dataset['images'], dataset['metadata'], dataset['labels']

    # Create categories based on the conditions
    categories = {
        'same_T_vertical_T': [i for i, entry in enumerate(metadata) if entry['Same'] and entry['Vertical']],
        'same_F_vertical_F': [i for i, entry in enumerate(metadata) if not entry['Same'] and not entry['Vertical']],
        'same_T_vertical_F': [i for i, entry in enumerate(metadata) if entry['Same'] and not entry['Vertical']],
        'same_F_vertical_T': [i for i, entry in enumerate(metadata) if not entry['Same'] and entry['Vertical']]
    }

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
