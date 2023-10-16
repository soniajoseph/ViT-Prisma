import random
import numpy as np


def draw_circle(image, center_row, center_col, radius=2):
        """Draw a circle on the given image."""
        for r in range(center_row-radius, center_row+radius+1):
            for c in range(center_col-radius, center_col+radius+1):
                if (r - center_row)**2 + (c - center_col)**2 <= radius**2 and 0 <= r < 32 and 0 <= c < 32:
                    image[r, c] = 1
        return image

def draw_line(image, center_row, center_col, line_length=4):
    for i in range(-line_length // 2, line_length // 2 + 1):
        if 0 <= center_row + i < 32 and 0 <= center_col < 32:
            image[center_row + i, center_col] = 1
    return image

def draw_x(image, center_row, center_col, x_length=5):
    # Drawing the X centered around the center_col
# Drawing the X centered around the start_col
    for i in range(x_length):
        image[center_row - x_length // 2 + i, center_col - x_length // 2 + i] = 1
        image[center_row - x_length // 2 + i, center_col + x_length // 2 - i] = 1
    return image


def draw_diagonal(image, center_row, center_col, line_length=4):
    for i in range(-line_length // 2, line_length // 2 + 1):
        if 0 <= center_row + i < 32 and 0 <= center_col + i < 32:
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


def generate_dataset():

    # generate one of each image combo, make sure spacing makes sense
    draw_functions = [draw_circle, draw_line, draw_x, draw_diagonal]
    padding = 4
    offset = 7

    images = []
    metadata = []

    for vertical in [True, False]:
        for a in range(padding, 32 - padding):
            for b in range(padding, 32 - padding - offset):
                Ax, Ay = a
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
    
    path = '../data/induction/induction_dataset.npz'
    print(f"Saving dataset to {path}")
    np.savez(path, images=images, metadata=metadata)

    return images, metadata