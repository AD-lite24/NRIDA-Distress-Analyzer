import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

def show_image(image, bboxes):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    for box in bboxes:
        x_min, y_min, x_max, y_max = box
        rect = patches.Rectangle((y_min, x_max), x_max - x_min,
                                 y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.savefig("test.png")


def show_image_2(image):

    r, g, b = image[0], image[1], image[2]
    rgb_image = np.stack([r, g, b], axis=-1)
    plt.imshow(rgb_image)
    plt.savefig("test1.png")
