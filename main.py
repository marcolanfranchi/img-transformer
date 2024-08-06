import sys
import numpy as np
import sys
import cv2
import os
import imageio
from pygifsicle import optimize


def compress_image(img, scale_percent):
    """
    Compress image by a given scale percentage
    """
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return img


def read_image(img_name, col=None):
    """
    Read image of file type .JPG, .PNG, or .JPEG, convert to selected color space, compress, then return image
    """
    img_path = 'images/' + img_name
    img = cv2.imread(img_path)

    if img is None:
        return None
    
    if col == 'bw':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # compress image by 75%
    img = compress_image(img, 25)

    return img


def get_mask(img):
    """
    Extract the white-yellow mask component from the image
    """
    _, mask = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    return mask


def increase_mask(mask, scale):
    """
    Make mask component slightly larger given a scale parameter
    """
    kernel_size = int(scale)  # Convert scale to integer
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def generate_layers(mask, max_growth):
    """
    Increase the white-yellow mask in scale range 1-20 and save the images
    """
    os.makedirs('output', exist_ok=True)
    increments = max_growth / 24
    for scale in np.arange(1+increments, max_growth+increments, increments):
        mask = increase_mask(mask, scale)
        cv2.imwrite(f'output/mask_{scale}.png', mask)


def create_gif(img_bw, layers, fps=10):
    """
    Create a gif of the images on top of the original black and white image
    """
    gif_name = 'output/moving.gif'

    with imageio.get_writer(gif_name, mode='I', fps=fps, loop=0) as writer:     
        # Add images in ascending order of scales
        for layer in layers:
            mask_path = f'output/{layer}'
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Overlay the mask on the original black and white image
            overlay = cv2.addWeighted(img_bw, 1, mask, 1, 0)
            writer.append_data(overlay)

        # Add images in descending order of scales but skip first and last
        for layer in reversed(layers[1:-1]):
            mask_path = f'output/{layer}'
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Overlay the mask on the original black and white image
            overlay = cv2.addWeighted(img_bw, 1, mask, 1, 0)
            writer.append_data(overlay)

    # clear the output folder of everything except the gif
    for layer in layers:
        os.remove(f'output/{layer}')


def main():
    # check if the program is called with the correct number of arguments
    if len(sys.argv) < 2:
        print("Error: Please provide the image file name as an argument.")
        return
    
    # get image name and specified color scheme
    img_name = sys.argv[1]
    if sys.argv[2]:
        col = sys.argv[2]
    else:
        col = None

    # read the image
    image = read_image(img_name, col)

    # extract the mask
    mask = get_mask(image)

    # generate the layers (mask scaled up)
    generate_layers(mask, 2.5)

    # get the layers and sort them by scale (ascending)
    layers = [layer for layer in os.listdir('output') if 'mask' in layer]
    layers = sorted(layers, key=lambda x: float(x.split('_')[1][:-4]))

    # create the gif
    create_gif(image, layers, fps=12)

    print("Gif created successfully! (output/moving.gif)")



if __name__ == '__main__':
    main()
