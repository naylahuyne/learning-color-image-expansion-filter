import numpy as np
import scipy.ndimage as ndimage
from skimage import color
from tqdm import tqdm
import os
import imageio
import matplotlib.pyplot as plt
import cv2

def expand_image(low_res_img, M_Y, M_I, M_Q, r, m):
    low_res_yiq = color.rgb2yiq(low_res_img)
    height, width, _ = low_res_yiq.shape
    expanded_height, expanded_width = height * r, width * r

    expanded_Y = np.zeros((expanded_height, expanded_width))
    expanded_I = np.zeros((expanded_height, expanded_width))
    expanded_Q = np.zeros((expanded_height, expanded_width))

    pad_width = m // 2
    padded_Y = np.pad(low_res_yiq[:,:,0], pad_width, mode='edge')
    padded_I = np.pad(low_res_yiq[:,:,1], pad_width, mode='edge')
    padded_Q = np.pad(low_res_yiq[:,:,2], pad_width, mode='edge')

    for y in tqdm(range(height)):
        for x in range(width):
            y_patch_Y = padded_Y[y:y+m, x:x+m].flatten()
            y_patch_I = padded_I[y:y+m, x:x+m].flatten()
            y_patch_Q = padded_Q[y:y+m, x:x+m].flatten()

            h_y, h_x = y * r, x * r
            expanded_Y[h_y:h_y+r, h_x:h_x+r] = (M_Y @ y_patch_Y).reshape(r, r)
            expanded_I[h_y:h_y+r, h_x:h_x+r] = (M_I @ y_patch_I).reshape(r, r)
            expanded_Q[h_y:h_y+r, h_x:h_x+r] = (M_Q @ y_patch_Q).reshape(r, r)

    expanded_yiq = np.stack([expanded_Y, expanded_I, expanded_Q], axis=2)
    expanded_rgb = color.yiq2rgb(expanded_yiq)
    return np.clip(expanded_rgb, 0, 1)

def downsample_image(img, factor=2):
    low_res_size = (img.shape[1] // factor, img.shape[0] // factor) 
    low_res_img = cv2.resize(img, low_res_size, interpolation=cv2.INTER_CUBIC)
    return low_res_img

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')

def load_sipi_images(directory, file_numbers):

    images = []
    for filename  in file_numbers:
        filepath  = os.path.join(directory, filename)
        if os.path.exists(filepath):
            img = plt.imread(filepath) / 255.0  # Normalize
            images.append(img)
        else:
            print(f"File {filepath} not found!")
    return images

def visualize_supports(M, m, r):
    channels = len(M)
    
    for c in range(channels):
        active = M[c][2].reshape(m, m) > 0
        plt.imshow(active, cmap = 'binary', interpolation=None)
        plt.show()
    
    plt.savefig("D:\img-proc\exp\filter_supports.png")
    plt.close()

if __name__ == "__main__":
    r = 2
    m = 11
    # a_alpha0 = 20

    sipi_dir = "D:\img-proc\data" 
    test_files = [f"4.2.0{i}.tiff" if i < 10 else f"4.2.{i}.tiff" for i in range(1, 7) if i not in [2]]
    test_high_res = load_sipi_images(sipi_dir, test_files)
    test_low_res = [downsample_image(img, factor=r).clip(0, 1) for img in test_high_res]
    
    M_Y = np.load("D:/img-proc/matrix/filter_Y.npy")
    M_I = np.load("D:/img-proc/matrix/filter_I.npy")
    M_Q = np.load("D:/img-proc/matrix/filter_Q.npy")

    #Expand and evaluate
    expanded_img = expand_image(test_low_res[2], M_Y, M_I, M_Q, r, m)
    expanded_img_uint8 = (expanded_img * 255).astype(np.uint8)
    imageio.imwrite("D:/img-proc/result/expanded_img.png", expanded_img_uint8)

    cubic_img = cv2.resize(test_low_res[2], (test_low_res[2].shape[1] * r, test_low_res[2].shape[0] * r), interpolation=cv2.INTER_CUBIC)
    cubic_img_uint8 = (cubic_img * 255).clip(0, 255).astype(np.uint8)
    imageio.imwrite("D:/img-proc/result/cubic_expanded_img.png", cubic_img_uint8)

    test_low_res_uint8 = (test_low_res[2] * 255).astype(np.uint8) 
    imageio.imwrite("D:/img-proc/result/low_res_image.png", test_low_res_uint8)
    # Calculate PSNR
    psnr = calculate_psnr(test_high_res[2], expanded_img)
    psnr_cubic = calculate_psnr(test_high_res[2], cubic_img)

    print(f"PSNR Proposed: {psnr:.2f} dB")
    print(f"PSNR Cubic Expansion: {psnr_cubic:.2f} dB")

    # visualize_supports((M_Y, M_I, M_Q), m, r)


