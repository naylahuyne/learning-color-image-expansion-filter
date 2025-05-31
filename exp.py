import numpy as np
import scipy.ndimage as ndimage
from skimage import color
from tqdm import tqdm
import os
import imageio
import matplotlib.pyplot as plt
import cv2

def expand_image(low_res_img, M_1, M_2, M_3, r, m, yiq=True):
    """
    This function need to be refactored. 
    Currently, channel 1 is Y or R, channel 2 is I or G, channel 3 is Q or B. 
    For example, expanded_1 is expanded for Y or R channel.
    """
    low_res_transformed = low_res_img
    if yiq:
        low_res_transformed = color.rgb2yiq(low_res_img)
    height, width, _ = low_res_transformed.shape
    expanded_height, expanded_width = height * r, width * r

    expanded_1 = np.zeros((expanded_height, expanded_width))
    expanded_2 = np.zeros((expanded_height, expanded_width))
    expanded_3 = np.zeros((expanded_height, expanded_width))

    pad_width = m // 2
    padded_1 = np.pad(low_res_transformed[:,:,0], pad_width, mode='edge')
    padded_2 = np.pad(low_res_transformed[:,:,1], pad_width, mode='edge')
    padded_3 = np.pad(low_res_transformed[:,:,2], pad_width, mode='edge')

    for y in tqdm(range(height)):
        for x in range(width):
            y_patch_1 = padded_1[y:y+m, x:x+m].flatten()
            y_patch_2 = padded_2[y:y+m, x:x+m].flatten()
            y_patch_3 = padded_3[y:y+m, x:x+m].flatten()

            h_y, h_x = y * r, x * r
            expanded_1[h_y:h_y+r, h_x:h_x+r] = (M_1 @ y_patch_1).reshape(r, r)
            expanded_2[h_y:h_y+r, h_x:h_x+r] = (M_2 @ y_patch_2).reshape(r, r)
            expanded_3[h_y:h_y+r, h_x:h_x+r] = (M_3 @ y_patch_3).reshape(r, r)

    expanded_img = np.stack([expanded_1, expanded_2, expanded_3], axis=2)
    if yiq:
        expanded_img = color.yiq2rgb(expanded_img)
    return np.clip(expanded_img, 0, 1)

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

if __name__ == "__main__":
    r = 2
    m = 11
    a_alpha0 = 20
    yiq = True
    proposed_psnr_list = []
    cubic_psnr_list = []
    

    sipi_dir = "D:\learning-color-image-expansion-filter\data" 
    test_files = [f"4.2.0{i}.tiff" if i < 10 else f"4.2.{i}.tiff" for i in range(1, 8) if i not in [2]]
    test_high_res = load_sipi_images(sipi_dir, test_files)
    test_low_res = [downsample_image(img, factor=r).clip(0, 1) for img in test_high_res]
    
    M_1 = None
    M_2 = None
    M_3 = None
    if yiq:
        M_1 = np.load(f"D:/learning-color-image-expansion-filter/matrix/YIQ/filter_Y_{a_alpha0}.npy")
        M_2 = np.load(f"D:/learning-color-image-expansion-filter/matrix/YIQ/filter_I_{a_alpha0}.npy")
        M_3 = np.load(f"D:/learning-color-image-expansion-filter/matrix/YIQ/filter_Q_{a_alpha0}.npy")
    else:
        M_1 = np.load(f"D:/learning-color-image-expansion-filter/matrix/RGB/filter_R_{a_alpha0}.npy")
        M_2 = np.load(f"D:/learning-color-image-expansion-filter/matrix/RGB/filter_G_{a_alpha0}.npy")
        M_3 = np.load(f"D:/learning-color-image-expansion-filter/matrix/RGB/filter_B_{a_alpha0}.npy")

    for test_index in range(0, 6):
        #Expand and evaluate
        expanded_img = expand_image(test_low_res[test_index], M_1, M_2, M_3, r, m, yiq=yiq)
        expanded_img_uint8 = (expanded_img * 255).astype(np.uint8)
        imageio.imwrite("D:/learning-color-image-expansion-filter/exp/expanded_img.png", expanded_img_uint8)

        cubic_img = cv2.resize(test_low_res[test_index], (test_low_res[test_index].shape[1] * r, test_low_res[test_index].shape[0] * r), interpolation=cv2.INTER_CUBIC)
        cubic_img_uint8 = (cubic_img * 255).clip(0, 255).astype(np.uint8)
        imageio.imwrite("D:/learning-color-image-expansion-filter/exp/cubic_expanded_img.png", cubic_img_uint8)

        test_low_res_uint8 = (test_low_res[test_index] * 255).astype(np.uint8) 
        imageio.imwrite("D:/learning-color-image-expansion-filter/exp/low_res_image.png", test_low_res_uint8)
        # Calculate PSNR
        psnr = calculate_psnr(test_high_res[test_index], expanded_img)
        psnr_cubic = calculate_psnr(test_high_res[test_index], cubic_img)
        proposed_psnr_list.append(psnr)
        cubic_psnr_list.append(psnr_cubic)

        print(f"PSNR Proposed: {psnr:.2f} dB")
        print(f"PSNR Cubic Expansion: {psnr_cubic:.2f} dB")

    print(f"Mean PSNR Proposed: {np.mean(proposed_psnr_list):.2f} dB")
    print(f"Mean PSNR Cubic Expansion: {np.mean(cubic_psnr_list):.2f} dB")

    #Expand and evaluate
    # expanded_img = expand_image(test_low_res[2], M_1, M_2, M_3, r, m, yiq=yiq)
    # expanded_img_uint8 = (expanded_img * 255).astype(np.uint8)
    # imageio.imwrite("D:/learning-color-image-expansion-filter/exp/expanded_img.png", expanded_img_uint8)

    # cubic_img = cv2.resize(test_low_res[2], (test_low_res[2].shape[1] * r, test_low_res[2].shape[0] * r), interpolation=cv2.INTER_CUBIC)
    # cubic_img_uint8 = (cubic_img * 255).clip(0, 255).astype(np.uint8)
    # imageio.imwrite("D:/learning-color-image-expansion-filter/exp/cubic_expanded_img.png", cubic_img_uint8)

    # test_low_res_uint8 = (test_low_res[2] * 255).astype(np.uint8) 
    # imageio.imwrite("D:/learning-color-image-expansion-filter/exp/low_res_image.png", test_low_res_uint8)
    # # Calculate PSNR
    # psnr = calculate_psnr(test_high_res[2], expanded_img)
    # psnr_cubic = calculate_psnr(test_high_res[2], cubic_img)
    # proposed_psnr_list.append(psnr)
    # cubic_psnr_list.append(psnr_cubic)

    # print(f"PSNR Proposed: {psnr:.2f} dB")
    # print(f"PSNR Cubic Expansion: {psnr_cubic:.2f} dB")

