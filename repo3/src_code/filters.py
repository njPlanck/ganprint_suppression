import cv2
import bm3d
import numpy as np
from skimage import io, img_as_float, img_as_ubyte
from skimage.restoration import denoise_nl_means, estimate_sigma
from numba import njit
import pywt
import os
from PIL import Image

def nl_means_filter(img):
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img_as_float(img)
    sigma_est = np.mean(estimate_sigma(img))
    denoised_image = denoise_nl_means(img,h=1.*sigma_est,
                                        fast_mode=False,patch_size=5,
                                        patch_distance=3)
    return denoised_image


def bm3d_filter(img):
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img_as_float(img)
    denoised_image = bm3d.bm3d(img,sigma_psd=0.2,stage_arg=bm3d.BM3DStages.ALL_STAGES)
    return denoised_image  

@njit
def sigma_filter(img, window_size=3, sigma=0.03):
    pad = window_size // 2
    h, w = img.shape
    denoised_image = np.zeros_like(img)
    for y in range(pad, h - pad):
        for x in range(pad, w - pad):
            sum_val = 0.0
            count = 0
            center = img[y, x]
            for dy in range(-pad, pad + 1):
                for dx in range(-pad, pad + 1):
                    ny, nx = y + dy, x + dx
                    val = img[ny, nx]
                    if abs(val - center) <= sigma:
                        sum_val += val
                        count += 1
            if count > 0:
                denoised_image[y, x] = sum_val / count
            else:
                denoised_image[y, x] = center
   
    return denoised_image



def wavelet_filter(img, wavelet_level=3, sigma_est=None):
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img_as_float(img)
    if sigma_est is None:
        coeffs2 = pywt.dwt2(img, 'db1')
        LL, (LH, HL, HH) = coeffs2
        sigma_est = np.median(np.abs(HH)) / 0.6745

    coeffs = pywt.wavedec2(img, 'db1', level=wavelet_level)

    threshold = sigma_est * np.sqrt(2 * np.log(img.size))
    new_coeffs = []
    new_coeffs.append(coeffs[0]) 
    for i in range(1, len(coeffs)):
        new_coeffs.append(tuple(pywt.threshold(c, value=threshold, mode='soft') for c in coeffs[i]))
    
    denoised_image = pywt.waverec2(new_coeffs, 'db1')

    return denoised_image

#extract fingerprint function

def extract_fingerprint(img, denoise_filter):
    '''
    This takes an image and filter as input.
    '''
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img_as_float(img)
        
    denoised = denoise_filter(img)
    
    # Check if the denoised image has a different shape and resize it.
    if denoised.shape != img.shape:
        denoised = cv2.resize(denoised, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        
    return img - denoised # raw fingerprint (float)


#for robust visualization of the fingerprint and comparison
def fingerprint_to_uint8(fp, method="robust", k=3.0):
    med = np.median(fp)
    mad = np.median(np.abs(fp - med)) + 1e-12
    sigma = mad / 0.6745
    T = max(k * sigma, 1e-6)
    fp_c = np.clip(fp - med, -T, T)
    vis = (fp_c + T) / (2 * T)
    return img_as_ubyte(vis)

def fft_transform(img):
    """Computes and returns the magnitude of the 2D FFT."""
    img_float32 = np.float32(img)
    f = np.fft.fft2(img_float32)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    magnitude_spectrum = (magnitude_spectrum - np.min(magnitude_spectrum)) / (np.max(magnitude_spectrum) - np.min(magnitude_spectrum))
    return magnitude_spectrum


def create_and_save_grid(images, output_path, rows, cols, padding=10):
    """
    Creates a grid of images from a list of numpy arrays and saves it.
    
    This function requires Pillow to be installed.
    """
    if len(images) != rows * cols:
        raise ValueError(f"Number of images ({len(images)}) does not match grid dimensions ({rows}x{cols}).")

    img_height, img_width = images[0].shape
    
   # Calculate grid dimensions with padding
    grid_width = cols * img_width + (cols - 1) * padding
    grid_height = rows * img_height + (rows - 1) * padding
    grid_image = Image.new('L', (grid_width, grid_height), color='white') 

    for i, img_data in enumerate(images):
        row = i // cols
        col = i % cols
        
        # Calculate the paste coordinates with padding
        paste_x = col * (img_width + padding)
        paste_y = row * (img_height + padding)

        # Convert numpy array to Pillow Image object
        img_pil = Image.fromarray(np.uint8(img_data * 255))
        
        # Paste the image onto the grid
        grid_image.paste(img_pil, (paste_x, paste_y))

    grid_image.save(output_path)
    print(f"Grid image with spacing saved to {output_path}")

# --- Main Script ---
# Define paths
input_path = '/home/chinasa/python_projects/denoising/images/plus/synthetic/cycleGAN/001-PLUS-FV3-Laser_PALMAR_001_01_02_01.png'
output_folder = '/home/chinasa/python_projects/denoising/output/'

try:
    # 1. Load and process images
    original_img = img_as_float(io.imread(input_path, as_gray=True))
    denoised_img = sigma_filter(original_img)
    
    # Store the raw, float fingerprint separately
    raw_fingerprint = extract_fingerprint(original_img, sigma_filter)
    
    # Create the uint8 visualization version of the fingerprint
    vis_fingerprint = fingerprint_to_uint8(raw_fingerprint)
    
    # 2. Perform FFT transformations
    fft_original = fft_transform(original_img)
    fft_denoised = fft_transform(denoised_img)
    
    # FFT must be on the raw floating-point fingerprint, not the uint8 version
    fft_fingerprint = fft_transform(raw_fingerprint)
    
    # 3. Create a list of all images in the desired order
    # Normalize the uint8 fingerprint back to [0,1] for create_and_save_grid
    images_to_grid = [
        original_img, denoised_img, vis_fingerprint / 255.0,
        fft_original, fft_denoised, fft_fingerprint
    ]

    # 4. Create and save the grid
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    file_name = os.path.basename(input_path).split('.')[0]
    grid_save_path = os.path.join(output_folder, f"{file_name}_grid.png")
    
    # Call the grid function with the corrected image list
    create_and_save_grid(images_to_grid, grid_save_path, rows=2, cols=3)

except FileNotFoundError:
    print(f"Error: The file at {input_path} was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")



