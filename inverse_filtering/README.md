# Removing GAN Fingerprints with Adaptive Inverse Filtering

This method estimates frequency-domain patterns associated with a reference image
class, compares them with the spectrum of a GAN-generated image, and selectively
modifies anomalous Fourier coefficients. An inverse Fourier transform then reconstructs
the processed image.

The goal is to reduce generator-specific periodic artifacts while preserving important
low-frequency image structure. Unlike a single global low-pass filter, the adaptive
mask is computed from the difference between a reference spectrum and each input
image's spectrum.

## How it works

1. **Estimate a reference spectrum.** The script
   [`src_code/noise_estimation.py`](src_code/noise_estimation.py) loads grayscale
   images, computes the magnitude of their centered 2D FFTs, and averages the spectra
   of same-sized images. By default it applies `log1p` to the mean spectrum and saves
   the result as a NumPy array.
2. **Compare the input and reference.** For each target image,
   [`src_code/adaptive_filtering.py`](src_code/adaptive_filtering.py) computes a
   grayscale, normalized, log-magnitude FFT and subtracts the reference spectrum. Only
   positive differences are retained.
3. **Build an adaptive mask.** Frequency differences above the global threshold
   (difference-map mean plus `num_stds` standard deviations) are selected. A central
   low-frequency region is excluded before the mask is dilated to include nearby
   frequency bins.
4. **Reconstruct the image.** The mask is applied to the original complex FFT
   coefficients, preserving phase, then the inverse FFT returns the result to the
   spatial domain. The output is clipped to the valid grayscale range and saved.

In the current implementation, the selected coefficients are multiplied by
`1 - 1.5 * mask`. Thus, with the default binary mask, selected coefficients are
modified by a factor of `-0.5`; this is stronger than simply zeroing or attenuating
them. The mask threshold and this multiplier should be considered when interpreting
results.

The existing filtering script loads its reference array into a variable named
`mean_real_spectrum`; use a reference spectrum with the intended meaning for your
experiment. The reference and target spectra must use the same image scaling and
transform convention, and must have compatible dimensions. The estimator skips images
whose dimensions differ from the first valid image; the filtering script resizes target
images to the reference dimensions.

## Running

The scripts contain machine-specific dataset and output paths. Edit the constants in
the `__main__` sections before running. Run both scripts from this directory so their
relative output paths resolve as expected:

```bash
python src_code/noise_estimation.py
python src_code/adaptive_filtering.py
```

The scripts use NumPy, OpenCV, and Matplotlib. Install them in your active Python
environment if they are not already available:

```bash
python -m pip install numpy opencv-python matplotlib
```

The estimator's input folder should contain grayscale-readable image files with
consistent dimensions. The filtering script processes supported image files in its
configured input folder and writes the processed images to its configured output
folder. Review the generated mask and output images, and compare image quality as well
as downstream PAD performance: frequency-pattern suppression does not by itself
guarantee improved detection or preservation of visual detail.
