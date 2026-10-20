# GAN Fingerprint Removal

This repository collects experiments on detecting and suppressing fingerprints left
by generative adversarial networks (GANs) in synthetic images. The methods explore
different representations of these artifacts, from learned image reconstruction to
spatial residuals and frequency-domain patterns.

## Method highlights

| Method | Approach |
| --- | --- |
| [Autoencoder](ae/README.md) | Trains a convolutional autoencoder to reconstruct images while suppressing recurring GAN artifacts. The learned reconstruction acts as a nonlinear denoiser; outputs can also be examined through Fourier features for downstream presentation-attack-detection experiments. |
| [Inverse filtering](inverse_filtering/README.md) | Compares image spectra with a reference spectrum, creates an adaptive mask for anomalous frequencies, and applies the mask to the image's Fourier transform before inverse-transforming it. The approach protects the central low-frequency region to limit broad image changes. |
| [DCT-based methods](dct_based/README.md) | Estimates fingerprint patterns in the discrete cosine transform (DCT) domain. Implementations compare genuine and synthetic mean spectra, emphasize frequency peaks, or use regression weights, then modify DCT coefficients and reconstruct the image. |
| [Spatial filtering](spatial/README.md) | Denoises images with filters such as Non-Local Means, BM3D, sigma, and wavelet thresholding. It estimates a GAN fingerprint from image-minus-denoised residuals, averages residuals into a reference pattern, and subtracts that pattern from images. |

These are complementary experimental approaches, not a single unified pipeline. Each
subproject has its own scripts, data assumptions, and outputs; consult its README for
method details and examples.

## Repository layout

- [`ae/`](ae/) — autoencoder-based denoising and fingerprint suppression.
- [`inverse_filtering/`](inverse_filtering/) — adaptive Fourier-domain filtering.
- [`dct_based/`](dct_based/) — DCT fingerprint estimation and coefficient attacks.
- [`spatial/`](spatial/) — spatial denoising, residual fingerprint estimation, and
  reference-pattern subtraction.
- [`feat_extraction/`](feat_extraction/) — shared/exploratory Fourier, local binary
  pattern, and lightweight CNN feature-extraction scripts.
- [`pad/`](pad/) — presentation-attack-detection classifier code.

## Data and running experiments

The scripts expect image datasets and, depending on the method, pretrained model
weights or reference fingerprints. Dataset paths and output locations are configured
inside individual scripts and may be specific to the original development environment.
Before running a script, inspect its entry point and update those paths to match your
local data. The data and generated output files in this repository are experimental
artifacts and are not a universal ready-to-run dataset or benchmark.