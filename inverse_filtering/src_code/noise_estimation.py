from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dctn


def compute_image_spectrum(image_path, transform_type="fft"):
    """Loads an image in grayscale (without resizing) and computes its 2D spectrum."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")

    img = img.astype(np.float32) / 255.0

    if transform_type == "fft":
        f_transform = np.fft.fft2(img)
        f_shift = np.fft.fftshift(f_transform)
        spectrum = np.abs(f_shift)
    else:
        raise ValueError("Unsupported transform_type. Use 'fft'.")

    return spectrum, img.shape


def estimate_synthetic_fingerprint(
    image_dir, transform_type="fft", log_scale=True
):
    """Computes the mean spectrum across identical-sized synthetic images across all subfolders."""
    image_extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
    image_paths = []

    # .rglob() recursively crawls through all nested directories
    for ext in image_extensions:
        image_paths.extend(Path(image_dir).rglob(ext))
        image_paths.extend(Path(image_dir).rglob(ext.upper()))

    # Remove potential duplicates if case-folding matches the same file
    image_paths = list(set(image_paths))

    if not image_paths:
        raise FileNotFoundError(
            f"No valid images found in directory or subdirectories: {image_dir}"
        )

    print(f"Processing {len(image_paths)} synthetic images across subfolders...")

    accumulated_spectrum = None
    expected_shape = None
    valid_count = 0

    for path in image_paths:
        try:
            spectrum, img_shape = compute_image_spectrum(
                path, transform_type=transform_type
            )

            if expected_shape is None:
                expected_shape = img_shape
                accumulated_spectrum = np.zeros(
                    expected_shape, dtype=np.float64
                )
            elif img_shape != expected_shape:
                print(
                    f"Skipping {path.name}: Shape {img_shape} does not match expected {expected_shape}"
                )
                continue

            accumulated_spectrum += spectrum
            valid_count += 1
        except Exception as e:
            print(f"Skipping {path.name}: {e}")

    if valid_count == 0:
        raise RuntimeError(
            "No images were successfully processed with matching dimensions."
        )

    mean_spectrum = accumulated_spectrum / valid_count

    if log_scale:
        mean_spectrum = np.log1p(mean_spectrum)

    return mean_spectrum


def save_and_plot_fingerprint(
    fingerprint, output_path="synthetic_fingerprint.png", transform_type="fft"
):
    """Visualizes and saves the mean spectrum fingerprint heatmap."""
    plt.figure(figsize=(8, 8))
    plt.imshow(fingerprint, cmap="viridis")
    plt.title(
        f"Estimated Synthetic Fingerprint ({transform_type.upper()} Mean Spectrum)"
    )
    plt.colorbar(label="Log Magnitude Spectrum")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    SYNTHETIC_DIR = "/home/chinasa/python_projects/pad_dataset/scut/spoofed"

    fingerprint = estimate_synthetic_fingerprint(
        image_dir=SYNTHETIC_DIR, transform_type="fft", log_scale=True
    )

    np.save("../gan_noise/scut_fingerprint_spoofed.npy", fingerprint)
    save_and_plot_fingerprint(
        fingerprint,
        output_path="../gan_noise/scut_fingerprint_spoofed.png",
        transform_type="fft",
    )