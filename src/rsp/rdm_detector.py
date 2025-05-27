import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy.ndimage import convolve

# Notes
# - Need to calculate the threshold from false alarm rate, CPI, etc.


def gaussian_kernel(size, sigma):
    """
    Generate a 2D Gaussian kernel.

    Args:
        size (int): Size of the kernel (must be odd).
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        numpy.ndarray: 2D Gaussian kernel.
    """
    assert size % 2 == 1, "gaussian_kernel size must be odd"

    x, y = np.meshgrid(
        np.arange(-size // 2 + 1, size // 2 + 1), np.arange(-size // 2 + 1, size // 2 + 1)
    )
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / kernel.sum()


if __name__ == "__main__":
    # Sample 2D data
    data = np.array(
        [
            [1, 2, 3, 4, 5],
            [4, 4, 4, 11, 6],
            [7, 1, 9, 4, 5],
            [4, 4, 18, 11, 6],
            [4, 4, 4, 11, 6],
            [4, 4, 4, 11, 16],
        ]
    )

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(data)

    kernel = gaussian_kernel(3, 0.85)  # Gaussian kernel for smoothing
    smoothed_data = convolve(data, kernel)  # Apply convolution
    axs[1].imshow(smoothed_data)
    print(smoothed_data)

    peaks = peak_local_max(data, min_distance=2)
    speaks = peak_local_max(data, min_distance=2)

    print(f"{peaks=}")
    print(f"{speaks=}")
