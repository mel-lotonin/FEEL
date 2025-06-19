import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Reference material positions and loadings
ref_centers = [(50, 150), (165, 160), (275, 150), (380, 150)]  # (x, y) in pixels
ref_radii = [50, 50, 50, 50]  # Radius of each reference sample
ref_loadings = [0, 22.3, 60, 102.8]  # Known loadings

# Target sample bounds in image
sample_bounds = [(40, 35, 155, 80), (235, 20, 345, 50)]


def circular_mask(shape, center, radius):
    """Returns a mask for a circle"""
    Y, X = np.ogrid[:shape[0], :shape[1]]  # create index grids
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    return dist_from_center <= radius


def rectangular_mask(shape, bounds):
    """Returns a mask for a rectangle"""
    x1, y1, x2, y2 = bounds
    Y, X = np.ogrid[:shape[0], :shape[1]]  # create index grids
    return (y1 <= Y) & (Y <= y2) & (x1 <= X) & (X <= x2)


# Load Data
data = pd.read_csv("in/count_map2.txt", sep=';', header=None)
count_map = data.to_numpy()  # Convert to numpy array for processing

# Plot heatmap using 'cool' colormap
plt.figure(figsize=(8, 6))
plt.imshow(count_map, cmap='YlGnBu')
plt.colorbar(label='Radiation Counts')  # Show scale
plt.title('Radiation Heatmap')
plt.axis('off')
plt.tight_layout()
plt.savefig('out/counts.png')
plt.show()

mean_counts = []  # Mean radiation count for references
for center, radius in zip(ref_centers, ref_radii):
    mask = circular_mask(count_map.shape, center, radius)  # Create mask
    mean_counts.append(count_map[mask].mean())  # Average all data in circle

# Display loadings
for loading, counts in zip(ref_loadings, mean_counts):
    print(loading, "µg/cm^2 averages", counts, "counts")

# Linear regression: fit line to (counts → µg/cm²)
slope, intercept, r_value, p_value, std_err = linregress(mean_counts, ref_loadings)
print(f"Calibration: [pt] = {slope:.4f}c + {intercept:.4f}")
print(f"R = {r_value:.4f}, STD-err = {std_err:.4f}")

# Compute loading map from counts map using calibration curve
loading_map = slope * count_map + intercept

# Display loading map
plt.figure(figsize=(8, 6))
plt.imshow(loading_map, cmap='YlGnBu')
plt.colorbar(label='Concentration (µg/cm²)')  # Show scale
plt.title('Loading Heatmap')
plt.axis('off')
plt.tight_layout()
plt.savefig('out/loading.png')
plt.show()


# Determine uniformity and loading of samples
fig, axs = plt.subplots(1, len(sample_bounds), sharey=True, sharex=True, figsize=(16, 4))
for idx, (bounds, axis) in enumerate(zip(sample_bounds, axs)):
    mask = rectangular_mask(loading_map.shape, bounds)
    mean = loading_map[mask].mean()  # Calculate mean of retangular area
    print(f"Sample {idx + 1} has an estimated loading of {mean:.4f} µg/cm^2")

    # Only examine the points inside the rectangle
    masked_loading = loading_map[mask]

    # 3. Plot histogram
    axis.hist(masked_loading, bins=8, color='skyblue', density=True)
    axis.set_xlabel('Loading (µg/cm²)')
    axis.set_ylabel('Frequency')
    axis.set_title(f'Sample {idx + 1} Uniformity')
fig.suptitle('Sample Uniformity')
plt.tight_layout()
plt.savefig('out/samples.png')
plt.show()

# Determine uniformity and estimated loading for reference samples
fig, axs = plt.subplots(2, 2, sharex=True, figsize=(16, 8))
for idx, (center, radius, axis) in enumerate(zip(ref_centers, ref_radii, axs.flatten())):
    mask = circular_mask(count_map.shape, center, radius)
    mean = loading_map[mask].mean()
    print(f"Reference {idx + 1} has an estimated loading of {mean:.4f} µg/cm^2")

    masked_loading = loading_map[mask]

    # 3. Plot histogram
    axis.hist(masked_loading, bins=8, color='skyblue', density=True)
    axis.set_xlabel('Loading (µg/cm²)')
    axis.set_ylabel('Frequency')
    axis.set_title(f'Reference {idx + 1} Uniformity')
fig.suptitle('Reference Uniformity')
plt.tight_layout()
plt.savefig('out/reference.png')
plt.show()
