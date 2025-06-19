import numpy as np
import pandas as pd
import skimage as sk
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.stats import linregress
import json


def plot_map(map, bar):
    fig, ax = plt.subplots(figsize=(8, 6))
    map = ax.imshow(map, cmap='YlGnBu')
    fig.colorbar(map, label=bar)
    ax.set_title('Heatmap')
    ax.set_axis_off()
    return fig, ax


def detect_references(count_map, ref_loadings):
    # Smooth map to remove noise and normalize the image
    smoothed_map = sk.filters.gaussian(count_map, sigma=3)
    normalized = (smoothed_map - smoothed_map.min()) / (np.ptp(smoothed_map))

    # Edge detection
    edge_map = sk.feature.canny(normalized, sigma=1)

    # Circle transform detection
    circle_radii = np.arange(40, 60, 1)  # Try circles with radii 40-60px
    circle_res = sk.transform.hough_circle(edge_map, circle_radii)

    # Circle detection
    accums, cx, cy, radii = sk.transform.hough_circle_peaks(circle_res, circle_radii, total_num_peaks=len(ref_loadings))

    # Determine lading and info
    reference_lookup = []
    for (x, y, r) in zip(cx, cy, radii):
        mask = circular_mask(count_map.shape, (x, y), r)
        mean = count_map[mask].mean()
        ref_data = {
            'pos': (x.item(), y.item()),
            'radius': r.item(),
            'counts': mean.item(),
            'loading': 0
        }
        reference_lookup.append(ref_data)

    # Sort counts and loadings and match them together
    reference_lookup.sort(key=lambda key: key['counts'])
    ref_loadings.sort()
    for ref_data, ref_loading in zip(reference_lookup, ref_loadings):
        ref_data['loading'] = ref_loading
    return reference_lookup


def circular_mask(shape, center, radius):
    """Returns a mask for a circle"""
    Y, X = np.ogrid[:shape[0], :shape[1]]  # create index grids
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    return dist_from_center <= radius


def main():
    # Create dataframe and numpy array of counts data
    counts_df = pd.read_csv("in/count_map.txt", sep=';', header=None)
    count_map = counts_df.to_numpy()

    # 22.3
    references = detect_references(count_map, [60, 102.8])

    # Linear regression: fit line to (counts → µg/cm²)
    slope, intercept, r_value, p_value, std_err = linregress([ref['counts'] for ref in references],
                                                             [ref['loading'] for ref in references])
    print(f'Detected References: {json.dumps(references,indent=4)}')
    print(f"Calibration: [pt] = {slope:.4f}c + {intercept:.4f}")
    print(f"R = {r_value:.4f}, STD-err = {std_err:.4f}")

    fig, ax = plot_map(count_map, "counts")
    for reference in references:
        ax.add_patch(patches.Circle(reference['pos'], radius=reference['radius'], color='red', fill=False))
    fig.tight_layout()
    fig.show()

    pass


if __name__ == '__main__':
    main()
