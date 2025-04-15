import numpy as np
from numpy import linalg
import scipy
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
import MDSplus as mds

"""
Global CONSTANTS
"""
LINE_LENGTH = 3.0 # Line length in meters
LINE_DENSITY = 10000 # Number of points per line (line subdivisions)
dL = LINE_LENGTH / LINE_DENSITY
NUM_LINES = 10 # Test number of sight lines to create

def get_shinethru(shotnum, treename="wham"):
    """
    Get shinethru data from MDSplus tree
    """
    tree = mds.Tree(treename, shotnum)
    basepath = "diag.shinethru.linedens.linedens_"
    impact_path = "diag.shinethru.linedens.detector_pos"

    impact_parameters = []
    times = []
    channels = []

    impact_node = tree.getNode(impact_path)
    impact_parameters = np.array(impact_node.data(), dtype=float)

    for ch in range(1, 16):  # In Python, range(1, 16) goes from 1 to 15
        if ch != 6:
            absolutepath = basepath + f"{ch:02d}"  # Using f-string for formatting
            see_node = tree.getNode(absolutepath)
            ch_time = np.array(see_node.dim_of().data(), dtype=float)
            ch_data = np.array(see_node.data(), dtype=float)
            times.append(ch_time)
            channels.append(ch_data)

    return impact_parameters, times, channels

"""
Compute the transformation matrix T for a set of lines of sight.
Lines of sight should be given as a list of 1D arrays containing the coordinates of points on the line.
numPixels is the number of pixels used in the reconstruction.
"""
def get_T_matrix(sightLines, numPixels):

    maxR = 0.1 # Maximum radius of pixels. This is set by default.
    RArr = np.linspace(0, maxR, numPixels + 1) # Create an array of pixel radii. 0 represents the center of the reconstruction.

    normLineArr = np.empty(shape=(len(sightLines), LINE_DENSITY))

    """
    Transform the lines of sight into a 2D array of normalized distances from the
    center of the reconstruciton, called point O.
    """
    for l, sight_line in enumerate(sightLines):
        for p, point in enumerate(sight_line):
            normLineArr[l][p] = np.linalg.norm(point)

    # Find the least norm (smallest radius from origin O) among all lines of sight
    minNorm = np.array([np.min(line) for line in normLineArr])
    RArr = RArr[RArr >= np.min(minNorm)] # Keep only elements greater than or equal to minNorm
    RArr = np.sort(RArr) # Sort RArr to be in ascending order, with the smallest index being the smallest element
    if RArr[0] != 0:
        RArr = np.insert(RArr, 0, 0) # If the smallest radius is not 0, insert 0 at the beginning of the array

    pixels = []
    for k in range(len(RArr) - 1):
        pix = [RArr[k], RArr[k+1]]
        pixels.append(pix)

    Nch = len(sightLines) # Number of data channels used in the reconstruction.
    Npix = len(pixels) # Number of pixels in reconstruction, minus 1 to remove the center point.

    T = np.empty((Nch, Npix)) # Initialize an empty Nch x Npix matrix.

    # Calculation for Tij; the length of the line of sight of line i through pixel j
    for i, l_norm in enumerate (normLineArr):
        for j, pix in enumerate(pixels):
            # Count points that are between pix[0] and pix[1]
            T[i, j] = np.sum((l_norm >= pix[0]) & (l_norm < pix[1]))

    T = dL * T

    return T, pixels

def main():



if __name__ == "__main__":
    main()
