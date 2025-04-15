import numpy as np
from numpy import linalg
import scipy
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

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

    # Convert to numpy array at the end if needed
    pixels = np.array(pixels)

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

def create_lines_from_impact_parameters(impact_params, line_length=3.0, line_density=1000):
    """
    Creates a set of horizontal lines based on impact parameters.
    
    Parameters:
    -----------
    impact_params : array-like
        Array of impact parameters (vertical distances from center)
    line_length : float
        Length of each line in meters
    line_density : int
        Number of points to generate for each line
    
    Returns:
    --------
    lines : numpy.ndarray
        3D array with shape (num_lines, line_density, 2)
    """
    num_lines = len(impact_params)
    
    # Initialize output array
    lines = np.empty((num_lines, line_density, 2))
    
    # Calculate the start and end x-coordinates for all lines
    x_start = -line_length / 2
    x_end = line_length / 2
    
    # Create x-coordinates array (same for all lines)
    x_coords = np.linspace(x_start, x_end, line_density)
    
    # Create each line
    for i, impact_param in enumerate(impact_params):
        # For each impact parameter, create a horizontal line
        # with y-coordinate equal to the impact parameter
        y_coords = np.full(line_density, impact_param)
        
        # Store in output array
        lines[i, :, 0] = x_coords  # x-coordinates
        lines[i, :, 1] = y_coords  # y-coordinates
    
    return lines

def heat_map(timeArr, pixel_centers, g):
    #https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap

    fig, ax = plt.subplots()
    c = ax.contourf(timeArr, pixel_centers, g)  # Use pixel_centers instead of np.linspace
    ax.set_xlim([2.5, 15])  # Set the x-axis limits

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Radial Position (m)')
    ax.set_title(f'WHAM Reconstructed Density')

    fig.colorbar(c, ax=ax, label='Density (arb. units)')
    plt.show()
    

def main():
    tree = 'wham'
    shot = 241227100

    # Load test data for shinethru
    dataObjShinethru = np.load('/home/epenne/python_scripts/WHAMpy/data/241227100_shinethru.npz')
    # This is the line integrated density for the nth shinethru detector
    f = dataObjShinethru['lineIntegratedDensArr']
    print(f.min())
    timeArr = np.linspace(-5, 30, f.shape[1]) # ms

    # Corresponding impact parameters
    impactParamArr = np.array([2.75, 2.34, 1.7, 1.35, 0.79, -0.71, -1.07, -1.66, -1.98, -2.55, -0.41, -0.13, 0.22, 0.42]) * 2.54 / 1e2

    lines = create_lines_from_impact_parameters(impactParamArr)

    T, pixels = get_T_matrix(lines, len(lines))
    pixel_centers = [(p[0]+p[1])/2 for p in pixels]  # Midpoints of radial bins
    print(T.shape)

    # Replace least squares with Ridge regression
    # You can adjust the alpha parameter (regularization strength)
    ridge = Ridge(alpha=1.0, fit_intercept=False)  # fit_intercept=False since we don't want an intercept term
    
    # Initialize array to store results
    g = np.zeros((T.shape[1], f.shape[1]))
    
    # Fit Ridge regression for each time point
    for i in range(f.shape[1]):
        ridge.fit(T, f[:, i])
        g[:, i] = ridge.coef_
    
    print(g.shape)
    print(g.min())

def temp():
    for data in g:
        plt.plot(data)
    
    ax = plt.gca()
    ax.set_ylim([-0.3e23, 0.3e23])
    plt.show()
        
if __name__ == "__main__":
    main()