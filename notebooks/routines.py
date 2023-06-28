

def visualize_ellipse(e1, e2, x, y, ax, A2d, B2d,color='red'):
    """
    Visualizes an ellipse based on the ellipticity components and additional parameters.

    Parameters:
        e1 (float): Ellipticity component e1.
        e2 (float): Ellipticity component e2.
        x (float): x-coordinate of the ellipse center.
        y (float): y-coordinate of the ellipse center.
        ax (Axes): The matplotlib axes object on which to draw the ellipse.
        A2d (float): Semi-major axis length in 2D.
        B2d (float): Semi-minor axis length in 2D.

    Returns:
        None (displays the ellipse plot).

    Example usage:
        e1 = 0.3
        e2 = -0.2
        x = 0.0
        y = 0.0
        A2d = 1.0
        B2d = 0.5
        fig, ax = plt.subplots()
        visualize_ellipse(e1, e2, x, y, ax, A2d, B2d)
        plt.show()
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import numpy as np
    ax.set_aspect('equal')

    # Compute ellipticity magnitude (e)
    e = np.sqrt(e1**2 + e2**2)

    # Compute rotation angle (phi) in radians
    phi = 0.5 * np.arctan2(e2, e1)

    # Create an Ellipse patch
    ellipse = Ellipse((x, y), width=A2d, height=B2d, angle=np.degrees(phi),
                      edgecolor='black', facecolor=color, alpha=0.3)

    # Add the ellipse to the plot
    ax.add_patch(ellipse)

    # Set plot limits
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    # Set aspect ratio and grid
    ax.set_aspect('equal')
    ax.grid(True)
    
    

def IndexToDeclRa(index, nside,nest= False):
    import healpy as hp
    import numpy as np
    theta,phi=hp.pixelfunc.pix2ang(nside ,index,nest=nest)
    return -np.degrees(theta-np.pi/2.),np.degrees(phi)
