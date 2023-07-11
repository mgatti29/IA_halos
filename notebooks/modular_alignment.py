import numpy as np
from halotools.utils.mcrotations import random_perpendicular_directions, random_unit_vectors_3d
from halotools.utils.vector_utilities import vectors_normal_to_planes, normalized_vectors, angles_between_list_of_vectors, project_onto_plane, elementwise_dot
from halotools.utils.mcrotations import rotate_vector_collection
from halotools.empirical_models.ia_models import DimrothWatson
from halotools.empirical_models.ia_models.ia_strength_models import alignment_strength
from astropy.utils.misc import NumpyRNGContext
from halotools.utils.rotations3d import rotation_matrices_from_angles, rotation_matrices_from_vectors
from warnings import warn
from healpy.pixelfunc import ang2vec, vec2ang
from halotools.empirical_models.ia_models.ia_model_components import axes_correlated_with_input_vector, axes_correlated_with_z

# Column labels for Skysim5000
# The key is what I'm used to calling it, the value is the column in skysim
skysim_labels = {
    "ra" : "ra_true",                                                   # RA
    "dec" : "dec_true",                                                 # dec
    "redshift" : "redshiftHubble",                                      # Redshift
    "x" : "x",                                                          # Galaxy x (same as getting it from transforming RA, dec, z?)
    "y" : "y",                                                          # Galaxy y
    "z" : "z",                                                          # Galaxy z
    "halo_x" : "baseDC2/target_halo_x",                                 # Host halo x
    "halo_y" : "baseDC2/target_halo_y",                                 # Host halo y
    "halo_z" : "baseDC2/target_halo_z",                                 # Host halo z
    "halo_axisA_x" : "baseDC2/target_halo_axis_A_x",                    # Host halo major axis x
    "halo_axisA_y" : "baseDC2/target_halo_axis_A_y",                    # Host halo major axis y
    "halo_axisA_z" : "baseDC2/target_halo_axis_A_z",                    # Host halo major axis z
    "halo_mvir" : "baseDC2/host_halo_mvir",                             # Host halo virial mass, useful for getting virial radius for radial alignment
    "isCentral" : "isCentral"                                           # Whether the galaxy is a central or not
}

##################################################################################################################################################################################################
##### Helper Functions ###########################################################################################################################################################################
##################################################################################################################################################################################################

# Expand to Cartesian
def to_cartesian(cosmology, ra, dec, redshift):
    
    # Get x, y, z in units of Mpc/h
    x, y, z = ( ang2vec( ra, dec, lonlat=True ).T ) * ( cosmology.comoving_distance(redshift).value * (cosmology.H0.value/100) )

    return np.vstack( [x, y, z] ).T

def get_galaxy_positions(table, label_map):
    return np.array( table[ label_map["x"] ] ), np.array( table[ label_map["y"] ] ), np.array( table[ label_map["z"] ] )

def get_host_positions(table, label_map):
    return np.array( table[ label_map["halo_x"] ] ), np.array( table[ label_map["halo_y"] ] ), np.array( table[ label_map["halo_z"] ] )

def get_host_orientation(table, label_map):
    return np.array( table[ label_map["halo_axisA_x"] ] ), np.array( table[ label_map["halo_axisA_y"] ] ), np.array( table[ label_map["halo_axisA_z"] ] )

def get_host_rvir(table, label_map):
    mvir = np.array( table[ label_map["halo_mvir"] ] )
    return pow(mvir,1.0/3.0)

def radially_dependent_satellite_alignment_strength(table, table_keys, Lbox, radial_params):
    mask = ~table[ table_keys["isCentral"] ]            # Only grab satellites

    halo_rvir = get_host_rvir( table[mask], table_keys )
    x, y, z = get_galaxy_positions(table[mask], table_keys)
    halo_x, halo_y, halo_z = get_host_positions(table[mask], table_keys)

    r_vec, r = get_radial_vector( halo_x, halo_y, halo_z, x, y, z, Lbox )

    scaled_r = r_vec/halo_rvir

    return alignment_strength_radial_dependence(scaled_r, radial_params)

##################################################################################################################################################################################################
##### Alignment Modules ##########################################################################################################################################################################
##################################################################################################################################################################################################

# Type of alignments to modularize (that's a word?)
#    RandomAlignment       (align randomly)                                                       -    DONE    (align_randomly)
#    CentralAlignment      (align centrals with respect to host major axis)                       -    DONE    (align_to_halo)
#    SatelliteAlignment    (align satellites with respect to host major axis)                     -    DONE    (align_to_halo)
#    SubhaloAlignment      (align with respect to subhalo major axis)                             -    DONE    (align_to_halo)
#    RadialAlignment       (align with respect to radial vector from host center to satellite)    -    DONE    (align_radially)
#
# Others
#    MajorAxisSatelliteAlignment (Same thing as SatelliteAlignment? SatelliteAlignment might have been meant to do what SubhaloAlignment does, but didn't)
#    HybridSatelliteAlignment

def assign_alignment_type(alignment_type, table, table_keys):
    assert( alignment_type.lower() in ["central", "radial"] )

    params = []

    if alignment_type.lower() == "central":
        mask = table[ table_keys["isCentral"] ]
        func = align_to_halo
        keys = [ "halo_axisA_x", "halo_axisA_y", "halo_axisA_z" ]
    elif alignment_type.lower() == "radial":
        mask = ~table[ table_keys["isCentral"] ]
        func = align_radially
        keys = [ "halo_x", "halo_y", "halo_z", "x", "y", "z" ]

    for key in keys:
        if isinstance( key, tuple ):
            f = key[0]
            k = key[1]
            params.append( f( table[mask], table_keys ) )
        else:
            params.append( table[ table_keys[key] ][mask] )

    return func, params, mask

def align_randomly(N):
    major_v = random_unit_vectors_3d(N)
    inter_v = random_perpendicular_directions(major_v)
    minor_v = normalized_vectors(np.cross(major_v, inter_v))
    
    return major_v, inter_v, minor_v

# Requires:
#    halo_axisA_x, halo_axisA_y, halo_axisA_z
#    alignment strength, prim_gal_axis
# Use this for all cases where a glaxy is being oriented with respect to some halo, but pass in the right gal_type sample
# In the case of subhalo, pass in the axis values for the subhalo, not the host
def align_to_halo(halo_axisA_x, halo_axisA_y, halo_axisA_z,
                    alignment_strength, prim_gal_axis='A'):
    
    assert( ( isinstance(alignment_strength, float) ) or ( isinstance(alignment_strength, np.ndarray) ) )
    
    # Make sure the alignment strengths are in the appropriate form
    if isinstance( alignment_strength, float ):
        alignment_strength = alignment_strength * np.ones( len(halo_axisA_x) )
    
    # set prim_gal_axis orientation
    major_input_vectors = np.vstack((halo_axisA_x, halo_axisA_y, halo_axisA_z)).T
    
    return align_to_axis( major_input_vectors, alignment_strength, prim_gal_axis )

# Requires:
#    halo_x, halo_y, halo_z, halo_rvir, x, y, z
#    Lbox, prim_galaxy_axis, alignment_strength (single float or array of same length as x, y, z)
# Ignoring the gal_type mask. Assume the orientation will be calculated for all entries
# NOTE: This function assumes it will be passed a list of satellite galaxy coords and their corresponding host halo coords
# This function comes from halotools_ia.ia_model_components.RadialSatellitesAlignment.assign_satellite_orientation
#    Minor changes have been made to decouple it from an HOD model
def align_radially(halo_x, halo_y, halo_z, 
                   x, y, z, 
                   Lbox, alignment_strength, prim_gal_axis='A'):
    
    assert( ( isinstance(alignment_strength, float) ) or ( isinstance(alignment_strength, np.ndarray) ) )
        
    # calculate the radial vector between galaxy and halo center
    major_input_vectors, r = get_radial_vector(halo_x, halo_y, halo_z, x, y, z, Lbox)
    
    # check for length 0 radial vectors
    mask = (r <= 0.0) | (~np.isfinite(r))
    N_bad_axes = np.sum(mask)
    if N_bad_axes > 0:
        major_input_vectors[mask,:] = random_unit_vectors_3d(N_bad_axes)
        msg = ('{0} galaxies have a radial distance equal to zero (or infinity) from their host. '
               'These galaxies will be re-assigned random alignment vectors.'.format(int(N_bad_axes)))
        warn(msg)
    
    # Make sure the alignment strengths are in the appropriate form
    if isinstance( alignment_strength, float ):
        alignment_strength = alignment_strength * np.ones( len(x) )
    
    return align_to_axis( major_input_vectors, alignment_strength, prim_gal_axis )

# Since central, subhalo, and radial alignments all do the ame thing but with respect to a different major axis, use this function to take care of the common part
# given the reference axis, align to that
def align_to_axis(major_input_vectors, alignment_strength, prim_gal_axis="A"):
    
    assert( ( isinstance(alignment_strength, float) ) or ( isinstance(alignment_strength, np.ndarray) ) )
    # Make sure the alignment strengths are in the appropriate form
    if isinstance( alignment_strength, float ):
        alignment_strength = alignment_strength * np.ones( len(major_input_vectors) )

    A_v = axes_correlated_with_input_vector(major_input_vectors, p=alignment_strength)
    
    # check for nan vectors
    mask = (~np.isfinite(np.prod(A_v, axis=-1)))
    N_bad_axes = np.sum(mask)
    if N_bad_axes>0:
        A_v[mask,:] = random_unit_vectors_3d(N_bad_axes)
        msg = ('{0} correlated alignment axis(axes) were found to be not finite. '
               'These will be re-assigned random vectors.'.format(int(N_bad_axes)))
        warn(msg)

    # randomly set secondary axis orientation
    B_v = random_perpendicular_directions(A_v)

    # the tertiary axis is determined
    C_v = vectors_normal_to_planes(A_v, B_v)

    # depending on the prim_gal_axis, assign correlated axes
    if prim_gal_axis == 'A':
        major_v = A_v
        inter_v = B_v
        minor_v = C_v
    elif prim_gal_axis == 'B':
        major_v = B_v
        inter_v = A_v
        minor_v = C_v
    elif prim_gal_axis == 'C':
        major_v = B_v
        inter_v = C_v
        minor_v = A_v
        
    return major_v, inter_v, minor_v

##################################################################################################################################################################################################
##### Support functions for alignment modules ####################################################################################################################################################
##################################################################################################################################################################################################
# These are pulled directly from halotools_ia.ia_models.ia_model_components

def alignment_strength_radial_dependence(r, radial_params):
        """
        Parameters
        ==========
        r : array_like
            scaled radial position

        Returns
        =======
        alignment_strength : numpy.array
            array fo values bounded between [-1,1]
        """

        r = np.atleast_1d(r)
        a = radial_params["a"]
        gamma = radial_params["gamma"]

        ymax = 0.99
        ymin = -0.99

        result = np.zeros(len(r))
        result = a*(r**gamma).astype(float)

        mask = (result > ymax)
        result[mask] = ymax

        mask = (result < ymin)
        result[mask] = ymin

        return result

def get_radial_vector(halo_x, halo_y, halo_z, x, y, z, Lbox):
    """
    caclulate the radial vector for satellite galaxies

    Parameters
    ==========
    x, y, z : array_like
        galaxy positions

    halo_x, halo_y, halo_z : array_like
        host halo positions

    halo_r : array_like
        halo size

    Lbox : array_like
        array len(3) giving the simulation box size along each dimension

    Returns
    =======
    r_vec : numpy.array
        array of radial vectors of shape (Ngal, 3) between host haloes and satellites

    r : numpy.array
        radial distance
    """

    # define halo-center - satellite vector
    # accounting for PBCs
    dx = (x - halo_x)
    mask = dx>Lbox[0]/2.0
    dx[mask] = dx[mask] - Lbox[0]
    mask = dx<-1.0*Lbox[0]/2.0
    dx[mask] = dx[mask] + Lbox[0]

    dy = (y - halo_y)
    mask = dy>Lbox[1]/2.0
    dy[mask] = dy[mask] - Lbox[1]
    mask = dy<-1.0*Lbox[1]/2.0
    dy[mask] = dy[mask] + Lbox[1]

    dz = (z - halo_z)
    mask = dz>Lbox[2]/2.0
    dz[mask] = dz[mask] - Lbox[2]
    mask = dz<-1.0*Lbox[2]/2.0
    dz[mask] = dz[mask] + Lbox[2]

    r_vec = np.vstack((dx, dy, dz)).T
    r = np.sqrt(np.sum(r_vec*r_vec, axis=-1))

    return r_vec, r

def obsolete_axes_correlated_with_z(p, seed=None):
    r"""
    Calculate a list of 3d unit-vectors whose orientation is correlated
    with the z-axis (0, 0, 1).
    Parameters
    ----------
    p : ndarray
        Numpy array with shape (npts, ) defining the strength of the correlation
        between the orientation of the returned vectors and the z-axis.
        Positive (negative) values of `p` produce galaxy principal axes
        that are statistically aligned with the positive (negative) z-axis;
        the strength of this alignment increases with the magnitude of p.
        When p = 0, galaxy axes are randomly oriented.

    seed : int, optional
        Random number seed used to choose a random orthogonal direction

    Returns
    -------
    unit_vectors : ndarray
        Numpy array of shape (npts, 3)

    Notes
    -----
    The `axes_correlated_with_z` function works by modifying the standard method
    for generating random points on the unit sphere. In the standard calculation,
    the z-coordinate :math:`z = \cos(\theta)`, where :math:`\cos(\theta)` is just a
    uniform random variable. In this calculation, :math:`\cos(\theta)` is not
    uniform random, but is instead implemented as a clipped power law
    implemented with `scipy.stats.powerlaw`.
    """

    p = np.atleast_1d(p)
    npts = p.shape[0]

    with NumpyRNGContext(seed):
        phi = np.random.uniform(0, 2*np.pi, npts)
        # sample cosine theta nonuniformily to correlate with in z-axis
        if np.all(p == 0):
            uran = np.random.uniform(0, 1, npts)
            cos_t = uran*2.0 - 1.0
        else:
            k = alignment_strength(p)
            d = DimrothWatson()
            cos_t = d.rvs(k)

    sin_t = np.sqrt((1.-cos_t*cos_t))

    x = sin_t * np.cos(phi)
    y = sin_t * np.sin(phi)
    z = cos_t

    return np.vstack((x, y, z)).T

def obsolete_axes_correlated_with_input_vector(input_vectors, p=0., seed=None):
    r"""
    Calculate a list of 3d unit-vectors whose orientation is correlated
    with the orientation of `input_vectors`.

    Parameters
    ----------
    input_vectors : ndarray
        Numpy array of shape (npts, 3) storing a list of 3d vectors defining the
        preferred orientation with which the returned vectors will be correlated.
        Note that the normalization of `input_vectors` will be ignored.

    p : ndarray, optional
        Numpy array with shape (npts, ) defining the strength of the correlation
        between the orientation of the returned vectors and the z-axis.
        Default is zero, for no correlation.
        Positive (negative) values of `p` produce galaxy principal axes
        that are statistically aligned with the positive (negative) z-axis;
        the strength of this alignment increases with the magnitude of p.
        When p = 0, galaxy axes are randomly oriented.

    seed : int, optional
        Random number seed used to choose a random orthogonal direction

    Returns
    -------
    unit_vectors : ndarray
        Numpy array of shape (npts, 3)
    """

    input_unit_vectors = normalized_vectors(input_vectors)
    assert input_unit_vectors.shape[1] == 3
    npts = input_unit_vectors.shape[0]
    
    # For some reason, this function is very sensitive, and the difference between float32 and float64 drastically changes results
    # With only a single alignment strength, the np.ones(length)*alignmnet_strength gives float64 numbers
    # but pulling the satellite_slignment_strength column from the table gives float32
    # At values of exactl 1 and -1, the float64 numbers do fine, but float32 don'table
    # Not sure why. But they do. So I put in this next line
    p = np.array(p).astype("float64")

    z_correlated_axes = axes_correlated_with_z(p, seed)

    z_axes = np.tile((0, 0, 1), npts).reshape((npts, 3))

    angles = angles_between_list_of_vectors(z_axes, input_unit_vectors)
    rotation_axes = vectors_normal_to_planes(z_axes, input_unit_vectors)
    matrices = rotation_matrices_from_angles(angles, rotation_axes)

    return rotate_vector_collection(matrices, z_correlated_axes)

##################################################################################################################################################################################################
##### Projection Functionality ###################################################################################################################################################################
##################################################################################################################################################################################################

# Project the unit vector to the North Celestial Pole (NCP), which is [0,0,1] in 3D cartesian, to the plane perpendicular to the line of sight
# This projection will give the unit vector (after normalization) towards NCP in the frame of the plane, which will become the vertical basis axis
def project_NCP(lines_of_sight):
    """
    Project the unit vector to the North Celestial Pole (NCP), which is [0,0,1] in 3D cartesian, to the plane perpendicular to the line of sight
    This projection will give the unit vector (after normalization) towards North in the frame of the plane, which will become the vertical basis axis

    Parameters
    ----------
    lines_of_sight : np.ndarray
        an array of shape (npts, 3) conatining npts different lines of sight to npts number of objects.
        This is the vector perpendicular to the plane upon which that object's projection will lie

    Returns
    -------
    projected_ncp : np.ndarray
        An array of shape (npts, 3) containing the normalized projections of the NCP onto the plane of interest

    west : np.ndarray
        An array of shape (npts, 3) containing the axis pointing west. This must point "right" from the NCP
        i.e. making the line of sight our x, and the projected NCP our z, west is our y axis (using the traditional x,y,z coordinate system of x out of the board, y to the right, and z up)
        In this case, the line of sight is going into the board, the projected NCP goes up from the point where it hits the galaxy, then we cross in that order to get west
    """
    ncp = np.array([0,0,1])
    ncp = np.tile( ncp, len(lines_of_sight) ).reshape( len(lines_of_sight), 3 )

    projected_ncp = project_onto_plane( ncp, lines_of_sight )
    
    # the project ncp is the new vertical axis, get the new horizontal axis
    west = np.cross(lines_of_sight, projected_ncp)

    return normalized_vectors(projected_ncp), normalized_vectors(west)

def project_alignments_with_NCP(axes, lines_of_sight):
    """
    Projects a 3D alignment onto a 2D plane perpendicular to the line of sight. For use projecting a 3D snapshot onto a lightcone.
    The main functionality of projecting comes from halotools.utils.vector_utilities project_onto_plane.
    This function uses that result (which gives a 3D vector on the plane), and flattens it into 2D, using the North Celestial Pole (NCP) as the verical axis

    Parameters
    ----------
    axes : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d points.
        The axes in 3D cartesian coordinates to be projected.

    lines_of_sight : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d points.
        The 3D positions of the object in cartesian coordinates with the observer as the origin.

    Returns
    -------
    projecyions_2d : ndarray
        Numpy array of shape (npts, 2) storing a collection of 2d projected axes.
        These are the x,y coordinates in the frame of the plane perpendicular to the line of sight
    
    ncp : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d axes
        These axes are the unit vectors pointing towards the ncp on the plane perpendicular to the line of sight for each object.
        Note that these vectors are still represented in 3D, and the planes are defined in 3D cartesian space.
        Also note that the ncp is the vertical axis and west is the horizontal axis ("up" and right respectively)
    """
    
    projections = project_onto_plane( axes, lines_of_sight )                        # These projections are still in the same 3D cartesian space as the plane

    ncp, west = project_NCP(lines_of_sight)                                         # vertical and horizontal axes for the basis of this new 2D plane

    # Get new x and y coordinates in the basis of the plane (for each projection)
    xp = elementwise_dot( projections, west )
    yp = elementwise_dot( projections, ncp )
    projections_2d = np.vstack( [ xp, yp ] ).T

    return projections_2d, ncp, west

def get_position_angle(orientations, radians=True):
    """
    Return the angle counterclockwise from the projection of the NCP
    Angle goes from 0 to pi (assuming counterclockwise from NCP)
    Take anything in the positive west axis (right) and instead report the angle for the other half
    e.g. anything pointing NW also has a side pointing SE, report the SE angle from NCP

    Parameters:
    -----------
    orientations : np.ndarray
        ndarray of shape (npts, 2) giving the 2D vector of each orientation with respect to NCP and west. The "x" component is how much
        of the original projected (onto 3D plane perpendicular to line of sight) falls along the projected west axis, while the "y" 
        component is the same with respect to the projected NCP.

    radians : bool, default True
        Whether to return angles in radians. Returns degrees if False

    Returns:
    --------
    position_angles : np.ndarray
        ndarray of shape (npts,) returning the counterclockwise angle from the projected NCP ("y" axis)
    """
    north = np.array([0,1])
    west = np.array([1,0])

    position_angles = angles_between_list_of_vectors(orientations, north)
    signs = np.sign( elementwise_dot(orientations, west) )

    # Angle goes from 0 to pi (assuming counterclockwise from NCP)
    # Take anything in the positive west axis (right) and instead report the angle for the other half
    # e.g. anything pointing NW also has a side pointing SE, report the SE angle from NCP
    position_angles[ signs > 0 ] = np.pi - position_angles[ signs > 0 ]

    if not radians:
        return position_angles * (180/np.pi)
    return position_angles

##################################################################################################################################################################################################
##### Full Pipeline ##############################################################################################################################################################################
##################################################################################################################################################################################################

# Assume central alignment for centrals and radial alignment for satellites
def inject_ia_to_lightcone(galaxy_table, central_alignment_strength, satellite_alignment_strength, Lbox, table_keys=skysim_labels, prim_gal_axis="A"):
    
    # Steps:
    #   1 - Go from lightcone to 3D cartesian -> Actually already done in skysim
    #   2 - Align galaxies
    #   3 - Project back to 2D
    #   4 - Determine position angle
    
    # TODO: Allow alignment type to be passed in. Use that to determine what other quantities are needed
    
    align_centrals, central_params, central_mask = assign_alignment_type("central", galaxy_table, table_keys)               # Central alignment
    align_satellites, satellite_params, satellite_mask = assign_alignment_type("radial", galaxy_table, table_keys)          # Radial alignment

    if isinstance(satellite_alignment_strength, dict):
        satellite_alignment_strength = radially_dependent_satellite_alignment_strength(galaxy_table, table_keys, Lbox, satellite_alignment_strength)

    # Add other parameters not inclluded in galaxy_table
    central_params.append( central_alignment_strength )
    satellite_params.append( Lbox )
    satellite_params.append( satellite_alignment_strength )

    # Step 2: Perform the alignment
    cen_major, cen_inter, cen_minor = align_centrals( *central_params, prim_gal_axis=prim_gal_axis )
    sat_major, sat_inter, sat_minor = align_satellites( *satellite_params, prim_gal_axis=prim_gal_axis )

    # Get the coordinates for each galaxy type selection for the projection later
    x, y, z = get_galaxy_positions( galaxy_table[central_mask], table_keys )
    cen_coords = np.array([x,y,z]).T
    x, y, z = get_galaxy_positions( galaxy_table[satellite_mask], table_keys )
    sat_coords = np.array([x,y,z]).T

    # Step 3: Project
    # Project the axes to 2D planes (Perpendicular to line of sight)
    # Also return the projected ncp and west axes (in 3D cartesian) as a reference for what the projected axis means
    cen_projected_axes, cen_north, cen_west = project_alignments_with_NCP( cen_major, cen_coords )
    sat_projected_axes, sat_north, sat_west = project_alignments_with_NCP( sat_major, sat_coords )

    # Get position angle
    cen_phi = get_position_angle(cen_projected_axes)
    sat_phi = get_position_angle(sat_projected_axes)

    return (cen_phi, central_mask), (sat_phi, satellite_mask)