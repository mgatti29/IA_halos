import numpy as np
from halotools.empirical_models.ia_models.ia_model_components import alignment_strength
from vonmises_distribution import VonMisesHalf
from astropy.utils.misc import NumpyRNGContext
from halotools.utils.vector_utilities import normalized_vectors
from halotools.utils.mcrotations import random_unit_vectors_2d
from warnings import warn
import pyccl

###############################################################################
##### ALIGNMENT FUNCTIONS #####################################################
###############################################################################

def align_to_axis(major_input_vectors, alignment_strength, prim_gal_axis="A", as_vector=False):
    assert( ( isinstance(alignment_strength, float) ) or ( isinstance(alignment_strength, np.ndarray) ) )
    # Make sure the alignment strengths are in the appropriate form
    if isinstance( alignment_strength, float ):
        alignment_strength = alignment_strength * np.ones( len(major_input_vectors) )

    A_v = axes_correlated_with_input_vector(major_input_vectors, p=alignment_strength, as_vector=as_vector)

    if as_vector:
        return align_vector_to_axis(A_v, prim_gal_axis=prim_gal_axis)
    return align_angle_to_axis(A_v)

def align_vector_to_axis(A_v, prim_gal_axis="A"):
    # check for nan vectors
    mask = (~np.isfinite(np.prod(A_v, axis=-1)))
    N_bad_axes = np.sum(mask)
    if N_bad_axes>0:
        A_v[mask,:] = random_unit_vectors_2d(N_bad_axes)
        msg = ('{0} correlated alignment axis(axes) were found to be not finite. '
               'These will be re-assigned random vectors.'.format(int(N_bad_axes)))
        warn(msg)

    B_v = perpendicular_vector_2d(A_v)

    # depending on the prim_gal_axis, assign correlated axes
    if prim_gal_axis == 'A':
        major_v = A_v
        minor_v = B_v
    elif prim_gal_axis == 'B':
        major_v = B_v
        minor_v = A_v

    return major_v, minor_v

def align_angle_to_axis(A_v):
    mask = ~np.isfinite(A_v)
    N_bad_angles = np.sum(mask)
    if N_bad_angles > 0:
        A_v[mask] = np.random.uniform(0, np.pi, size=N_bad_angles)
        msg = ('{0} correlated alignment angle(s) were found to be not finite. '
               'These will be re-assigned random values.'.format(int(N_bad_angles)))
        warn(msg)
    
    return A_v

def align_to_tidal_field(sxx, syy, sxy, z, alignment_strength, prim_gal_axis="A", as_vector=False):

    assert( ( isinstance(alignment_strength, float) ) or ( isinstance(alignment_strength, np.ndarray) ) )
    
    # Make sure the alignment strengths are in the appropriate form
    if isinstance( alignment_strength, float ):
        alignment_strength = alignment_strength * np.ones( len(z) )

    phi = tidal_angle(sxx, syy, sxy, z)
    reference_loc = phi
    if as_vector:
        vecs = np.ones( ( len(z), 2 ) )
        vecs[:,0] = np.cos(phi)
        vecs[:,1] = np.sin(phi)
        reference_loc = vecs
        
    return align_to_axis( reference_loc, alignment_strength, prim_gal_axis=prim_gal_axis, as_vector=as_vector )

def align_radially():
    pass

def align_randomly(N):
    major_v = random_unit_vectors_2d(N)
    minor_v = perpendicular_vector_2d(major_v)

    return major_v, minor_v

###############################################################################
##### HELPER FUNCTIONS ########################################################
###############################################################################

def perpendicular_vector_2d(vecs):
    normal_vecs = np.zeros(vecs.shape)
    normal_vecs[:,0] = -vecs[:,1]
    normal_vecs[:,1] = vecs[:,0]

    return normal_vecs

def axes_correlated_with_north(p, seed=None):
    r"""
    Calculate a list of angles referenced to the North Celestial Pole (NCP).
    Parameters
    ----------
    p : ndarray
        Numpy array with shape (npts,) defining the strength of the correlation
        to the NCP. Positive (negative) values of `p` produce angles
        that are statistically aligned with (pi/2 radians from) the NCP;
        the strength of this alignment increases with the magnitude of p.
        When p = 0, galaxy axes are randomly oriented.

    seed : int, optional
        Random number seed used to choose a random orthogonal direction

    Returns
    -------
    angles : ndarray
        Numpy array of shape (npts,)
    """

    p = np.atleast_1d(p)
    npts = p.shape[0]

    #with NumpyRNGContext(seed):
    if np.all(p == 0):
        angles = np.random.uniform(0, np.pi, npts)
    else:
        kappa = alignment_strength(p)
        vm = VonMisesHalf()
        angles = vm.rvs(kappa=kappa, size=npts)

    return angles

def axes_correlated_with_input_vector(input_vectors, p=0., seed=None, as_vector=False):
    r"""
    Calculate a list of 2d unit-vectors whose orientation is correlated
    with the orientation of `input_vectors`.

    Parameters
    ----------
    input_vectors : ndarray
        Numpy array of shape (npts, 2) storing a list of 2d vectors defining the
        preferred orientation with which the returned vectors will be correlated.
        Note that the normalization of `input_vectors` will be ignored.
        If as_vector is false, input_vectors is assumed to be a Numpy array of
        shape (npts,) storing a list of angles from the North Celestial Pole (NCP).

    p : ndarray, optional
        Numpy array with shape (npts, ) defining the strength of the correlation
        between the orientation of the returned vectors and the input vectors.
        Default is zero, for no correlation.
        Positive (negative) values of `p` produce galaxy principal axes
        that are statistically aligned parallel (perpendicular) to the input vectors;
        the strength of this alignment increases with the magnitude of p.
        When p = 0, galaxy axes are randomly oriented.

    seed : int, optional
        Random number seed used to choose a random orthogonal direction

    as_vector : bool, optional
        If true, the input_vector parameter will be treated as an array of 2d vectors.
        If false, the input_vector parameter will be treated as an array of angles from the NCP.

    Returns
    -------
    unit_vectors : ndarray
        Numpy array of shape (npts, 2)
    """

    if as_vector:
        input_values = normalized_vectors(input_vectors)
        assert input_values.shape[1] == 2
    else:
        input_values = input_vectors
        assert len(input_values.shape) == 1
    npts = input_values.shape[0]

    # Check to ensure multiple p values get passed into axes_correlated_with_north
    if isinstance(p, float) or isinstance(p, int):
        p = np.ones(npts)*p

    # For some reason, this function is very sensitive, and the difference between float32 and float64 drastically changes results
    # With only a single alignment strength, the np.ones(length)*alignmnet_strength gives float64 numbers
    # but pulling the satellite_slignment_strength column from the table gives float32
    # At values of exactl 1 and -1, the float64 numbers do fine, but float32 don'table
    # Not sure why. But they do. So I put in this next line
    p = np.atleast_1d(p).astype("float64")

    ncp_correlated_angles = axes_correlated_with_north(p, seed)                 # Get angles correlated to the NCP
    angles = input_values                                                       # Get the given angles. If as_vector is true, adjust in next step
    if as_vector:
        angles = np.arctan2( input_values[:,1], input_values[:,0] )             # Calculate the angles of the given unit vectors

    angles = angles + ncp_correlated_angles                                     # "Rotate" by adding the correlated angle to the given angle
    angles = np.where( angles > np.pi, angles - np.pi, angles )                 # Make sure to keep all angles in the range (0,pi)
    angles = np.where( angles < 0, angles + np.pi, angles )

    # Return in the format received (following as_vector)
    if as_vector:
        vecs = np.zeros((npts,2))
        vecs[:,0] = np.cos(angles)
        vecs[:,1] = np.sin(angles)
        return vecs
    return angles

###############################################################################
##### TIDAL FUNCTIONS #########################################################
###############################################################################

def tidal_angle(sxx, syy, sxy, z, domain=(0,np.pi)):
    # Parameters here taken from Joachim's notebook
    if not domain is None:
        # If an explicit domain is given, it must be at least pi
        # This allows every angle to be represented (since we have pi symmetry)
        assert( domain[1] - domain[0] >= np.pi )
    cosmo = pyccl.Cosmology(
        Omega_c=0.22, Omega_b=0.0448, 
        h=0.71, sigma8 = 0.801, n_s= 0.963,w0=-1.00,wa=0.0, Omega_k=0.0)

    Om_m = cosmo['Omega_m']
    rho_crit = pyccl.ccllib.cvar.constants.RHO_CRITICAL
    Aia=1.0 #1.5,2
    C2=1.0
    sigma_epsilon = 0.27

    e1 = Epsilon1_NLA(cosmo, z, Aia, rho_crit, sxx, syy)
    e2 = Epsilon2_NLA(cosmo, z, Aia, rho_crit, sxy)
    # e1 (e2) is e * cos (sin) of 2*phi, so
    # phi is (1/2) * arctan(e2/e1)
    phi = np.arctan2(e2, e1)/2

    if not domain is None:
        # If a domain is given, shift the angles to resie within it
        while not ( (phi >= domain[0]) & (phi <= domain[1]) ).all():
            # If the angle is below the domain min or above the domain max,
            # shift by pi in whichever direction
            # This is why the size of the domain must be at least pi
            low = phi < domain[0]
            high = phi > domain[1]
            phi[low] += np.pi
            phi[high] -= np.pi

    return phi

# Taken from Joachim's notebook
def Epsilon1_NLA(cosmo,z,A1,rho_crit,sxx,syy):
    gz = pyccl.growth_factor(cosmo, 1./(1+z))
    Fact = -1*A1*5e-14*rho_crit*cosmo['Omega_m']/gz
    #e1_NLA =  - Fact  * (sxx - syy) 
    e1_NLA =  Fact  * (sxx - syy)
    return e1_NLA

# Taken from Joachim's notebook
def Epsilon2_NLA(cosmo,z,A1,rho_crit,sxy):
    gz = pyccl.growth_factor(cosmo, 1./(1+z))
    Fact = -1*A1*5e-14*rho_crit*cosmo['Omega_m']/gz
    e2_NLA = 2 *Fact* sxy
    return e2_NLA

