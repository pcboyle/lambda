"""
Code to calculate and lambda for two images. 

Written by: Peter Boyle

"""

import numpy as np
from numba import njit
from numba import cuda
import math

_TETRA_POINTS = np.array(
    [[0, 1, 2, 5], [0, 4, 5, 7], [0, 2, 5, 7], [0, 2, 3, 7], [2, 5, 6, 7]],
    dtype=np.int32,
)

DEVICE = 0

_OCTANT_1_X = np.array([0, 1, 1, 0, 0, 1, 1, 0])
_OCTANT_1_Y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
_OCTANT_1_Z = np.array([0, 0, 0, 0, 1, 1, 1, 1])

_X_REFLECT = np.array([1, -1, -1, 1, 1, -1, -1, 1])
_Y_REFLECT = np.array([1, 1, -1, -1, 1, 1, -1, -1])
_Z_REFLECT = np.array([1, 1, 1, 1, -1, -1, -1, -1])

_N_VERTS_PER_CELL = 8
_N_OCTANTS = 8

_OUTPUT_COLUMNS = [
    "Lambda",
    "Theta",
    "X Dist",
    "Y Dist",
    "Z Dist",
    "DTA",
    "HU Diff",
    "X Calc",
    "Y Calc",
    "Z Calc",
    "X Ref",
    "Y Ref",
    "Z Ref",
    "HU Ref",
]

THREADS_PER_BLOCK = (8, 8, 8)
BLOCKS_PER_GRID = (64, 64, 64)

def calculate_meshlambda(
    reference_image,
    comparison_image,
    dhu=30,
    dta=0.5,
    search_envelope = None,
    max_difference=1000,
    min_difference=1.0,
    return_dvfs=False,
    mask=None,
    import_call=False,
    print_status=True,
    device = DEVICE
):
    """

    Function that calculates meshlambda between a reference and comparison image, with lambda weighted appropriately
    by dhu and dta.

    The dimensions of the reference image and comparison image must match.

    This function is called upon import with import_call set to True to ensure that it is compiled immediately.

    Parameters:
    ----------
    reference_image: 3D numpy array of floats.

    comparison_image: 3D numpy array of floats.

    dhu: float
        dhu parameter
        Default is 30

    dta: float
        Distance parameter
        Default is 0.5

    search_envelope:
        Max search distance.
        If None, defaults to 1 + int(np.ceil(dta)). 
        If less than 1 + int(np.ceil(dta)), default to 1 + int(np.ceil(dta))
        Default is None. 

    max_difference: float
        Maximum difference used to determine maximum lambda value.

    min_difference: float
        Minimum difference used to set a minimum difference threshold.

    return_dvfs: Boolean
        Sets the return mode according to:
            False: Lambda, Theta, HU Ref, HU Comp
            True:  Lambda, Theta, X Dist, Y Dist, Z Dist,
                   DTA, HU Diff, X Calc, Y Calc, Z Calc,
                   X Ref, Y Ref, Z Ref, HU Ref, HU Comp
        Default is False.

    mask: Numpy array of Booleans
        3D mask used to determine which reference voxels to use in the lambda calculation.
        True values will be used in the lambda calculation. Must be the same dimension as the reference image.
        Default is None.

    import_call: Boolean
        If True, will run all calculations but return nothing.
        Default is False.

    print_status: Boolean
        If True, will print 'Lambda calculated.' to the console.
        Default is True.
        
    device: int
        GPU Device number to perform the calculation on. 
        Default is 0. 

    Outputs:
    -------
    output_dictionary: Dictionary with keys and values corresponding to the return_dvfs parameter.

    """

    cuda.select_device(device)

    # Margin has a minimum value of 2 to ensure we are initially checking out a sphere of radius >= 1.
    margin = 1 + int(np.ceil(dta))

    if type(search_envelope) != type(None):
        if search_envelope > margin: 
            margin = 1 * int(np.ceil(margin))

    # Vertex shifts for producing the adjacent cells, relative to (0, 0, 0)
    vertex_shifts = produce_cells_sorted(margin).astype(np.int32)  
    # [n cells, n vertices per cell, (dx, dy, dz, dist)]

    # Produce scaled arrays here to avoid scaling in the iterateMeshLambda function,
    # which makes scaling easier to keep track of.
    scaled_vertex_shifts = (np.ones(vertex_shifts.shape, dtype=np.float32) * vertex_shifts.astype(np.float32) / dta).astype(np.float32)

    # Set the maximum acceptable lambda vector here so we can easily retrieve it
    # if necessary.
    max_d_xyz = margin / dta
    max_d_dhu = max_difference / dhu
    min_d_dhu = min_difference / dhu
    max_lambda = np.sqrt(max_d_xyz * max_d_xyz + max_d_xyz * max_d_xyz + max_d_xyz * max_d_xyz + max_d_dhu * max_d_dhu)
    max_lambda_vector = np.array([max_lambda, max_d_xyz, max_d_xyz, max_d_xyz, max_d_dhu, -1000.0], dtype=np.float32,)

    # Mask check.
    if type(mask) == type(None):
        reference_image_mask = np.ones(reference_image.shape).astype(bool)

    else:
        reference_image_mask = np.ones(reference_image.shape).astype(bool) * mask

    # lambda, dx, dy, dz, d_dhu
    # Set up arrays for GPU calculations

    x_size, y_size, z_size = reference_image.shape

    gpu_cell_tetrahedra = cuda.to_device(np.zeros((x_size, y_size, z_size, 4, 4), dtype=np.float32))  # Needed so each location has its own tetrahedra array
    gpu_reference_image = cuda.to_device(reference_image.astype(np.float32))
    gpu_comparison_image = cuda.to_device(comparison_image.astype(np.float32))
    gpu_reference_mask = cuda.to_device(reference_image_mask)
    gpu_meshlambda_vector = cuda.to_device(np.ones((x_size, y_size, z_size, 6), dtype=np.float32) * max_lambda_vector)  # lambda, dx, dy, dz, d_dhu, theta
    gpu_vertex_shifts = cuda.to_device(vertex_shifts)
    gpu_scaled_vertex_shifts = cuda.to_device(scaled_vertex_shifts)

    gpu_params = cuda.to_device(np.array([margin, min_difference, dta, dhu], dtype=np.float32))

    _GPU_TETRA_POINTS = cuda.to_device(_TETRA_POINTS)

    iterate_meshlambda[BLOCKS_PER_GRID, THREADS_PER_BLOCK](
        gpu_reference_image,
        gpu_comparison_image,
        gpu_reference_mask,
        gpu_meshlambda_vector,
        gpu_vertex_shifts,
        gpu_scaled_vertex_shifts,
        gpu_params,
        gpu_cell_tetrahedra,
        _GPU_TETRA_POINTS,
    )

    meshlambda_vector = gpu_meshlambda_vector.copy_to_host()
    cuda.synchronize()

    if print_status:
        print("Lambda calculated.")

    # Delete everything
    del gpu_reference_image
    del gpu_comparison_image
    del gpu_reference_mask
    del gpu_meshlambda_vector
    del gpu_vertex_shifts
    del gpu_scaled_vertex_shifts
    del gpu_cell_tetrahedra

    if import_call:
        return

    # ['Lambda', 'Theta', 'X Dist', 'Y Dist', 'Z Dist',
    #  'DTA', 'HU Diff', 'X Calc', 'Y Calc', 'Z Calc',
    #  'HU Calc', 'X Ref', 'Y Ref', 'Z Ref', 'HU Ref']
    output_dictionary = {}

    output_dictionary["Lambda"] = meshlambda_vector[:, :, :, 0]
    output_dictionary["Theta"] = meshlambda_vector[:, :, :, 5]
    output_dictionary["HU Ref"] = reference_image
    output_dictionary["HU Comp"] = comparison_image
    output_dictionary["metadata"] = {
        "dHU": dhu,
        "DTA": dta,
        "Margin": margin,
        "Max Difference": max_difference,
        "Min Difference": min_difference,
    }

    if return_dvfs:
        # position_array = np.indices(reference_image.shape)
        
        # meshlambda_vector has shape [x_dim, y_dim, z_dim, 6] where the last dim is [lambda, dx, dy, dz, d_dhu, theta]
        output_dictionary['dX'] = meshlambda_vector[:,:,:,1] * dta # swap due to python array indexing issues when plotting, now this matches original meshlambda code.
        output_dictionary['dY'] = meshlambda_vector[:,:,:,2] * dta
        output_dictionary['dZ'] = meshlambda_vector[:,:,:,3] * dta
        output_dictionary['DTA'] = np.sqrt(np.sum(meshlambda_vector[:,:,:,1:4]*meshlambda_vector[:,:,:,1:4], axis = 3)) * dta
        output_dictionary['dHU'] = meshlambda_vector[:,:,:,4] * dhu
        # output_dictionary['X Calc'] = output_dictionary['X Dist'] + position_array[0,:,:,:] # swap due to python array indexing issues when plotting, now this matches original meshlambda code.
        # output_dictionary['Y Calc'] = output_dictionary['Y Dist'] + position_array[1,:,:,:]
        # output_dictionary['Z Calc'] = output_dictionary['Z Dist'] + position_array[2,:,:,:]
        # output_dictionary['HU Calc'] = output_dictionary['HU Diff'] + reference_image
        # output_dictionary['X Ref'] = position_array[0,:,:,:] # swap due to python array indexing issues when plotting, now this matches original meshlambda code.
        # output_dictionary['Y Ref'] = position_array[1,:,:,:]
        # output_dictionary['Z Ref'] = position_array[2,:,:,:]

    return output_dictionary


@cuda.jit
def iterate_meshlambda(
    reference_image,
    comparison_image,
    reference_image_mask,
    meshlambda_vector,
    vertex_shifts,
    scaled_vertex_shifts,
    params,
    cell_tetrahedra,
    _GPU_TETRA_POINTS,
):
    """
    Calculates lambda by iterating through each location in the scaled reference image
    and comparing it to the appropriate adjacent value in the scaled comparison image.

    Parameters:
    ----------
    reference_image: 3D numpy array of floats

    comparison_image: 3D numpy array of floats

    reference_image_mask: 3D Numpy array of Booleans
        When the image mask is False, the meshlambda array value is:
            np.array([-1000.0, -1000.0, -1000.0, -1000.0])

    meshlambda_vector: 4D numpy array of floats
        Results array with shape (x_size, y_size, z_size, 6), with the
        last dimension storing [lambda, dx, dy, dz, d_dhu, theta].

    vertex_shifts: 3D numpy array of floats
        Relative shifts for each of the cells. Shape is [n_cells, _N_VERTS_PER_CELL, 4], where
        n_cells is determined by margin*margin*margin. The last dimension is (dx, dy, dz, distance)
        of each vertex.

    scaled_vertex_shifts: 3D numpy array of floats
        vertex_shifts scaled by the appropriate dhu value.

    params: 4-Vector
        Vector that contains [margin, min_difference, dta, dhu]
        min_difference is the minimum difference threshold for calculating lambda for a given position.
        If reference[x, y, z] - comparison[x, y, z] < min_difference then the lambda vector
        for that position is set to np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    cell_tetrahedra: Numpy array
        Array to hold tetrahedra values for each location.
        Has shape of (x_size, y_size, z_size, 4, 4).

    _GPU_TETRA_POINTS: Numpy Array
        Hardcoded array containing the points in each cell marking location of tetrahedral vertices.

    Outputs:
    -------
    meshlambda_vector: 4D array of floats
        Results array with shape (x_size, y_size, z_size, 6), with the
        last dimension storing [lambda, dx, dy, dz, d_dhu, theta].

    """

    x_size, y_size, z_size = reference_image.shape  # last index is fastest
    z_ind, y_ind, x_ind = cuda.grid(3)  # first index is fastest

    n_cells = vertex_shifts.shape[0]

    margin, min_difference, dta, dhu = params

    # Out of bounds check
    if x_ind < x_size and y_ind < y_size and z_ind < z_size:
        # Mask check
        if not reference_image_mask[x_ind, y_ind, z_ind]:
            for index in range(0, 6):
                meshlambda_vector[x_ind, y_ind, z_ind, index] = 0.0

        else:
            diff_value = (
                reference_image[x_ind, y_ind, z_ind]
                - comparison_image[x_ind, y_ind, z_ind]
            )

            if diff_value * diff_value < min_difference * min_difference:
                for index in range(0, 5):
                    meshlambda_vector[x_ind, y_ind, z_ind, index] = 0.0

            else:
                reference_value = reference_image[x_ind, y_ind, z_ind] / dhu
                comparison_value = comparison_image[x_ind, y_ind, z_ind] / dhu

                border = False
                if ((x_ind < margin)
                    or (x_ind >= x_size - margin)
                    or (y_ind < margin)
                    or (y_ind >= y_size - margin)
                    or (z_ind < margin)
                    or (z_ind >= z_size - margin)):
                    border = True

                x_scaled_loc_ref = x_ind / dta
                y_scaled_loc_ref = y_ind / dta
                z_scaled_loc_ref = z_ind / dta

                for cell_num in range(0, n_cells):
                    far_x = (
                        x_ind + vertex_shifts[cell_num, 6, 0]
                    )  # Index 6 corresponds to the furthest vertex of any cell.
                    far_y = y_ind + vertex_shifts[cell_num, 6, 1]
                    far_z = z_ind + vertex_shifts[cell_num, 6, 2]

                    # Want to skip cells that are outside the array.
                    if border and (
                        (far_x < 0)
                        or (far_x > x_size - 1)
                        or (far_y < 0)
                        or (far_y > y_size - 1)
                        or (far_z < 0)
                        or (far_z > z_size - 1)):
                        continue

                    # Collect tetrahedra points.
                    for tetra_index in range(0, 5):
                        for tetra_vertex_num, vertex_index in enumerate(
                            _GPU_TETRA_POINTS[tetra_index]
                        ):
                            # Make these calculations relative, add the reference vector back at the end.
                            cell_tetrahedra[
                                x_ind, y_ind, z_ind, tetra_vertex_num, 0
                            ] = (
                                scaled_vertex_shifts[cell_num, vertex_index, 0]
                                + x_scaled_loc_ref
                            )
                            cell_tetrahedra[
                                x_ind, y_ind, z_ind, tetra_vertex_num, 1
                            ] = (
                                scaled_vertex_shifts[cell_num, vertex_index, 1]
                                + y_scaled_loc_ref
                            )
                            cell_tetrahedra[
                                x_ind, y_ind, z_ind, tetra_vertex_num, 2
                            ] = (
                                scaled_vertex_shifts[cell_num, vertex_index, 2]
                                + z_scaled_loc_ref
                            )

                            cell_tetrahedra[
                                x_ind, y_ind, z_ind, tetra_vertex_num, 3
                            ] = (
                                comparison_image[
                                    int(
                                        x_ind + vertex_shifts[cell_num, vertex_index, 0]
                                    ),
                                    int(
                                        y_ind + vertex_shifts[cell_num, vertex_index, 1]
                                    ),
                                    int(
                                        z_ind + vertex_shifts[cell_num, vertex_index, 2]
                                    ),
                                ]
                                / dhu
                            )

                        # Calculate lambda, determine closest points.
                        calculate_lambda_vector(
                            x_scaled_loc_ref,
                            y_scaled_loc_ref,
                            z_scaled_loc_ref,
                            reference_value,
                            cell_tetrahedra[x_ind, y_ind, z_ind],
                            meshlambda_vector[x_ind, y_ind, z_ind],
                        )

            calculate_theta_gpu(meshlambda_vector[x_ind, y_ind, z_ind])

    return


@njit(cache=True)
def calculate_theta(meshlambda_vector, reference_image_mask):
    """

    Calculates theta (in degrees) using the meshlambda vector, which has form [x_size, y_size, z_size, (lambda, dx, dy, dz, d_dhu)].

    Uses np.arctan2(d_dhu, dist) * 180/np.pi

    """

    x_size, y_size, z_size = reference_image_mask.shape
    theta_array = np.zeros(reference_image_mask.shape)

    for x_ind in range(x_size):
        for y_ind in range(y_size):
            for z_ind in range(z_size):
                if reference_image_mask[x_ind, y_ind, z_ind]:
                    dx = meshlambda_vector[x_ind, y_ind, z_ind, 1]
                    dy = meshlambda_vector[x_ind, y_ind, z_ind, 2]
                    dz = meshlambda_vector[x_ind, y_ind, z_ind, 3]
                    d_dhu = meshlambda_vector[x_ind, y_ind, z_ind, 4]

                    dist = np.sqrt(dx * dx + dy * dy + dz * dz)  # spatial distance

                    theta = np.arctan2(d_dhu, dist) * (180 / np.pi)

                else:
                    theta = -1000.0

                theta_array[x_ind, y_ind, z_ind] = theta

    return theta_array


@cuda.jit(device=True)
def calculate_theta_gpu(lambda_vector):
    """

    Calculates theta (in degrees) using the meshlambda vector, which has form [x_size, y_size, z_size, (lambda, dx, dy, dz, d_dhu, theta)].

    Uses np.arctan2(d_dhu, dist) * 180/np.pi

    The specific call is: calculate_theta(meshlambda_vector[x_ind, y_ind, z_ind]) -> lambda_vector = [lambda, dx, dy, dz, d_dhu, theta]

    """

    dist = math.sqrt(
        lambda_vector[1] * lambda_vector[1]
        + lambda_vector[2] * lambda_vector[2]
        + lambda_vector[3] * lambda_vector[3]
    )

    lambda_vector[5] = math.atan2(lambda_vector[4], dist) * (180 / math.pi)

    return


@njit(cache=True)
def produce_cells_sorted(margin):
    """

    The results of standard margin sizes can be hardcoded in a data file.

    """

    n_cells = margin * margin * margin * _N_OCTANTS
    n_vertices = n_cells * _N_VERTS_PER_CELL

    vertex_shifts = np.zeros((n_cells, _N_VERTS_PER_CELL, 4))  # x, y z, distance
    cell_distances = np.zeros(
        n_cells
    )  # Needed to sort the cells, we use the minimum distance.

    cell_index = 0
    for reflect_ind in range(_N_OCTANTS):
        for x_shift in range(margin):
            for y_shift in range(margin):
                for z_shift in range(margin):
                    min_dist = np.sqrt(
                        margin * margin + margin * margin + margin * margin
                    )
                    for vertex_index in range(_N_VERTS_PER_CELL):

                        vertex_shift_x = (
                            _OCTANT_1_X[vertex_index] + x_shift
                        ) * _X_REFLECT[reflect_ind]
                        vertex_shift_y = (
                            _OCTANT_1_Y[vertex_index] + y_shift
                        ) * _Y_REFLECT[reflect_ind]
                        vertex_shift_z = (
                            _OCTANT_1_Z[vertex_index] + z_shift
                        ) * _Z_REFLECT[reflect_ind]

                        vertex_distance = np.sqrt(
                            vertex_shift_x * vertex_shift_x
                            + vertex_shift_y * vertex_shift_y
                            + vertex_shift_z * vertex_shift_z
                        )

                        vertex_shifts[cell_index, vertex_index, 0] = vertex_shift_x
                        vertex_shifts[cell_index, vertex_index, 1] = vertex_shift_y
                        vertex_shifts[cell_index, vertex_index, 2] = vertex_shift_z
                        vertex_shifts[cell_index, vertex_index, 3] = vertex_distance

                        if min_dist > vertex_distance:
                            min_dist = vertex_distance

                    cell_distances[cell_index] = min_dist
                    cell_index += 1

    sorted_cell_indices = np.argsort(
        cell_distances
    )  # This is the slowest part of this function.
    sorted_vertex_shifts = vertex_shifts[sorted_cell_indices]

    return sorted_vertex_shifts


@cuda.jit(device=True)
def calculate_lambda_vector(
    x_ind, y_ind, z_ind, index_hu, cell_tetrahedra, lambda_vector
):
    """
    Get both the cell tetrahedra values and the closest points

    Uses get_tetra_distances_obscene and get_closest_distance.

    """

    # 1 tetrahedra with 4 weights each.

    weight_epsilon = 0.000001

    # Get difference between reference point and the designated tetrahedra v4's.
    POI_R0 = x_ind - cell_tetrahedra[3, 0]
    POI_R1 = y_ind - cell_tetrahedra[3, 1]
    POI_R2 = z_ind - cell_tetrahedra[3, 2]
    POI_R3 = index_hu - cell_tetrahedra[3, 3]

    V_00 = cell_tetrahedra[0, 0] - cell_tetrahedra[3, 0]  # dX1
    V_01 = cell_tetrahedra[1, 0] - cell_tetrahedra[3, 0]  # dX2
    V_02 = cell_tetrahedra[2, 0] - cell_tetrahedra[3, 0]  # dX3

    V_10 = cell_tetrahedra[0, 1] - cell_tetrahedra[3, 1]  # dY1
    V_11 = cell_tetrahedra[1, 1] - cell_tetrahedra[3, 1]  # dY2
    V_12 = cell_tetrahedra[2, 1] - cell_tetrahedra[3, 1]  # dY3

    V_20 = cell_tetrahedra[0, 2] - cell_tetrahedra[3, 2]  # dZ1
    V_21 = cell_tetrahedra[1, 2] - cell_tetrahedra[3, 2]  # dZ2
    V_22 = cell_tetrahedra[2, 2] - cell_tetrahedra[3, 2]  # dZ3

    V_30 = cell_tetrahedra[0, 3] - cell_tetrahedra[3, 3]  # dH1
    V_31 = cell_tetrahedra[1, 3] - cell_tetrahedra[3, 3]  # dH2
    V_32 = cell_tetrahedra[2, 3] - cell_tetrahedra[3, 3]  # dH3

    # V_transpose times V matrix, or VTV
    VTV_00 = V_00 * V_00 + V_10 * V_10 + V_20 * V_20 + V_30 * V_30
    VTV_01 = V_00 * V_01 + V_10 * V_11 + V_20 * V_21 + V_30 * V_31
    VTV_02 = V_00 * V_02 + V_10 * V_12 + V_20 * V_22 + V_30 * V_32

    # VTV_10 = V_01*V_00 + V_11*V_10 + V_21*V_20 + V_31*V_30 = VTV_01
    VTV_11 = V_01 * V_01 + V_11 * V_11 + V_21 * V_21 + V_31 * V_31
    VTV_12 = V_01 * V_02 + V_11 * V_12 + V_21 * V_22 + V_31 * V_32

    # VTV_20 = V_02*V_00 + V_12*V_10 + V_22*V_20 + V_32*V_30 = VTV_02
    # VTV_21 = V_02*V_01 + V_12*V_11 + V_22*V_21 + V_32*V_31 = VTV_12
    VTV_22 = V_02 * V_02 + V_12 * V_12 + V_22 * V_22 + V_32 * V_32

    # inverse VTV matrix, or VTVi
    VTVi_00 = VTV_11 * VTV_22 - VTV_12 * VTV_12
    VTVi_01 = VTV_02 * VTV_12 - VTV_22 * VTV_01
    VTVi_02 = VTV_01 * VTV_12 - VTV_11 * VTV_02

    # VTVi_10 is the same value as VTVi_01
    VTVi_11 = VTV_00 * VTV_22 - VTV_02 * VTV_02
    VTVi_12 = VTV_02 * VTV_01 - VTV_12 * VTV_00

    # VTVi_20 is the same value as VTVi_02
    # VTVi_21 is the same value as VTVi_12
    VTVi_22 = VTV_00 * VTV_11 - VTV_01 * VTV_01

    VTV_det = VTV_00 * VTVi_00 + VTV_01 * VTVi_01 + VTV_02 * VTVi_02

    # VTVi time V transpose matrix, or VTViVT
    VTViVT_00 = VTVi_00 * V_00 + VTVi_01 * V_01 + VTVi_02 * V_02
    VTViVT_01 = VTVi_00 * V_10 + VTVi_01 * V_11 + VTVi_02 * V_12
    VTViVT_02 = VTVi_00 * V_20 + VTVi_01 * V_21 + VTVi_02 * V_22
    VTViVT_03 = VTVi_00 * V_30 + VTVi_01 * V_31 + VTVi_02 * V_32

    VTViVT_10 = VTVi_01 * V_00 + VTVi_11 * V_01 + VTVi_12 * V_02
    VTViVT_11 = VTVi_01 * V_10 + VTVi_11 * V_11 + VTVi_12 * V_12
    VTViVT_12 = VTVi_01 * V_20 + VTVi_11 * V_21 + VTVi_12 * V_22
    VTViVT_13 = VTVi_01 * V_30 + VTVi_11 * V_31 + VTVi_12 * V_32

    VTViVT_20 = VTVi_02 * V_00 + VTVi_12 * V_01 + VTVi_22 * V_02
    VTViVT_21 = VTVi_02 * V_10 + VTVi_12 * V_11 + VTVi_22 * V_12
    VTViVT_22 = VTVi_02 * V_20 + VTVi_12 * V_21 + VTVi_22 * V_22
    VTViVT_23 = VTVi_02 * V_30 + VTVi_12 * V_31 + VTVi_22 * V_32

    # Weights
    wt0 = (
        VTViVT_00 * POI_R0
        + VTViVT_01 * POI_R1
        + VTViVT_02 * POI_R2
        + VTViVT_03 * POI_R3
    ) / VTV_det
    wt1 = (
        VTViVT_10 * POI_R0
        + VTViVT_11 * POI_R1
        + VTViVT_12 * POI_R2
        + VTViVT_13 * POI_R3
    ) / VTV_det
    wt2 = (
        VTViVT_20 * POI_R0
        + VTViVT_21 * POI_R1
        + VTViVT_22 * POI_R2
        + VTViVT_23 * POI_R3
    ) / VTV_det
    wt3 = 1 - wt0 - wt1 - wt2

    wt0_bool = 1 if wt0 > weight_epsilon else 0
    wt1_bool = 1 if wt1 > weight_epsilon else 0
    wt2_bool = 1 if wt2 > weight_epsilon else 0
    wt3_bool = 1 if wt3 > weight_epsilon else 0

    wts_sum = wt0_bool * wt0 + wt1_bool * wt1 + wt2_bool * wt2 + wt3_bool * wt3

    dx = (
        (
            wt0 * cell_tetrahedra[0, 0] * wt0_bool
            + wt1 * cell_tetrahedra[1, 0] * wt1_bool
            + wt2 * cell_tetrahedra[2, 0] * wt2_bool
            + wt3 * cell_tetrahedra[3, 0] * wt3_bool
        )
        / wts_sum
    ) - x_ind
    dy = (
        (
            wt0 * cell_tetrahedra[0, 1] * wt0_bool
            + wt1 * cell_tetrahedra[1, 1] * wt1_bool
            + wt2 * cell_tetrahedra[2, 1] * wt2_bool
            + wt3 * cell_tetrahedra[3, 1] * wt3_bool
        )
        / wts_sum
    ) - y_ind
    dz = (
        (
            wt0 * cell_tetrahedra[0, 2] * wt0_bool
            + wt1 * cell_tetrahedra[1, 2] * wt1_bool
            + wt2 * cell_tetrahedra[2, 2] * wt2_bool
            + wt3 * cell_tetrahedra[3, 2] * wt3_bool
        )
        / wts_sum
    ) - z_ind
    dhu = (
        (
            wt0 * cell_tetrahedra[0, 3] * wt0_bool
            + wt1 * cell_tetrahedra[1, 3] * wt1_bool
            + wt2 * cell_tetrahedra[2, 3] * wt2_bool
            + wt3 * cell_tetrahedra[3, 3] * wt3_bool
        )
        / wts_sum
    ) - index_hu

    distance = math.sqrt(dx * dx + dy * dy + dz * dz + dhu * dhu)

    if lambda_vector[0] > distance:

        lambda_vector[0] = distance
        lambda_vector[1] = dx
        lambda_vector[2] = dy
        lambda_vector[3] = dz
        lambda_vector[4] = dhu

    return


@cuda.jit(device=True)
def get_closest_distance(reference_vector, closest_points, lambda_vector):
    """
    Calculate index of closest point in the cell_tetrahedra.

    Parameters:
    ----------
    reference_vector: 4-Vector

    closest_points: 5x4 vector, one for each cell point

    lambda_vector: 5-vector

    Outputs:
    -------
    lambda_vector:

    """

    for cell_index in range(5):
        dx = closest_points[cell_index][0] - reference_vector[0]
        dy = closest_points[cell_index][1] - reference_vector[1]
        dz = closest_points[cell_index][2] - reference_vector[2]
        dhu = closest_points[cell_index][3] - reference_vector[3]

        distance = math.sqrt(dx * dx + dy * dy + dz * dz + dhu * dhu)

        if lambda_vector[0] < distance:

            lambda_vector[0] = distance
            lambda_vector[1] = dx
            lambda_vector[2] = dy
            lambda_vector[3] = dz
            lambda_vector[4] = dhu

    return


@cuda.jit(device=True)
def get_tetra_distances_obscene(point_of_interest, cell_tetrahedra, closest_points):
    """
    Calculate best distance to all 5 tetrahedral simplices at once.

    This version of the function was written to minimize for loops, rewriting array elements, or other
    tasks that needlessly extend computation time.

    Variable names are shortened to a seemingly obfuscating degree to prevent lines from being very long.
    They are listed here for reference:

        POI_R - Point of Interest minus tetrahedral reference point.
        V - Tetrahedra points minus tetrahedral reference point.
        VTV - V transpose times V matrix.
        VTVi - Inverse VTV matrix.
        VTViVT - Inverse VTV times V transpose matrix.

    See getTetraDist function for a more readable version.

    Parameters:
    ----------
    point_of_interest: numpy array of floats
        1xN array of points, with N being the number of coordinates. This is the point we are measuring the distance to.

    cell_tetrahedra: numpy array of floats
        5x4xN array of tetrahedra points making up each cell, with N being the number of coordinates.
        The 4th point is the tetrahedral reference point.

    closest_points: numpy array of floats
        Output array:
            closest_points = np.zeros((5, 4))

    Outputs:
    -------
    closest_points: numpy array of floats
        5x4 array of points to the closest point in each tetrahedra.

    """

    # 5 tetrahedra with 4 weights each.

    weight_epsilon = 0.000001

    # Get weights. Iterate through the 5 tetrahedra and assign weight.
    for r in range(5):

        # Get difference between reference point and the designated tetrahedra v4's.
        POI_R0 = point_of_interest[0] - cell_tetrahedra[r][3, 0]
        POI_R1 = point_of_interest[1] - cell_tetrahedra[r][3, 1]
        POI_R2 = point_of_interest[2] - cell_tetrahedra[r][3, 2]
        POI_R3 = point_of_interest[3] - cell_tetrahedra[r][3, 3]

        # POI_R = np.array([point_of_interest[0] - cell_tetrahedra[r][3, 0],
        #                   point_of_interest[1] - cell_tetrahedra[r][3, 1],
        #                   point_of_interest[2] - cell_tetrahedra[r][3, 2],
        #                   point_of_interest[3] - cell_tetrahedra[r][3, 3]])

        V_00 = cell_tetrahedra[r][0, 0] - cell_tetrahedra[r][3, 0]  # dX1
        V_01 = cell_tetrahedra[r][1, 0] - cell_tetrahedra[r][3, 0]  # dX2
        V_02 = cell_tetrahedra[r][2, 0] - cell_tetrahedra[r][3, 0]  # dX3

        V_10 = cell_tetrahedra[r][0, 1] - cell_tetrahedra[r][3, 1]  # dY1
        V_11 = cell_tetrahedra[r][1, 1] - cell_tetrahedra[r][3, 1]  # dY2
        V_12 = cell_tetrahedra[r][2, 1] - cell_tetrahedra[r][3, 1]  # dY3

        V_20 = cell_tetrahedra[r][0, 2] - cell_tetrahedra[r][3, 2]  # dZ1
        V_21 = cell_tetrahedra[r][1, 2] - cell_tetrahedra[r][3, 2]  # dZ2
        V_22 = cell_tetrahedra[r][2, 2] - cell_tetrahedra[r][3, 2]  # dZ3

        V_30 = cell_tetrahedra[r][0, 3] - cell_tetrahedra[r][3, 3]  # dH1
        V_31 = cell_tetrahedra[r][1, 3] - cell_tetrahedra[r][3, 3]  # dH2
        V_32 = cell_tetrahedra[r][2, 3] - cell_tetrahedra[r][3, 3]  # dH3

        # V_transpose times V matrix, or VTV
        VTV_00 = V_00 * V_00 + V_10 * V_10 + V_20 * V_20 + V_30 * V_30
        VTV_01 = V_00 * V_01 + V_10 * V_11 + V_20 * V_21 + V_30 * V_31
        VTV_02 = V_00 * V_02 + V_10 * V_12 + V_20 * V_22 + V_30 * V_32

        # VTV_10 = V_01*V_00 + V_11*V_10 + V_21*V_20 + V_31*V_30 = VTV_01
        VTV_11 = V_01 * V_01 + V_11 * V_11 + V_21 * V_21 + V_31 * V_31
        VTV_12 = V_01 * V_02 + V_11 * V_12 + V_21 * V_22 + V_31 * V_32

        # VTV_20 = V_02*V_00 + V_12*V_10 + V_22*V_20 + V_32*V_30 = VTV_02
        # VTV_21 = V_02*V_01 + V_12*V_11 + V_22*V_21 + V_32*V_31 = VTV_12
        VTV_22 = V_02 * V_02 + V_12 * V_12 + V_22 * V_22 + V_32 * V_32

        # inverse VTV matrix, or VTVi
        VTVi_00 = VTV_11 * VTV_22 - VTV_12 * VTV_12
        VTVi_01 = VTV_02 * VTV_12 - VTV_22 * VTV_01
        VTVi_02 = VTV_01 * VTV_12 - VTV_11 * VTV_02

        # VTVi_10 is the same value as VTVi_01
        VTVi_11 = VTV_00 * VTV_22 - VTV_02 * VTV_02
        VTVi_12 = VTV_02 * VTV_01 - VTV_12 * VTV_00

        # VTVi_20 is the same value as VTVi_02
        # VTVi_21 is the same value as VTVi_12
        VTVi_22 = VTV_00 * VTV_11 - VTV_01 * VTV_01

        VTV_det = VTV_00 * VTVi_00 + VTV_01 * VTVi_01 + VTV_02 * VTVi_02

        # VTVi time V transpose matrix, or VTViVT
        VTViVT_00 = VTVi_00 * V_00 + VTVi_01 * V_01 + VTVi_02 * V_02
        VTViVT_01 = VTVi_00 * V_10 + VTVi_01 * V_11 + VTVi_02 * V_12
        VTViVT_02 = VTVi_00 * V_20 + VTVi_01 * V_21 + VTVi_02 * V_22
        VTViVT_03 = VTVi_00 * V_30 + VTVi_01 * V_31 + VTVi_02 * V_32

        VTViVT_10 = VTVi_01 * V_00 + VTVi_11 * V_01 + VTVi_12 * V_02
        VTViVT_11 = VTVi_01 * V_10 + VTVi_11 * V_11 + VTVi_12 * V_12
        VTViVT_12 = VTVi_01 * V_20 + VTVi_11 * V_21 + VTVi_12 * V_22
        VTViVT_13 = VTVi_01 * V_30 + VTVi_11 * V_31 + VTVi_12 * V_32

        VTViVT_20 = VTVi_02 * V_00 + VTVi_12 * V_01 + VTVi_22 * V_02
        VTViVT_21 = VTVi_02 * V_10 + VTVi_12 * V_11 + VTVi_22 * V_12
        VTViVT_22 = VTVi_02 * V_20 + VTVi_12 * V_21 + VTVi_22 * V_22
        VTViVT_23 = VTVi_02 * V_30 + VTVi_12 * V_31 + VTVi_22 * V_32

        # Weights
        wt0 = (
            VTViVT_00 * POI_R0
            + VTViVT_01 * POI_R1
            + VTViVT_02 * POI_R2
            + VTViVT_03 * POI_R3
        ) / VTV_det
        wt1 = (
            VTViVT_10 * POI_R0
            + VTViVT_11 * POI_R1
            + VTViVT_12 * POI_R2
            + VTViVT_13 * POI_R3
        ) / VTV_det
        wt2 = (
            VTViVT_20 * POI_R0
            + VTViVT_21 * POI_R1
            + VTViVT_22 * POI_R2
            + VTViVT_23 * POI_R3
        ) / VTV_det
        wt3 = 1 - wt0 - wt1 - wt2

        wt0_bool = 1 if wt0 > weight_epsilon else 0
        wt1_bool = 1 if wt1 > weight_epsilon else 0
        wt2_bool = 1 if wt2 > weight_epsilon else 0
        wt3_bool = 1 if wt3 > weight_epsilon else 0

        wts_sum = wt0_bool * wt0 + wt1_bool * wt1 + wt2_bool * wt2 + wt3_bool * wt3

        closest_points[r, 0] = (
            wt0 * cell_tetrahedra[r][0, 0] * wt0_bool
            + wt1 * cell_tetrahedra[r][1, 0] * wt1_bool
            + wt2 * cell_tetrahedra[r][2, 0] * wt2_bool
            + wt3 * cell_tetrahedra[r][3, 0] * wt3_bool
        ) / wts_sum
        closest_points[r, 1] = (
            wt0 * cell_tetrahedra[r][0, 1] * wt0_bool
            + wt1 * cell_tetrahedra[r][1, 1] * wt1_bool
            + wt2 * cell_tetrahedra[r][2, 1] * wt2_bool
            + wt3 * cell_tetrahedra[r][3, 1] * wt3_bool
        ) / wts_sum
        closest_points[r, 2] = (
            wt0 * cell_tetrahedra[r][0, 2] * wt0_bool
            + wt1 * cell_tetrahedra[r][1, 2] * wt1_bool
            + wt2 * cell_tetrahedra[r][2, 2] * wt2_bool
            + wt3 * cell_tetrahedra[r][3, 2] * wt3_bool
        ) / wts_sum
        closest_points[r, 3] = (
            wt0 * cell_tetrahedra[r][0, 3] * wt0_bool
            + wt1 * cell_tetrahedra[r][1, 3] * wt1_bool
            + wt2 * cell_tetrahedra[r][2, 3] * wt2_bool
            + wt3 * cell_tetrahedra[r][3, 3] * wt3_bool
        ) / wts_sum

    return


##################################################################
#
# Upon import, run a small 3x3x3 array through calculate_meshlambda to compile it immediately.
#
##################################################################

small_array = np.array(
    [
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    ]
)


calculate_meshlambda(small_array, small_array, import_call=True)

##################################################################

##################################################################


@njit
def getTetraDist(reference_point, cell_tetrahedra):
    """
    Calculate best distance to all 5 tetrahedral simplices at once.

    Parameters:
    ----------
    reference_point: numpy array of floats
        1xN array of points, with N being the number of coordinates.

    cell_tetrahedra: numpy array of floats
        5x4xN array of tetrahedra points making up each cell, with N being the number of coordinates.

    """
    # Retrieve N from the size of the reference point.
    N = reference_point.size

    # Simulatenously get difference between reference point and all of the designated tetrahedra v4's.
    V_R = reference_point - cell_tetrahedra[:, 3]

    # 5 tetrahedra with 4 weights each.
    weights_matrix = np.zeros((5, 4))
    V = np.zeros((N, 3))
    V_T = np.zeros((3, N))
    V_T_V = np.zeros((3, 3))
    V_T_V_i = np.zeros((3, 3))
    V_T_V_i_V_T = np.zeros((3, N))
    weights = np.zeros(4)

    # Get weights. Iterate through the 5 tetrahedra and assign weight.
    for r in range(5):
        V = assign_V_matrix(cell_tetrahedra[r], V)
        V_T = assign_V_T_matrix(V, V_T)

        V_T_V = assign_V_T_V_matrix(V_T, V, V_T_V)
        V_T_V_i = assign_V_T_V_i_matrix(V_T_V, V_T_V_i)
        V_T_V_i_V_T = assign_V_T_V_i_V_T_matrix(V_T_V_i, V_T, V_T_V_i_V_T)
        weights = assign_weights(V_T_V_i_V_T, V_R[r], weights)
        weights_matrix[r] = weights

    return weights_matrix


@njit
def assign_V_matrix(tetrahedra, V):

    V[0, 0] = tetrahedra[0, 0] - tetrahedra[3, 0]  # dV00
    V[0, 1] = tetrahedra[1, 0] - tetrahedra[3, 0]  # dV01
    V[0, 2] = tetrahedra[2, 0] - tetrahedra[3, 0]  # dV02

    V[1, 0] = tetrahedra[0, 1] - tetrahedra[3, 1]  # dV10
    V[1, 1] = tetrahedra[1, 1] - tetrahedra[3, 1]  # dV11
    V[1, 2] = tetrahedra[2, 1] - tetrahedra[3, 1]  # dV12

    V[2, 0] = tetrahedra[0, 2] - tetrahedra[3, 2]  # dV20
    V[2, 1] = tetrahedra[1, 2] - tetrahedra[3, 2]  # dV21
    V[2, 2] = tetrahedra[2, 2] - tetrahedra[3, 2]  # dV22

    V[3, 0] = tetrahedra[0, 3] - tetrahedra[3, 3]  # dV30
    V[3, 1] = tetrahedra[1, 3] - tetrahedra[3, 3]  # dV31
    V[3, 2] = tetrahedra[2, 3] - tetrahedra[3, 3]  # dV32

    return V


@njit
def assign_V_T_matrix(V, V_T):

    V_T[0, 0] = V[0, 0]
    V_T[0, 1] = V[1, 0]
    V_T[0, 2] = V[2, 0]
    V_T[0, 3] = V[3, 0]

    V_T[1, 0] = V[0, 1]
    V_T[1, 1] = V[1, 1]
    V_T[1, 2] = V[2, 1]
    V_T[1, 3] = V[3, 1]

    V_T[2, 0] = V[0, 2]
    V_T[2, 1] = V[1, 2]
    V_T[2, 2] = V[2, 2]
    V_T[2, 3] = V[3, 2]

    return V_T


@njit
def assign_V_T_V_matrix(V_T, V, V_T_V):
    """
    Specific to 4 coordinate matrices present in meshlambda

    """
    V_T_V[0, 0] = (
        V_T[0, 0] * V[0, 0]
        + V_T[0, 1] * V[1, 0]
        + V_T[0, 2] * V[2, 0]
        + V_T[0, 3] * V[3, 0]
    )
    V_T_V[0, 1] = (
        V_T[0, 0] * V[0, 1]
        + V_T[0, 1] * V[1, 1]
        + V_T[0, 2] * V[2, 1]
        + V_T[0, 3] * V[3, 1]
    )
    V_T_V[0, 2] = (
        V_T[0, 0] * V[0, 2]
        + V_T[0, 1] * V[1, 2]
        + V_T[0, 2] * V[2, 2]
        + V_T[0, 3] * V[3, 2]
    )

    V_T_V[1, 0] = (
        V_T[1, 0] * V[0, 0]
        + V_T[1, 1] * V[1, 0]
        + V_T[1, 2] * V[2, 0]
        + V_T[1, 3] * V[3, 0]
    )
    V_T_V[1, 1] = (
        V_T[1, 0] * V[0, 1]
        + V_T[1, 1] * V[1, 1]
        + V_T[1, 2] * V[2, 1]
        + V_T[1, 3] * V[3, 1]
    )
    V_T_V[1, 2] = (
        V_T[1, 0] * V[0, 2]
        + V_T[1, 1] * V[1, 2]
        + V_T[1, 2] * V[2, 2]
        + V_T[1, 3] * V[3, 2]
    )

    V_T_V[2, 0] = (
        V_T[2, 0] * V[0, 0]
        + V_T[2, 1] * V[1, 0]
        + V_T[2, 2] * V[2, 0]
        + V_T[2, 3] * V[3, 0]
    )
    V_T_V[2, 1] = (
        V_T[2, 0] * V[0, 1]
        + V_T[2, 1] * V[1, 1]
        + V_T[2, 2] * V[2, 1]
        + V_T[2, 3] * V[3, 1]
    )
    V_T_V[2, 2] = (
        V_T[2, 0] * V[0, 2]
        + V_T[2, 1] * V[1, 2]
        + V_T[2, 2] * V[2, 2]
        + V_T[2, 3] * V[3, 2]
    )

    return V_T_V


@njit
def assign_V_T_V_i_matrix(V_T_V, V_T_V_i):

    V_T_V_i[0, 0] = V_T_V[1, 1] * V_T_V[2, 2] - V_T_V[1, 2] * V_T_V[1, 2]
    V_T_V_i[0, 1] = V_T_V[0, 2] * V_T_V[1, 2] - V_T_V[2, 2] * V_T_V[0, 1]
    V_T_V_i[0, 2] = V_T_V[0, 1] * V_T_V[1, 2] - V_T_V[1, 1] * V_T_V[0, 2]

    V_T_V_i[1, 0] = V_T_V_i[0, 1]
    V_T_V_i[1, 1] = V_T_V[0, 0] * V_T_V[2, 2] - V_T_V[0, 2] * V_T_V[0, 2]
    V_T_V_i[1, 2] = V_T_V[0, 2] * V_T_V[0, 1] - V_T_V[1, 2] * V_T_V[0, 0]

    V_T_V_i[2, 0] = V_T_V_i[0, 2]
    V_T_V_i[2, 1] = V_T_V_i[1, 2]
    V_T_V_i[2, 2] = V_T_V[0, 0] * V_T_V[1, 1] - V_T_V[1, 2] * V_T_V[0, 0]

    return V_T_V_i


@njit
def assign_V_T_V_i_V_T_matrix(V_T_V_i, V_T, V_T_V_i_V_T):

    V_T_V_i_V_T[0, 0] = (
        V_T_V_i[0, 0] * V_T[0, 0]
        + V_T_V_i[0, 1] * V_T[1, 0]
        + V_T_V_i[0, 2] * V_T[2, 0]
    )
    V_T_V_i_V_T[0, 1] = (
        V_T_V_i[0, 0] * V_T[0, 1]
        + V_T_V_i[0, 1] * V_T[1, 1]
        + V_T_V_i[0, 2] * V_T[2, 1]
    )
    V_T_V_i_V_T[0, 2] = (
        V_T_V_i[0, 0] * V_T[0, 2]
        + V_T_V_i[0, 1] * V_T[1, 2]
        + V_T_V_i[0, 2] * V_T[2, 2]
    )
    V_T_V_i_V_T[0, 3] = (
        V_T_V_i[0, 0] * V_T[0, 3]
        + V_T_V_i[0, 1] * V_T[1, 3]
        + V_T_V_i[0, 2] * V_T[2, 3]
    )

    V_T_V_i_V_T[1, 0] = (
        V_T_V_i[1, 0] * V_T[0, 0]
        + V_T_V_i[1, 1] * V_T[1, 0]
        + V_T_V_i[1, 2] * V_T[2, 0]
    )
    V_T_V_i_V_T[1, 1] = (
        V_T_V_i[1, 0] * V_T[0, 1]
        + V_T_V_i[1, 1] * V_T[1, 1]
        + V_T_V_i[1, 2] * V_T[2, 1]
    )
    V_T_V_i_V_T[1, 2] = (
        V_T_V_i[1, 0] * V_T[0, 2]
        + V_T_V_i[1, 1] * V_T[1, 2]
        + V_T_V_i[1, 2] * V_T[2, 2]
    )
    V_T_V_i_V_T[1, 3] = (
        V_T_V_i[1, 0] * V_T[0, 3]
        + V_T_V_i[1, 1] * V_T[1, 3]
        + V_T_V_i[1, 2] * V_T[2, 3]
    )

    V_T_V_i_V_T[2, 0] = (
        V_T_V_i[2, 0] * V_T[0, 0]
        + V_T_V_i[2, 1] * V_T[1, 0]
        + V_T_V_i[2, 2] * V_T[2, 0]
    )
    V_T_V_i_V_T[2, 1] = (
        V_T_V_i[2, 0] * V_T[0, 1]
        + V_T_V_i[2, 1] * V_T[1, 1]
        + V_T_V_i[2, 2] * V_T[2, 1]
    )
    V_T_V_i_V_T[2, 2] = (
        V_T_V_i[2, 0] * V_T[0, 2]
        + V_T_V_i[2, 1] * V_T[1, 2]
        + V_T_V_i[2, 2] * V_T[2, 2]
    )
    V_T_V_i_V_T[2, 3] = (
        V_T_V_i[2, 0] * V_T[0, 3]
        + V_T_V_i[2, 1] * V_T[1, 3]
        + V_T_V_i[2, 2] * V_T[2, 3]
    )

    return V_T_V_i_V_T


@njit
def assign_weights(V_T_V_i_V_T, V_R, weights):

    weights[0] = (
        V_T_V_i_V_T[0, 0] * V_R[0]
        + V_T_V_i_V_T[0, 1] * V_R[1]
        + V_T_V_i_V_T[0, 2] * V_R[2]
        + V_T_V_i_V_T[0, 3] * V_R[3]
    )
    weights[1] = (
        V_T_V_i_V_T[1, 0] * V_R[0]
        + V_T_V_i_V_T[1, 1] * V_R[1]
        + V_T_V_i_V_T[1, 2] * V_R[2]
        + V_T_V_i_V_T[1, 3] * V_R[3]
    )
    weights[2] = (
        V_T_V_i_V_T[2, 0] * V_R[0]
        + V_T_V_i_V_T[2, 1] * V_R[1]
        + V_T_V_i_V_T[2, 2] * V_R[2]
        + V_T_V_i_V_T[2, 3] * V_R[3]
    )
    weights[3] = (
        (
            np.sign(weights[0]) * weights[0]
            + np.sign(weights[1]) * weights[1]
            + np.sign(weights[2]) * weights[2]
        )
        - weights[0]
        - weights[1]
        - weights[2]
    )

    return weights


#     for x_ind in range(x_size):
#         for y_ind in range(y_size):
#             for z_ind in range(z_size):

#                 # Assign negative values to meshlambda if mask is false.
#                 if not reference_image_mask[x_ind, y_ind, z_ind]:
#                     meshlambda_vector[x_ind, y_ind, z_ind, :] = np.ones(5)*-1000

#                 else:
#                     # Border locations are those less than the margin away from the edge in any direction.
#                     border = False
#                     if (x_ind < margin) or (x_ind >= x_size - margin) or (y_ind < margin) or (y_ind >= y_size - margin) or (z_ind < margin) or (z_ind >= z_size - margin):
#                         border = True

#                     lambda_vector = np.ones(5)*max_lambda_vector

#                     x_scaled_loc_ref = x_ind / dhu
#                     y_scaled_loc_ref = y_ind / dhu
#                     z_scaled_loc_ref = z_ind / dhu

#                     reference_value = reference_image[x_ind, y_ind, z_ind] / dhu
#                     comparison_value = comparison_image[x_ind, y_ind, z_ind] / dhu

#                     # Enforce minimum difference threshold. This also ensures that lambda is >= min_difference.
#                     if (reference_value - comparison_value)*(reference_value - comparison_value) < min_difference*min_difference:
#                         lambda_vector = np.zeros(5)

#                     else:
#                         reference_vector = np.array([x_scaled_loc_ref, y_scaled_loc_ref, z_scaled_loc_ref, reference_value])

#                         for cell_num in range(n_cells):
#                             far_x = x_ind + vertex_shifts[cell_num, 6, 0] # Index 6 corresponds to the furthest vertex of any cell.
#                             far_y = y_ind + vertex_shifts[cell_num, 6, 1]
#                             far_z = z_ind + vertex_shifts[cell_num, 6, 2]

#                             # Want to skip cells that are outside the array.
#                             if border and ((far_x < 0) or (far_x > x_size - 1) or (far_y < 0) or (far_y > y_size - 1) or (far_z < 0) or (far_z > z_size - 1)):
#                                 continue

#                             # Break if the closest vertex of the cell is larger in magnitude than the current lambda value
#                             # as every cell now has lambda values larger than the current one.
#                             # We can do this because the cells are presorted by distance.
#                             closest_vertex_distance = scaled_vertex_shifts[cell_num, 0, 3]
#                             if closest_vertex_distance >= lambda_vector[0]:
#                                 break

#                             # Collect tetrahedra points.
#                             for tetra_index in range(5):
#                                 for tetra_vertex_num, vertex_index in enumerate(_TETRA_POINTS[tetra_index]):
#                                     # Make these calculations relative, add the reference vector back at the end.
#                                     x_scaled_loc_comp = scaled_vertex_shifts[cell_num, vertex_index, 0] + x_scaled_loc_ref
#                                     y_scaled_loc_comp = scaled_vertex_shifts[cell_num, vertex_index, 1] + y_scaled_loc_ref
#                                     z_scaled_loc_comp = scaled_vertex_shifts[cell_num, vertex_index, 2] + z_scaled_loc_ref

#                                     shift_x_ind_comp = int(x_ind + vertex_shifts[cell_num, vertex_index, 0])
#                                     shift_y_ind_comp = int(y_ind + vertex_shifts[cell_num, vertex_index, 1])
#                                     shift_z_ind_comp = int(z_ind + vertex_shifts[cell_num, vertex_index, 2])

#                                     cell_tetrahedra[tetra_index, tetra_vertex_num, 0] = x_scaled_loc_comp
#                                     cell_tetrahedra[tetra_index, tetra_vertex_num, 1] = y_scaled_loc_comp
#                                     cell_tetrahedra[tetra_index, tetra_vertex_num, 2] = z_scaled_loc_comp
#                                     cell_tetrahedra[tetra_index, tetra_vertex_num, 3] = scaled_comparison_image[shift_x_ind_comp, shift_y_ind_comp, shift_z_ind_comp]

#                             # Calculate lambda, determine closest points.
#                             get_tetra_distances_obscene(reference_vector, cell_tetrahedra, closest_points) # 5x4
#                             closest_points_differences = closest_points - reference_vector # Ordered this way so that reference + result = point.
#                             closest_distances = np.sqrt(np.sum(closest_points_differences*closest_points_differences, axis = 1))
#                             closest_index = np.argmin(closest_distances)

#                             if lambda_vector[0] > closest_distances[closest_index]:
#                                 lambda_vector[0] = closest_distances[closest_index]
#                                 lambda_vector[1:] = closest_points_differences[closest_index]

#                     meshlambda_vector[x_ind, y_ind, z_ind, :] = lambda_vector[:]

#     return
