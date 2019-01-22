import numpy as np
from synthetics_utils import shuffle_matrices


def generate_uniform_circle_data(n, center, radius):
    length = np.sqrt(np.random.uniform(0, 1, n)) * radius
    angle = np.pi * np.random.uniform(
        0, 2, n
    )  # cover full range from 0 to 2pi
    x = length * np.cos(angle) + center[0]
    y = length * np.sin(angle) + center[1]
    return np.vstack((x, y)).T

def get_circle_idx(X, center, radius):
    h, k = center
    idx = np.sqrt((X[:, 0] - h) ** 2 + (X[:, 1] - k) ** 2) < radius
    return idx

def radius_for_lf_metric(target_val, slice_radius, metric=None):
    assert isinstance(slice_radius, int) or isinstance(slice_radius, float)

    if metric == 'recall':
        return np.sqrt(target_val * (slice_radius**2))
    elif metric == 'precision':
        return np.sqrt((slice_radius**2) / target_val)
    else: 
        return slice_radius

def generate_deathstar_data(config, x_var=None, x_val=None, verbose=False, return_overlap=False):        
    centers = config['mus']
    radii = config['variances']
    class_props = config['props']
    N = config['N']
    labels = config['labels']
    lf_metrics = config['lf_metrics']    

    # overwrite recall value for head lf
    if x_var == 'slice_proportion': # slice proportion
        class_props[0] = x_val
        class_props[1] = 1 - x_val
        
    elif x_var == 'head_recall':
        target_metric = 'recall'
        lf_metrics[0] = ('recall', x_val)        
        
    # override head precision value
#     head_precision_override = None
    if x_var == 'head_precision':
        target_metric = 'precision'
#         head_precision_override = x_val
        lf_metrics[0] = ('precision', x_val)
    
    # Set slice 1
    n_per_slice = [int(N*prop) for prop in class_props]
    slice_1 = generate_uniform_circle_data(
        n_per_slice[1], 
        centers[1], 
        radii[1]
    )

    # get idx for slice 0 within slice 1
    slice_0_idx = get_circle_idx(slice_1, tuple(centers[0]), radii[0])
    
    # remove slice 0 idx from slice 1
    # NOTE: because we remove the slice, if the variances are not 
    # proportional to the slice size, the original props will not be maintained
    slice_1 = slice_1[np.logical_not(slice_0_idx)]
    n_per_slice[1] = len(slice_1)

    # Set slice 0
    slice_0 = generate_uniform_circle_data(
        n_per_slice[0], 
        centers[0], 
        radii[0]
    )

    # combine slices
    Xu = [slice_0, slice_1] # data points
    Yu = [label * np.ones(n) for n, label in zip(n_per_slice, labels)] # class labels
    Cu = [i * np.ones(n) for i, n in enumerate(n_per_slice)] # slice labels

    X, Y, C = shuffle_matrices([np.vstack(Xu), np.hstack(Yu), np.hstack(Cu)])
    
    # generate label matrix
    L = np.zeros((sum(n_per_slice), 2))

    # set LF0 to target slice 0
    lf0_target_metric, lf0_target_value = lf_metrics[0]
    lf_0_idx = get_circle_idx(
        X, 
        tuple(centers[0]), 
        radius_for_lf_metric(lf0_target_value, radii[0], metric=lf0_target_metric)
    )
    
#     if head_precision_override:
#         lf_0_idx = lf_circ_idx_for_slice_precision(
#             x_val, X, C==0, tuple(centers[0]), radii[0], verbose=True
#         )
    
    L[lf_0_idx, 0] = labels[0]
    
    # set LF1 to target slice 1
    lf1_target_metric, lf1_target_value = lf_metrics[1]
    lf_1_idx = get_circle_idx(
        X, 
        tuple(centers[1]), 
        radius_for_lf_metric(lf1_target_value, radii[1], metric=lf1_target_metric)
    )
    
    L[lf_1_idx, 1] = labels[1]

    overlap_idx = np.logical_and(lf_0_idx, lf_1_idx)
    
    if return_overlap:
        return X, Y, C, L, overlap_idx
    
    return X, Y, C, L

