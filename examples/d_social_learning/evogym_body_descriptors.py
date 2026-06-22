import numpy as np
from scipy.ndimage import shift


def relative_activity(body: np.ndarray):
    return np.count_nonzero(body > 2) / np.count_nonzero(body > 0)

def size(body: np.ndarray):
    return np.count_nonzero(body > 0)

def compactness(body: np.ndarray) -> float:
    convex_hull = body > 0
    if True not in convex_hull:
        return 0.0
    new_found = True
    while new_found:
        new_found = False
        false_coordinates = np.argwhere(convex_hull == False)
        for coordinate in false_coordinates:
            x, y = coordinate[0], coordinate[1]
            adjacent_count = 0
            adjacent_coordinates = []
            for d in [-1, 1]:
                adjacent_coordinates.append((x, y + d))
                adjacent_coordinates.append((x + d, y))
                adjacent_coordinates.append((x + d, y + d))
                adjacent_coordinates.append((x + d, y - d))
            for adj_x, adj_y in adjacent_coordinates:
                if 0 <= adj_x < body.shape[0] and 0 <= adj_y < body.shape[1] and convex_hull[adj_x][adj_y]:
                    adjacent_count += 1
            if adjacent_count >= 5:
                convex_hull[x][y] = True
                new_found = True

    return (body > 0).sum() / convex_hull.sum()


def elongation(body: np.ndarray) -> float:
    """
    Computes elongation of a 2D voxel robot using
    the ellipse of equal second moments.

    Returns a value in [0, 1).
    """
    body = body.astype(bool)
    coords = np.argwhere(body)

    # Fewer than 2 voxels → no elongation
    if len(coords) < 2:
        return 0.0

    # Center the coordinates
    coords = coords.astype(float)
    centroid = coords.mean(axis=0)
    centered = coords - centroid

    # Covariance matrix (2x2)
    cov = np.cov(centered, rowvar=False)

    # Eigenvalues (sorted)
    eigvals = np.linalg.eigvalsh(cov)
    lambda_min, lambda_max = eigvals

    if lambda_max <= 0:
        return 0.0

    # Semi-axis ratio
    b_over_a = np.sqrt(lambda_min / lambda_max)

    # Elongation definition
    return np.sqrt(1.0 - b_over_a ** 2)


def symmetry(body: np.ndarray, axis: str = 'horizontal') -> float:
    coords = np.argwhere(body > 0)
    if len(coords) == 0:
        return 0.0

    r_min, c_min = coords.min(axis=0)
    r_max, c_max = coords.max(axis=0)
    cropped = body[r_min:r_max + 1, c_min:c_max + 1]

    if axis == 'horizontal':
        flipped = np.fliplr(cropped)
    elif axis == 'vertical':
        flipped = np.flipud(cropped)
    else:
        raise ValueError("axis must be 'horizontal' or 'vertical'")

    return float(np.mean(cropped == flipped))

def aligned_hamming_distance(body_a: np.ndarray, body_b: np.ndarray):
    if body_a.shape != body_b.shape or body_a.shape[0] != body_a.shape[1]:
        raise ValueError(
            f"Expected equal square arrays, got {body_a.shape} and {body_b.shape}"
        )

    gl = body_a.shape[0]

    A_non_zero = np.count_nonzero(body_a)
    B_non_zero = np.count_nonzero(body_b)

    min_dist = np.inf
    shifts = range(-gl + 1, gl)

    for dx_a in shifts:
        A_shifted = shift(body_a, shift=(dx_a, 0), order=0, cval=0)
        for dy_a in shifts:
            A_final = shift(A_shifted, shift=(0, dy_a), order=0, cval=0)

            A_nz = np.count_nonzero(A_final)
            if A_nz != A_non_zero:
                continue

            for dx_b in shifts:
                B_shifted = shift(body_b, shift=(dx_b, 0), order=0, cval=0)
                for dy_b in shifts:
                    B_final = shift(B_shifted, shift=(0, dy_b), order=0, cval=0)

                    B_nz = np.count_nonzero(B_final)
                    if B_nz != B_non_zero:
                        continue

                    dist = np.count_nonzero(A_final != B_final)
                    min_dist = min(min_dist, dist)

    return min_dist