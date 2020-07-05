import numpy as np
from numba import njit


@njit
def get_distances(arr, P):
    distances = np.empty(arr.shape)
    for r in range(distances.shape[0]):
        for c in range(distances.shape[1]):
            distances[r, c] = np.sqrt(np.power(r - P[0], 2) + np.power(c - P[1], 2))

    return distances


@njit
def ultimate_fusion(new_image, prob_nn, prob_dd, iterations, radius):
    rows, cols = new_image.shape[:2]
    prob_ff = np.zeros((rows+2, cols+2))
    
    prob_ff[1:rows+1, 1:cols+1] = (prob_nn + prob_dd) / 2
    # Find last row
    last_row = rows - 1
    while not np.any(prob_nn[last_row]):
        last_row -= 1
    
    M = np.sum(np.arange(0, cols) * prob_nn[last_row]) / np.sum(prob_nn[last_row])
    baseline = np.array([last_row, int(M)])
    
    distances = get_distances(prob_nn, baseline)
    decay = 1 - distances / cols
    
    w_nn = 1
    w_dd = 1
    
    for _ in range(iterations):
        for r in range(1, rows):
            for c in range(1, cols):
                # These are image coordinates
                i, j = r-1, c-1
                sum_of_weights = w_nn
                prod = w_nn * prob_nn[i, j]
                
                if distances[i, j] < round(cols / 4):
                    sum_of_weights += w_dd * decay[i, j]
                    prod += w_dd * prob_dd[i, j] * decay[i, j]
                else:
                    sum_of_weights += w_dd * 0.75
                    prod += w_dd * prob_dd[i, j] * 0.75
                
                # North
                if i > 0:
                    w_n = 1 - np.sum(np.abs(new_image[i, j, :] - new_image[i-1, j, :]) / 3)
                else:
                    w_n = 0
                
                sum_of_weights += w_n
                prod += prob_ff[r-1, c] * w_n

                # West
                if j < cols:
                    w_w = 1 - np.sum(np.abs(new_image[i, j, :] - new_image[i, j+1, :]) / 3)
                else:
                    w_w = 0
                
                sum_of_weights += w_w
                prod += prob_ff[r, c+1] * w_w
                
                # South
                if i < rows:
                    w_s = 1 - np.sum(np.abs(new_image[i, j, :] - new_image[i+1, j, :]) / 3)
                else:
                    w_s = 0
                
                sum_of_weights += w_s
                prod += prob_ff[r+1, c] * w_s

                # East
                if j > 0:
                    w_e = 1 - np.sum(np.abs(new_image[i, j, :] - new_image[i, j-1, :]) / 3)
                else:
                    w_e = 0
                
                sum_of_weights += w_e
                prod += prob_ff[r, c+1] * w_e
                
                # North East
                if i > 0 and j > 0:
                    w_ne = (1 - np.sum(np.abs(new_image[i, j, :] - new_image[i-1, j-1, :]) / 3))/np.sqrt(2)
                else:
                    w_ne = 0
                
                sum_of_weights += w_ne
                prod += prob_ff[r-1, c-1] * w_ne

                # North West
                if i > 0 and j < cols:
                    w_nw = (1 - np.sum(np.abs(new_image[i, j, :] - new_image[i - 1, j + 1, :]) / 3))/np.sqrt(2)
                else:
                    w_nw = 0

                sum_of_weights += w_nw
                prod += prob_ff[r - 1, c + 1] * w_nw

                # South West
                if i < rows and j < cols:
                    w_sw = (1 - np.sum(np.abs(new_image[i, j, :] - new_image[i + 1, j + 1, :]) / 3))/np.sqrt(2)
                else:
                    w_sw = 0

                sum_of_weights += w_sw
                prod += prob_ff[r + 1, c + 1] * w_sw

                # South East
                if i < rows and j > 0:
                    w_se = (1 - np.sum(np.abs(new_image[i, j, :] - new_image[i + 1, j - 1, :]) / 3))/np.sqrt(2)
                else:
                    w_se = 0

                sum_of_weights += w_se
                prod += prob_ff[r + 1, c - 1] * w_se

                prod /= sum_of_weights
                
                prob_ff[r, c] = prod
                
    
    return prob_ff






