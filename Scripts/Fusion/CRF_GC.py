import numpy as np
import maxflow


def GC(image, prob_NN, prob_DD):

    imageR = image[:, :, 0]/255
    imageG = image[:, :, 1]/255
    imageB = image[:, :, 2]/255

    rows, cols = prob_NN.shape
    g = maxflow.Graph[float](rows, cols)
    nodeids = g.add_grid_nodes(prob_NN.shape)

    for i in range(rows):
        for j in range(cols):
            val = cols * i

            if j< cols - 1 and i< rows -1:
                weight1 = 1 - (abs(imageR[i][j] - imageR[i][j + 1]) +
                               abs(imageG[i][j] - imageG[i][j + 1]) +
                               abs(imageB[i][j] - imageB[i][j + 1])) / 3
                weight2 = 1 - (abs(imageR[i][j] - imageR[i + 1][j]) +
                               abs(imageG[i][j] - imageG[i + 1][j]) +
                               abs(imageB[i][j] - imageB[i + 1][j])) / 3

                g.add_edge(val + j, val + j + 1, weight1, weight1)

                g.add_edge(val + j, val + j + cols, weight2, weight2)

            elif j == cols - 1 and i < rows - 1:
                weight2 = 1 - (abs(imageR[i][j] - imageR[i + 1][j]) +
                               abs(imageG[i][j] - imageG[i + 1][j]) +
                               abs(imageB[i][j] - imageB[i + 1][j])) / 3
                g.add_edge(val + j, val + j + cols, weight2, weight2)

            elif j < cols - 1 and i == rows - 1:
                weight1 = 1 - (abs(imageR[i][j] - imageR[i][j + 1]) +
                               abs(imageG[i][j] - imageG[i][j + 1]) +
                               abs(imageB[i][j] - imageB[i][j + 1])) / 3
                g.add_edge(val + j, val + j + 1, weight1, weight1)

    t_edge_weights = (prob_NN + prob_DD)/2
    g.add_grid_tedges(nodeids, t_edge_weights,1 - t_edge_weights)

    # Find the maximum flow.
    g.maxflow()

    # Get the segments of the nodes in the grid.
    sgm = g.get_grid_segments(nodeids)
    sgm = np.int_(np.logical_not(sgm))
    return sgm








