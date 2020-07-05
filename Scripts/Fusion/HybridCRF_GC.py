import numpy as np
import maxflow


def weight(i, j, k, m, prob_FF, prob_NN, prob_DD, imageRGB, mean):
    #print(alpha)
    alpha = 1 - (prob_FF[i][j] - prob_FF[k][m]) ** 2
    if alpha < 0.01:
        alpha = 0.01
    weight1 = alpha * np.exp(-((imageRGB[i][j] - imageRGB[k][m]) ** 2) / (2 * mean))
    #print(weight1)
    weight1 += alpha * np.exp(-(prob_NN[i][j] - prob_NN[k][m]) ** 2)  # try to include and vary alpha
    weight1 += alpha * np.exp(-(prob_DD[i][j] - prob_DD[k][m]) ** 2)

    #print(weight1)

    return (weight1*2)/3


def HybridGC(image, prob_NN, prob_DD, iterations):

    imageR = image[:, :, 0]
    imageG = image[:, :, 1]
    imageB = image[:, :, 2]
    imageRGB = (imageR + imageG + imageB)/3
    #list = []


    prob_FF = (prob_NN + prob_DD) / 2
    rows, cols = prob_NN.shape
    count = 0
    mean = 0
    flatten = imageRGB.flatten()
    flatten = np.random.choice(flatten, 1000)

    for i in range(len(flatten) - 1):
        for j in range(i + 1, len(flatten)):

            mean = (((flatten[i] - flatten[j]) ** 2) + mean * count)/(count + 1)
            count += 1

    for i in range(1):

        g = maxflow.Graph[float](rows, cols)
        nodeids = g.add_grid_nodes(prob_NN.shape)

        for i in range(rows):
            for j in range(cols):
                val = cols * i

                if j < cols - 1 and i < rows - 1:
                    weight1 = weight(i, j, i, j + 1, prob_FF, prob_NN, prob_DD, imageRGB, mean)
                    weight2 = weight(i, j, i + 1, j, prob_FF, prob_NN, prob_DD, imageRGB, mean)
                    g.add_edge(val + j, val + j + 1, weight1, weight1)

                    g.add_edge(val + j, val + j + cols, weight2, weight2)
                    #list.append(weight1)
                    #list.append(weight2)
                    # print(weight1,weight2)

                elif j == cols - 1 and i < rows - 1:
                    weight2 = weight(i, j, i + 1, j, prob_FF, prob_NN, prob_DD, imageRGB, mean)
                    g.add_edge(val + j, val + j + cols, weight2, weight2)

                elif j < cols - 1 and i == rows - 1:
                    weight1 = weight(i, j, i, j + 1, prob_FF, prob_NN, prob_DD, imageRGB, mean)
                    g.add_edge(val + j, val + j + 1, weight1, weight1)

        t_edge_weights_source = np.exp(-(1 - prob_NN)) + np.exp(-(1 - prob_DD))  # source = 1 and sink = 0
        t_edge_weights_sink = np.exp(-prob_NN) + np.exp(-prob_DD)
        g.add_grid_tedges(nodeids, t_edge_weights_source, t_edge_weights_sink)

        # Find the maximum flow.
        g.maxflow()

        # Get the segments of the nodes in the grid.
        sgm = g.get_grid_segments(nodeids)
        sgm = np.int_(np.logical_not(sgm))

        t_edge_weights_sum = t_edge_weights_sink + t_edge_weights_source

        #for i in range(rows):
         #   for j in range(cols):
          #      if sgm[i][j] == 1:
           #         prob_FF[i][j] = (np.maximum(prob_NN[i][j], prob_DD[i][j]) + prob_FF[i][j])/2


            #    else:
             #       prob_FF[i][j] = (np.minimum(prob_NN[i][j], prob_DD[i][j]) + prob_FF[i][j])/2

    #print(np.unique(t_edge_weights_source))
    #print(np.unique(t_edge_weights_sink))
    #print(min(list))
    #print(max(list))




    return sgm








