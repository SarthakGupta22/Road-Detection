import cv2
import numpy as np
import matplotlib.pyplot as plt


def find_disparity(left_image, right_image, window_size=15, wls=False, lmbda=8000, sigma=1.2):

    sgbm_left = cv2.StereoSGBM_create(
        minDisparity=-16,
        numDisparities=128,
        blockSize=5,
        P1=8 * 3 * window_size * window_size,
        P2=32 * 3 * window_size * window_size,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    if not wls:
        disparity = sgbm_left.compute(left_image, right_image)
        cv2.normalize(src=disparity, dst=disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
        disparity = np.uint8(disparity)
        return disparity

    sgbm_right = cv2.ximgproc.createRightMatcher(sgbm_left)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=sgbm_left)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    disp_left = sgbm_left.compute(left_image, right_image)
    disp_right = sgbm_right.compute(right_image, left_image)
    disp_left = np.int16(disp_left)
    disp_right = np.int16(disp_right)
    filtered_disp = wls_filter.filter(disp_left, left_image, None, disp_right)
    cv2.normalize(src=filtered_disp, dst=filtered_disp, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filtered_disp = np.uint8(filtered_disp)

    return filtered_disp


class Vdisparity:

    def __init__(self, left_image, right_image, nn_mask=None):
        self.left_image = left_image
        self.right_image = right_image
        self.v_disp_map = np.zeros((left_image.shape[0], 256), dtype=np.uint8)
        self.v_disp_map_with_nn = np.zeros((left_image.shape[0], 256), dtype=np.uint8)
        self.nn_mask = nn_mask
        self.line_mask = None
        self.final_mask = None
        self.disparity = None

    def compute(self, disparity):
        self.disparity = disparity
        compNN = True
        if self.nn_mask is None:
            compNN = False

        print("Row : ", disparity.shape[0])
        print("Col : ", disparity.shape[1])
        for row in range(disparity.shape[0]):
            for col in range(disparity.shape[1]):
                if disparity[row, col] != 0:
                    self.v_disp_map[row, disparity[row, col]] += 1
                    if compNN:
                        if self.nn_mask[row, col] != 0:
                            self.v_disp_map_with_nn[row, disparity[row, col]] += 1

    def plot_vdisp_line(self):
        plt.figure()
        plt.imshow(self.v_disp_map)
        plt.title('Original V-disparity Map')

        if self.nn_mask is not None:
            plt.figure()
            plt.imshow(self.v_disp_map_with_nn)
            plt.title('V-disparity Map with NN prior')

        if self.line_mask is not None:
            plt.figure()
            plt.imshow(self.line_mask)
            plt.title('Lines fit with Hough Transform')

        plt.show()

    def fit_line(self, pts_of_intersection=180):
        self.line_mask = np.zeros(self.v_disp_map.shape, dtype=np.uint8)
        print(self.v_disp_map_with_nn.shape)
        if self.nn_mask is None:
            lines = cv2.HoughLines(self.v_disp_map, 1, np.pi/180, 180)
        else:
            lines = cv2.HoughLines(self.v_disp_map_with_nn, 1, np.pi / 180, pts_of_intersection)

        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * -b)
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * -b)
            y2 = int(y0 - 1000 * a)

            cv2.line(self.line_mask, (x1, y1), (x2, y2), 255, 1, cv2.LINE_AA)

    def show_road_mask(self):
        self.final_mask = np.zeros(self.left_image.shape, dtype=np.uint8)

        for row in range(self.line_mask.shape[0]):
            for col in range(self.line_mask.shape[1]):
                if self.line_mask[row, col] != 0:
                    intensity = col
                    for itr in range(self.disparity.shape[1]):
                        if abs(self.disparity[row, itr] - intensity) <= 6:
                            self.final_mask[row, itr] = 255

        plt.imshow(self.left_image)
        plt.imshow(self.final_mask, alpha=0.4)
        plt.show()
