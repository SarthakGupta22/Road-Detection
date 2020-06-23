# Python code to convert disparity map to depth map
import numpy as np
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import pcl
import cv2
import pcl.pcl_visualization
viewer = pcl.pcl_visualization.PCLVisualizering()


def do_ransac_plane_segmentation(point_cloud, max_distance):

  segmenter = point_cloud.make_segmenter()

  segmenter.set_model_type(pcl.SACMODEL_PLANE)
  segmenter.set_method_type(pcl.SAC_RANSAC)
  segmenter.set_distance_threshold(max_distance)

  #obtain inlier indices and model coefficients
  inlier_indices, coefficients = segmenter.segment()

  inliers = point_cloud.extract(inlier_indices, negative = False)
  outliers = point_cloud.extract(inlier_indices, negative = True)
  print(coefficients)

  return inliers, coefficients


def getRoad(disp, img):

    point_cloud = [] 

#generate Point cloud
    for i in range(0, disp.shape[0]):
        for j in range(0, disp.shape[1]):
            point = []

            z = float(0.54*721.0/(disp[i][j] + 0.1))
            
            if z < 100.0:
                x = float((j - disp.shape[1]/2.0)*z/721.0)
                y = float((i - disp.shape[0]/2.0)*z/721.0)
                if y > 0.0:
                    point = [x,y,z]
                    point_cloud.append(point)

    pc = np.asanyarray(point_cloud).reshape(-1,3).astype(np.float32)

# Convert to PCL point cloud
    pclPointCloud = pcl.PointCloud()
    pclPointCloud.from_array(pc)

#segment plane
    pcl_pc, coeffs = do_ransac_plane_segmentation(pclPointCloud, 0.07)

    pc = np.asanyarray(pcl_pc).reshape(-1,3).astype(np.float32)

#project the points on this image
    for i in range(pc.shape[0]):
        x = int((pc[i][0]*721.0/pc[i][2]) + disp.shape[1]/2.0)
        y = int((pc[i][1]*721.0/pc[i][2]) + disp.shape[0]/2.0)

        if x < disp.shape[1] and y < disp.shape[0]:
            img = cv2.circle(img, (x,y), 10, 0, -1)

    cv2.imshow("road masked", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    while 1:
        viewer.AddPointCloud(pcl_pc, b'scene_cloud', 0)
        viewer.SpinOnce()
        viewer.RemovePointCloud(b'scene_cloud', 0)
    
if __name__=="__main__":
    # Read the disparity map
    disp = cv2.imread("D_uu_000097.png",0) #grayscale image
    img =  img = cv2.imread("uu_000097.png",0) #left image
    getRoad(disp,img)