# Python code to convert disparity map to depth map
import numpy as np
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import pcl
import cv2
import pcl.pcl_visualization
viewer = pcl.pcl_visualization.PCLVisualizering()

# Read the disparity map
disp = cv2.imread("D_um_000076.png",0) #grayscale image
point_cloud = [] #list
print(disp.shape[0])


for i in range(0, disp.shape[0]):
    for j in range(0, disp.shape[1]):
        point = []
        z = float(0.54*721.0/(disp[i][j] + 0.001))
        if z < 100.0:
                x = float((j - disp.shape[1]/2.0)*z/721.0)
                y = float((i - disp.shape[0]/2.0)*z/721.0)
                if y > 0.5:
                    point = [x,y,z]
                    point_cloud.append(point)

pc = np.asanyarray(point_cloud).reshape(-1,3).astype(np.float32)
print(pc)

# Display point cloud in python pcl
pclPointCloud = pcl.PointCloud()
pclPointCloud.from_array(pc)


# Read the RGB image and project these points on the image and detect the road plane
img = cv2.imread("um_000076_Left.png")

#project the points on this image
for i in range(pc.shape[0]):
    x = int((pc[i][0]*721.0/pc[i][2]) + disp.shape[1]/2.0)
    y = int((pc[i][1]*721.0/pc[i][2]) + disp.shape[0]/2.0)
    
    if x < disp.shape[1] and y < disp.shape[0]:
        print([x,y])
        img = cv2.circle(img, (x,y), 5, 0, -1)
        #img[y,x] = 0

cv2.imshow("road masked", img)

cv2.waitKey(0)
cv2.destroyAllWindows()


while 1:
    viewer.AddPointCloud(pclPointCloud, b'scene_cloud', 0)
    viewer.SpinOnce()
    viewer.RemovePointCloud(b'scene_cloud', 0)
    