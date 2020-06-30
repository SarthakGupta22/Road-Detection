# Generate training data 
# 2 and 3 layer models
# from command line -> take the path of the point cloud
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import os
import glob
import open3d as o3d
import numpy as np
import readCalib 
import matplotlib.pyplot as plt

pathPC = "/home/deepak/Downloads/data_road_velodyne/training/velodyne"
path2D = "/home/deepak/Downloads/data_road_velodyne/data/2D_layer"
path3D = "/home/deepak/Downloads/data_road_velodyne/data/3D_layer"
pathImg = "/home/deepak/Downloads/data_road/training/image_2"

pathCalib = "/home/deepak/Downloads/data_road/training/calib"

print(pathPC)

for pcd_name in os.listdir(pathPC):        # Read point clouds
    if ".pcd" in pcd_name:
        print(pcd_name)
        pc = o3d.io.read_point_cloud(os.path.join(pathPC,pcd_name))  # Read the point cloud

        #Read the corresponding calibration parameters
        filename = pcd_name.replace(".pcd",".txt")
        T,K,R = readCalib.getMatrices(os.path.join(pathCalib, filename))

        image_name = filename.replace(".txt",".png")
        print(image_name)
        img = cv2.imread(os.path.join(pathImg,image_name))

        #Convert point cloud to numpy array and project on the image
        pc_np = np.asanyarray(pc.points).reshape(-1,3)
        pc_np_h = np.ones([pc_np.shape[0],4])
        pc_np_h[:,0:3] = pc_np 

        projPoints = np.zeros([pc_np_h.shape[0],3], dtype=np.float32)

        #project the points in camera frame  (3D)
        for i in range(pc_np.shape[0]):
            projPoints[i] = np.dot(R,np.dot(T, pc_np_h[i]))
            
        #points projected in camera frame of reference
        #now convert to pixel coordinates
        camPixCoordsH = np.zeros([pc_np.shape[0],3], dtype=int)
        camPixCoords = np.zeros([pc_np.shape[0],2], dtype=int)

        for i in range(projPoints.shape[0]):
            camPixCoordsH[i] = np.dot(K,projPoints[i]/projPoints[i,2])
        
        print(camPixCoordsH)

        camPixCoords = camPixCoordsH[:,0:2]
        
        num2D = np.zeros([img.shape[0], img.shape[1], 2] ,dtype=np.float32)

        for i in range(0, camPixCoords.shape[0]):
            [col, row] = camPixCoords[i]
            if projPoints[i,2] > 0:
                if 0<row<img.shape[0] and 0<col<img.shape[1]:
                    num2D[row,col,0] = projPoints[i,2]  #depth
                    num2D[row,col,1] = projPoints[i,1]  #height
                    img = cv2.circle(img, (col,row),5, (127 - 127*projPoints[i][1], 255  - 255*projPoints[i][1] , 255*projPoints[i][1]- 255), -1)
        
        np_file_name = image_name.replace(".png",".npy")
        
        np.save(os.path.join(path2D, np_file_name), num2D)   
        cv2.imshow("image", img)
        cv2.waitKey(10)
        
