import numpy as np
import cv2

def triangulate_points_calibrated(P_list1, P_list2, points2d_list1, points2d_list2):
    # Initialize camera matrix

    # Triangulate 3D points from image correspondences
    triangulated_points = []
    for i in range(len(points2d_list1)):

        P1 = P_list1[i]
        P2 = P_list2[i]

        points1 = np.float32(points2d_list1[i]).reshape(-1, 1, 2)
        points2 = np.float32(points2d_list2[i]).reshape(-1, 1, 2)


        # Triangulate 3D points
        points_4d = cv2.triangulatePoints(P1, P2, points1, points2)
        points_3d = points_4d[:3] / points_4d[3]
        triangulated_points.append(points_3d.T)


    return triangulated_points
