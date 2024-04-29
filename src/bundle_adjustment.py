
import numpy as np
import cv2
from scipy.optimize import minimize, least_squares, Bounds

def reprojection_error(params, points_3d, points_2d, K, dist_coeffs):
    """
    Calculate reprojection error given camera parameters and 3D-2D correspondences.
    
    Args:
    - params (numpy array): concatenated array of rotation vectors, translation vectors, and 3D points
    - points_3d (numpy array): 3D points in the scene
    - points_2d (numpy array): corresponding 2D points in the image plane
    - K (numpy array): camera intrinsic matrix
    - dist_coeffs (numpy array): distortion coefficients
    
    Returns:
    - error (float): sum of squared reprojection errors
    """
    num_points = len(points_3d)
    #num_params_per_point = 3
    #num_params_per_camera = 6
    
    # Extract rotation vector, translation vector, and 3D points
    rvec = params[:3]
    tvec = params[3:6]
    optimized_points_3d = params[6:].reshape((num_points, 3))

    #print("rvec", rvec)
    #print("tvec", rvec)
    #print("points3d", optimized_points_3d)
    
    # Project 3D points to 2D using extrinsic parameters
    projected_points, _ = cv2.projectPoints(optimized_points_3d, rvec, tvec, K, dist_coeffs)
    #projected_points = projected_points.squeeze()
    
    # Calculate reprojection error
    error = cv2.norm(projected_points - points_2d, cv2.NORM_L2) / len(projected_points)
    #print("error", error)
    
    return error

def bundle_adjustment(points3d_list, points2d_list, P_list, K, dist_coeffs, img_shape):
    
    rvecs_list = []
    tvecs_list = [] 
    points3d_list_optimized = []

    # Print lengths of bounds
    #print("Length of bounds:", len(bounds))

    for points_2d, points_3d, P in zip(points2d_list, points3d_list, P_list):
        
        num_points = len(points_3d)

        # Define bounds for rotation vectors
        bounds_rvec = [(-np.pi/72, np.pi/72)] * 3  # 3 rotation vectors

        # Define bounds for translation vectors
        bounds_tvec = [(-10, 10)] * 3  # 3 translation vectors

        # Define bounds for 3D points
        bounds_points3d = [(-10, 10)] * (num_points * 3)  # Adjust num_points as needed

        # Concatenate bounds for all parameters
        bounds = bounds_rvec + bounds_tvec + bounds_points3d

        # Use solvePnP to get initial guess for rotation and translation
        #_, rvec_init, tvec_init = cv2.solvePnP(points_3d, points_2d, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        #print("rvec",rvec_init.shape)
        #print("tvec",tvec_init.shape)

        # Using the previous projection matrices instead:
        _, rvec_init, tvec_init, _, _, _, _  = cv2.decomposeProjectionMatrix(P)
        tvec_init = tvec_init[:3] / tvec_init[3]
        rvec_init, _ = cv2.Rodrigues(rvec_init)
        #print("rvec", rvec_init.shape)
        #print("tvec", tvec_init.shape)

        # Decompose the projection matrix into the intrinsic matrix (K) and rotation-translation matrix (RT)
        
        # Flatten and concatenate rvec_init and tvec_init separately with points_3d
        initial_params = np.hstack([rvec_init.flatten(), tvec_init.flatten(), points_3d.flatten()])

        #print("\ninit_tvec", tvec_init)
        #print("init_rvec", rvec_init)

        # Perform optimization
        result = minimize(reprojection_error, 
                          initial_params, 
                          method="L-BFGS-B",
                          bounds=bounds,
                          options = {'maxiter': 1000},
                          args=(points_3d, points_2d, K, dist_coeffs))


        # Extract optimized parameters
        optimized_params = result.x
        optimized_rvec = optimized_params[:3]
        optimized_tvec = optimized_params[3:6]
        optimized_points_3d = optimized_params[6:].reshape((num_points, 3))

        #print("opt_tvec", optimized_tvec)
        #print("opt_rvec\n", optimized_rvec)

        rvecs_list.append(optimized_rvec)
        tvecs_list.append(optimized_tvec)
        points3d_list_optimized.append(optimized_points_3d)
    
    return rvecs_list, tvecs_list, points3d_list_optimized



def rectify_images(images, rvecs_list, tvecs_list, K, dist_coeffs):
    num_images = len(images)
    rectified_images = []

    for i in range(num_images):
        image = images[i]
        rvecs = np.zeros((1, 3)) if i == 0 else rvecs_list[i - 1]
        tvecs = np.zeros((1, 3)) if i == 0 else tvecs_list[i - 1]
        
        # Compute the rotation matrix from rotation vector
        R, _ = cv2.Rodrigues(rvecs)
        
        # Combine the rotation matrix and translation vector
        Rt = np.hstack((R, tvecs.reshape(3, 1)))
        
        # Compute the rectification matrix
        rectification_matrix = K.dot(Rt)
        
        # Apply distortion correction
        rectified_image = cv2.undistort(image, K, dist_coeffs)
        
        # Apply rectification transformation to the image
        rectified_image = cv2.warpPerspective(rectified_image, rectification_matrix, (image.shape[1], image.shape[0]))
                
        # Append the rectified image to the list
        rectified_images.append(rectified_image)
    
    return rectified_images