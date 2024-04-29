
import argparse
from getImages import *
from estimateFeatures import *
from estimateCamera import *
from estimateShape import *
from structure3D import *
from projectStitch import *
from bundle_adjustment import *

def main(args):
    # Perform image stitching:

    print("calibrate camera")

    images, img_shape = get_images(folderName=args.calibName,  
                                    pre_process=False)    
    K, dist_coeffs = calibrate_camera(images=images, boardShape=(8,6))


    print("getImages")
    allImages, img_shape = get_images(folderName=args.folderName,  
                                   pre_process=False)
    
    print("rectifyCamera")
    allImages = rectifyCamera(images=allImages, K=K, dist_coeffs=dist_coeffs)

    #images = allImages[:]
    images = allImages[12 : 28]

    print("getKeypointsAndDescriptors")
    keypoints_list, descriptors_list = getSIFTKeypointsAndDescriptors(images=images)

    print("matchDescriptors")
    matched_descriptors              = matchDescriptors(descriptors_list=descriptors_list, 
                                                        distance_ratio_threshold=0.70, 
                                                        nMatches=50)

    print("matchPoints")
    points2d_list1, points2d_list2   = matchPoints(keypoints_list=keypoints_list, 
                                                   matched_descriptor_list=matched_descriptors)
    
    print("estimate_projection_matrices")
    P_list1, P_list2 = estimate_projection_matrices(points2d_list1=points2d_list1, 
                                                    points2d_list2=points2d_list2, K=K)

    print("triangulate_points_calibrated")
    points3d_list = triangulate_points_calibrated(P_list1=P_list1,
                                                  P_list2=P_list2,
                                                  points2d_list1=points2d_list1, 
                                                  points2d_list2=points2d_list2)
    

    print("bundle_adjustment")
    rvecs_list, tvecs_list, points3d_list,  = bundle_adjustment(points3d_list=points3d_list, 
                                                                points2d_list=points2d_list2, 
                                                                P_list=P_list2,
                                                                K=K, 
                                                                dist_coeffs=dist_coeffs,
                                                                img_shape=img_shape)

    print("estimate_cylinder_dimensions")
    cylinder_radius = estimate_cylinder_dimensions(triangulated_points=points3d_list, scaling_factor=1)
    #cylinder_radius = estimate_cylinder_dimensions(triangulated_points=points3d_list, scaling_factor=100)

    print("rectify_images_with_cylinder")
    rectified_images_ba = rectify_images_cylindrical_ba(images=images,
                                                        K=K,
                                                        rvecs=rvecs_list, 
                                                        tvecs=tvecs_list, 
                                                        r=cylinder_radius)


    #print("stitch_images features")
    #panorama_no_warp     = stitchImagesFeatures(images = images)
    #panorama_ba          = stitchImagesFeatures(images = rectified_images_ba)

    #displayImages(img1=panorama_no_warp, img2=panorama_ba)

    #print("stitch_images OpenCV")
    panorama_no_warp_cv     = stitch_images_cv_iterative(images = images, batch_size=len(images))
    panorama_ba_cv          = stitch_images_cv_iterative(images = rectified_images_ba, batch_size=len(images))

    #print("displayImages")
    #displayImages(img1=panorama_no_warp_cv, img2=panorama_ba_cv)

    print("saveImage")
    #saveImage(file_path = "panorama_no_warp.png", image=panorama_no_warp)
    #saveImage(file_path = "panorama_ba.png", image=panorama_ba)
    saveImage(file_path = "panorama_no_warp_cv.png", image=panorama_no_warp_cv)
    saveImage(file_path = "panorama_ba_cv.png", image=panorama_ba_cv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-processess the images.')
    parser.add_argument('-folderName', type=str, default="../data/processed/demo_1", help='The name of the target (demo) folder.')
    parser.add_argument('-calibName', type=str, default="../data/processed/Chessboard", help='The name of the target (demo) folder.')
    args = parser.parse_args()
    
    main(args=args)
    
    print("Done.")