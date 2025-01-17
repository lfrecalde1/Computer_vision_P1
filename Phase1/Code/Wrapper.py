#!/usr/bin/evn python
"""
RBE/CS Spring 2025: Classical and Deep LEarning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code
Worcester Polytechnic Institute
"""

# Code Starts Here
import numpy as np
import cv2

# Add any python libraries here
import argparse
import os
from skimage import feature
import random

def HarrisCorners(img):
    return cv2.cornerHarris(img,blockSize=3,ksize=11,k=0.01)

def scale_img(img):
    img = img/np.max(np.abs(img))
    return img.astype(np.float32)

def ANMS(Cimg,img,Nbest,path):
    """
    :param Cimg: Corner score Image obtained using cornermetric
    :param img: Input image
    :param Nbest: Number of best corners needed
    :param path: Path to save the anms feature image
    :return keypoints: Keypoints of all local maxima
    """
    region_max_coords = feature.peak_local_max(Cimg,min_distance=3,threshold_rel=0.005)
    # region_max_coords = region_max_coords[:,[1,0]]
    r = np.ones(len(region_max_coords))*np.inf
    ED = np.inf
    for i in range(len(region_max_coords)):
        for j in range(len(region_max_coords)):
            if Cimg[region_max_coords[j][0],region_max_coords[j][1]] > Cimg[region_max_coords[i][0],region_max_coords[i][1]]:
                ED = (region_max_coords[j][1] - region_max_coords[i][1])**2 + (region_max_coords[j][0] - region_max_coords[i][0])**2
            if ED < r[i]:
                r[i] = ED
    
    best_coords_idx = np.argsort(r)[::-1][:Nbest]
    best_coords = region_max_coords[best_coords_idx]
    keypoints = []
    for coord in best_coords:
        keypoints.append(cv2.KeyPoint(float(coord[1]), float(coord[0]), 1))
        img[coord[0],coord[1]] = [0,0,255]
    cv2.imwrite(path, img)
    return keypoints

def featureDescriptors(keypoints,img,path):
    """
    Describe each feature point by a feature vector
    :param keypoints: Keypoints of feature vector
    :return featuredescriptor
    :rytpe: ndarray
    """
    # Take a patch of size 41x41 around each feature point

    padded_img = cv2.copyMakeBorder(img, 
                                   20, 
                                   20, 
                                   20, 
                                   20, 
                                   cv2.BORDER_CONSTANT, 
                                   None,
                                   value=0)  # Black padding (value 0 for grayscale)
    featuredescriptor = []
    for keypoint in keypoints:
        # Apply Gaussian blur
        gauss_img = cv2.GaussianBlur(padded_img[int(keypoint.pt[1]):int(keypoint.pt[1])+41,int(keypoint.pt[0]):int(keypoint.pt[0])+41],(5,5),0)
        # Subsample the blurred output
        gauss_img.resize(8,8)
        # Reshape to 64x1
        gauss_vec = gauss_img.flatten()
        # Standardize the vector to have 0 mean and 1 variance
        mean_vec = gauss_vec - np.mean(gauss_vec)
        std_vec  = mean_vec/np.std(mean_vec)
        featuredescriptor.append(std_vec)

    return featuredescriptor

def featureMatching(descriptors_img_1,keypts_1,descriptors_img_2,keypts_2):
    matching_pairs = []
    mapping        = []
    for i in range(len(descriptors_img_1)):
        min_val  = np.inf
        smin_val = np.inf
        for j in range(len(descriptors_img_2)):
            distance = sum((descriptors_img_1[i] - descriptors_img_2[j])**2)
            if distance < min_val:
                smin_val = min_val
                min_val = distance
                arg_min = j
            elif distance < smin_val:
                smin_val = distance
        if min_val/smin_val < 0.5: #0.31 works best with minimal errors!
            matching_pairs.append(cv2.DMatch(i,arg_min,min_val))
            mapping.append([keypts_1[i].pt[0],keypts_1[i].pt[1],keypts_2[arg_min].pt[0],keypts_2[arg_min].pt[1],min_val])
    return matching_pairs,mapping

def homography(A):
    eigen_vals,eigen_vecs = np.linalg.eig(np.matmul(np.transpose(A),A))
    idx = np.argmin(eigen_vals)
    return eigen_vecs(idx)

def calculate_inliers(H,maps,threshold):
    inlier_vec = []
    for sample in maps:
        temp_pt = H*np.array([[sample[0]],
                              [sample[1]],
                              [1]])
        SSD = (temp_pt[0]/(temp_pt[2]+1e-6)-sample[2])**2 + (temp_pt[1]/(temp_pt[2]+1e-6)-sample[3])**2
        if SSD < threshold:
            inlier_vec.append(sample)
    return len(inlier_vec),inlier_vec

def RANSAC(keypoints_list,feature_des_list,maps,Nmax,threshold):
    Inliers = 0
    prev_inliers = 0
    max_inliers_vec = []
    pano = True
    for _ in range(Nmax):
        p = random.sample(maps,4)
        A = np.empty((0, 9))
        for samp in p:
            xs = samp[0]
            ys = samp[1]
            xd = samp[2]
            yd = samp[3]
            Avec = np.array([xs,ys,1,0,0,0,-xd*xs,-xd*ys,-xd],
                            [0,0,0,xs,ys,1,-yd*xs,-yd*ys,-yd])
            A = np.vstack([A,Avec])
            H = np.reshape(homography(A),(3,3))
            H = (1/H[2,2])*H
            num_inliers,inlier_vec = calculate_inliers(H,maps,threshold)
            if num_inliers>0.9*len(maps):
                break
            if num_inliers > prev_inliers:
                max_inliers_vec = inlier_vec
    
    
    if num_inliers < 4:
        pano = False
    

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ImagesPath', default='../Data/Train/Set1', help='Path where the images are stored, Default:../Data/Train/Set1')
    Parser.add_argument('--NumFeatures', default=200, help='Number of best features to extract from each image, Default:100')

    Args = Parser.parse_args()
    # IMAGE_DIR = '../DataTrain/Set1'
    IMAGE_DIR = Args.ImagesPath
    NumFeatures = Args.NumFeatures
    CORNERS_DIR = './Corners/'
    CORNERS_DIR = os.path.join(CORNERS_DIR,os.path.basename(IMAGE_DIR) + '/')
    if not os.path.exists(CORNERS_DIR):
        os.makedirs(CORNERS_DIR)    
    """
    Read a set of images for Panorama Stitching
    """
    images = []
    feature_des_list = []
    keypoints_list = []

    for filename in os.listdir(IMAGE_DIR):
        image_path = os.path.join(IMAGE_DIR,filename)
        img      = cv2.imread(image_path)
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        """
	    Corner Detection
	    Save Corner detection output as corners.png
	    """
        harris_corners = HarrisCorners(gray_img)
        # corners = cv2.dilate(harris_corners,None)
        corners = harris_corners
        #####################33
        base_name = os.path.splitext(filename)[0]      
        output_path = os.path.join(CORNERS_DIR,'corners_harris_' + base_name + '.png')
        harris_img = scale_img(corners)
        img_markers = img.copy()
        img_markers[corners > 0.0025*corners.max()] = [0,0,255]
        cv2.imwrite(output_path,img_markers)

        """
	    Perform ANMS: Adaptive Non-Maximal Suppression
	    Save ANMS output as anms.png
	    """
        keypoints = ANMS(harris_img,img,NumFeatures,os.path.join(CORNERS_DIR,'anms_' + base_name + '.png'))
        keypoints_list.append(keypoints)

        """
	    Feature Descriptors
	    Save Feature Descriptor output as FD.png
	    """
        feature_des = featureDescriptors(keypoints,gray_img,os.path.join(CORNERS_DIR,'FD_' + base_name + '.png'))
        feature_des_list.append(feature_des)
    """
    Feature Matching
    Save Feature Matching output as matching.png
    """
    img_names = [f for f in os.listdir(IMAGE_DIR)]
    for i in range(len(img_names)):
        for j in range(i+1,len(img_names)):
            img1 = cv2.imread(os.path.join(IMAGE_DIR,img_names[i]))
            img2 = cv2.imread(os.path.join(IMAGE_DIR,img_names[j]))
            matches,maps = featureMatching(feature_des_list[i],keypoints_list[i],feature_des_list[j],keypoints_list[j])
            matching_img = cv2.drawMatches(img1,keypoints_list[i],img2,keypoints_list[j],matches, None, matchColor=(0, 255, 255), flags=2)
            output_path = os.path.join(CORNERS_DIR,'feature_matching' + str(i) + str(j) + '.png')
            cv2.imwrite(output_path,matching_img)

            """
            Refine: RANSAC, Estimate Homography
            """
            # RANSAC(keypoints_list,feature_des_list,maps)
            """
            Image Warping + Blending
            Save Panorama output as mypano.png
            """
    #     images.append(harris_img)
    #     cv2.imshow("filename",img_markers)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    

if __name__ == "__main__":
    main()