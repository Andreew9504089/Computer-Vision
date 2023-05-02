import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import random
import math
import sys
import os

# read the image file & output the color & gray image
def read_img(path):
    # opencv read image in BGR color space
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray

# the dtype of img must be "uint8" to avoid the error of SIFT detector
def img_to_gray(img):
    if img.dtype != "uint8":
        print("The input image dtype is not uint8 , image type is : ", img.dtype)
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

# create a window to show the image
# It will show all the windows after you call im_show()
# Remember to call im_show() in the end of main
def creat_im_window(window_name,img):
    cv2.imshow(window_name,img)

# show the all window you call before im_show()
# and press any key to close all windows
def im_show():
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def FindSIFT(img):
    SIFT_Detector = cv2.SIFT_create()
    kp, des = SIFT_Detector.detectAndCompute(img, None)
    
    return kp, des

def DrawKeyPoints(kp, gray, rgb):
    tmp = rgb.copy()
    res = cv2.drawKeypoints(gray, kp, tmp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return res

def PlotKeypoints(plts, cnt):
    for i,plt in enumerate(plts):
        if i == 0:
            res = plt
        else:
            res = np.concatenate((res, plt), axis=1)
            
    creat_im_window("Keypoints"+str(cnt), res)
    return res
    
def KNNFeatureMatching(kp0, kp1, des0, des1, threshold = 0.5):
    k = 2
    # Find k nearest des1 neighbor for des0 using brute force
    matches = {}
    for i, feature0 in enumerate(des0):
        matches[i] = []
        for j, feature1 in enumerate(des1):
            dist = np.linalg.norm(feature0 - feature1)
            
            if j < k:
                matches[i].append([j,dist])
            else:
                for t in range(k):
                    if dist < matches[i][t][1]:
                        matches[i][t] = [j, dist]
                        break
    
    good_pairs = []
    # Perform Lowe's ratio check
    for p in matches.keys():
        if matches[p][0][1] < threshold*matches[p][1][1]:
            good_pairs.append([p,matches[p][0][0]]) 
    
    good_matches = []
    for pair in good_pairs:
        good_matches.append(list(kp0[pair[0]].pt + kp1[pair[1]].pt))
    
    good_matches = np.asarray(good_matches)
    
    return good_matches

def PlotMatching(matches, img_cat):
    match_img = cv2.cvtColor(img_cat.copy(), cv2.COLOR_BGR2RGB)
    offset = img_cat.shape[1]/2
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(match_img).astype('uint8'))
    
    ax.plot(matches[:, 0], matches[:, 1], 'xr')
    ax.plot(matches[:, 2] + offset, matches[:, 3], 'xr')
     
    ax.plot([matches[:, 0], matches[:, 2] + offset], [matches[:, 1], matches[:, 3]],
            'r', linewidth=0.5)

    plt.show()

def FindHomography(matches):
    rows = []
    for i in range(matches.shape[0]):
        p1 = np.append(matches[i][0:2], 1)
        p2 = np.append(matches[i][2:4], 1)
        row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]]
        row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]]
        rows.append(row1)
        rows.append(row2)
    rows = np.array(rows)
    U, s, V = np.linalg.svd(rows)
    H = V[-1].reshape(3, 3)
    H = H/H[2, 2] 
    
    return H

def Randompoints(matches, k=4):
    idx = random.sample(range(len(matches)), k)
    point = [matches[i] for i in idx ]
    
    return np.array(point)

def ComputeErrors(points, H):
    num_points = len(points)
    all_p1 = np.concatenate((points[:, 0:2], np.ones((num_points, 1))), axis=1)
    all_p2 = points[:, 2:4]
    estimate_p2 = np.zeros((num_points, 2))
    for i in range(num_points):
        temp = np.dot(H, all_p1[i])
        estimate_p2[i] = (temp/temp[2])[0:2] # set index 2 to 1 and slice the index 0, 1
    # Compute error
    errors = np.linalg.norm(all_p2 - estimate_p2 , axis=1) ** 2

    return errors

def RANSAC(matches, threshold = 0.5, iters = 3000):
    num_best_inliers = 0
    
    for i in range(iters):
        points = Randompoints(matches)
        H = FindHomography(points)
        
        #  avoid dividing by zero 
        if np.linalg.matrix_rank(H) < 3:
            continue
            
        errors = ComputeErrors(matches, H)
        idx = np.where(errors < threshold)[0]
        inliers = matches[idx]

        num_inliers = len(inliers)
        if num_inliers > num_best_inliers:
            best_inliers = inliers.copy()
            num_best_inliers = num_inliers
            best_H = H.copy()
            
    print("inliers/matches: {}/{}".format(num_best_inliers, len(matches)))
    
    return best_inliers, best_H

def StitchImage(img1, img2, H):
    print("stiching image ...")
    
    # Convert to double and normalize. Avoid noise.
    left = cv2.normalize(img1.astype('float'), None, 
                            0.0, 1.0, cv2.NORM_MINMAX)   
    # Convert to double and normalize.
    right = cv2.normalize(img2.astype('float'), None, 
                            0.0, 1.0, cv2.NORM_MINMAX)   
    
    # left image
    height_l, width_l, channel_l = left.shape
    height_r, width_r, channel_r = right.shape
    h = np.minimum(height_l, height_r)
    w = np.minimum(width_l, width_r)
    
    corners = [[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]
    corners_new = [np.dot(H, corner) for corner in corners]
    corners_new = np.array(corners_new).T 
    x_news = corners_new[0] / corners_new[2]
    y_news = corners_new[1] / corners_new[2]
    y_min = min(y_news)
    x_min = min(x_news)

    translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    H = np.dot(translation_mat, H)
    
    # Get height, width
    height_new = int(round(abs(y_min) + height_l))
    width_new = int(round(abs(x_min) + width_l))
    size = (width_new, height_new)

    # right image
    warped_l = cv2.warpPerspective(src=left, M=H, dsize=size)
    
    height_new = int(round(abs(y_min) + height_r))
    width_new = int(round(abs(x_min) + width_r))
    size = (width_new, height_new)
    

    warped_r = cv2.warpPerspective(src=right, M=translation_mat, dsize=size)
     
    black = np.zeros(3)  # Black pixel.
    
    # Stitching procedure, store results in warped_l.
    for i in tqdm(range(warped_r.shape[0])):
        for j in range(warped_r.shape[1]):
            pixel_l = warped_l[i, j, :]
            pixel_r = warped_r[i, j, :]
            
            if not np.array_equal(pixel_l, black) and np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_l
            elif np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_r
            elif not np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                opt = [pixel_l, pixel_r]
                warped_l[i, j, :] = (pixel_l + pixel_r)/2#opt[np.argmax([np.linalg.norm(pixel_l), np.linalg.norm(pixel_r)])]
            else:
                pass
                  
    stitch_image = warped_l[:warped_r.shape[0], :warped_r.shape[1], :]
    
    return stitch_image

if __name__ == '__main__':
    # the example of image window
    # creat_im_window("Result",img)
    # im_show() 

    # you can use this function to store the result
    # cv2.imwrite("result.jpg",img)
    
    baseline_paths = ['m1.jpg', 'm2.jpg', 'm3.jpg', 'm4.jpg', 'm5.jpg', 'm6.jpg']
    bonus_paths = ['m1.jpg', 'm2.jpg', 'm3.jpg', 'm4.jpg']
    paths = [baseline_paths, bonus_paths]

    src = 1
    res = []
    res_gray = []
    for i in range(len(paths[src])):
        if src == 0:
            if i == 0:
                img0 , img_gray0 = read_img('./baseline/'+paths[src][i])
            else:
                img0 , img_gray0 = res, res_gray
                
            if i < len(paths[src])-1:   
                img1 , img_gray1 = read_img('./baseline/'+paths[src][i+1])
            else:
                break

        elif src == 1:
            if i == 0:
                img0, img_gray0 = read_img('./bonus/'+paths[src][i])
            else:
                img0, img_gray0 = res, res_gray
                
            if i < len(paths[src])-1:    
                img1, img_gray1 = read_img('./bonus/'+paths[src][i+1])
            else:
                break

        kp0, des0 = FindSIFT(img_gray0)
        kp1, des1 = FindSIFT(img_gray1)
        plt0 = DrawKeyPoints(kp0, img_gray0, img0)
        plt1 = DrawKeyPoints(kp1, img_gray1, img1)
        
        plts = [plt0, plt1]
        
        #PlotKeypoints(plts, i)
        
        #img_cat = np.concatenate((img0, img1), axis=1)
        
        matches = KNNFeatureMatching(kp0, kp1, des0, des1, threshold=0.5)
        #PlotMatching(matches, img_cat)
        
        inliers, H = RANSAC(matches, 0.5, 3000)
        #PlotMatching(inliers, img_cat)
        
        res = StitchImage(img0, img1, H).astype(np.float32)
        res_gray = (cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)*255).astype(np.uint8)
    
        cv2.imwrite('./bonus/result_'+str(i)+'.jpg', (res*255).astype(np.uint8))
        cv2.imwrite('./bonus/result_gray_'+str(i)+'.jpg', res_gray)
        
    creat_im_window("res", res)    
    im_show()
    
    
    
    