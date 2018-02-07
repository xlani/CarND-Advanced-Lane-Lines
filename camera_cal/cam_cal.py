# imports

import numpy as np
import cv2
import glob
import pickle

# numbers x/y dimensions of corner points
nx = 9
ny = 6

# prepare object points
objp = np.zeros((nx * ny, 3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

# arrays to store object points and image points from all the images
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane

# make a list of calibration images
images = glob.glob('calibration*.jpg')

#print(images)

# step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # if found, add object points and image points
    if ret == True:
        print('working on', fname)
        objpoints.append(objp)
        imgpoints.append(corners)

        # draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        print(fname)
        write_name = 'corners_'+fname+'.jpg'
        cv2.imwrite(write_name, img)

#print(objpoints[0].shape)
#print(imgpoints[0].shape)

# load image for reference
img = cv2.imread('calibration1.jpg')
img_size = (img.shape[1], img.shape[0])

# camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2. calibrateCamera(objpoints, imgpoints, img_size, None, None)

# undistort the reference image with found calibration parameters
img_undist = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('calibration1_undistorted.jpg', img_undist)

# save the camera calibration result for later use
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open('calibration_pickle.p', "wb"))
