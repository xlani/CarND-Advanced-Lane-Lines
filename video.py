################################################################################
# code sources:
#   * Udacity project lecture
#   * Udacity Q&A project video (https://www.youtube.com/watch?v=vWY8YUayf9Q)
################################################################################

################################################################################
# imports and load data
################################################################################

# imports
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from tracker import tracker

# read in the saved camera matrix and distortion coefficients
dist_pickle = pickle.load(open( "camera_cal/calibration_pickle.p", "rb" ))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

################################################################################
# helper functions
################################################################################

# function that applies Sobel x or y
# then takes an absolute value and applies a threshold
def abs_sobel_thresh(img, orient = 'x', thresh = (0, 255)):

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    if orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    # take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)

    # scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # create a mask of 1's where the scaled gradient magnitude
    # is >= thresh_min and <= thresh_max and return it
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary

# function that thresholds a chosen channel of chosen color_space
# default = RGB, no threshold
def color_select(img, color_space = None, channel = 0, thresh = (0, 255)):

    # change to RGB per default or convert to chosen color space
    if (color_space == None or color_space == 'RGB'):
        color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif color_space == 'HSV':
        color = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif color_space == 'HLS':
        color = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    elif color_space == 'LUV':
        color = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    elif color_space == 'YCrCb':
        color = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # select channel
    if channel == 0:
        color_channel = color[:,:,0]
    if channel == 1:
        color_channel = color[:,:,1]
    if channel == 2:
        color_channel = color[:,:,2]

    # apply a threshold to the channel and return a binary image of threshold result
    binary = np.zeros_like(color_channel)
    binary[(color_channel >= thresh[0]) & (color_channel <= thresh[1])] = 1
    return binary

# function to draw a window_mask for a given width, height, center & level
def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height), \
           max(0,int(center-width)):min(int(center+width),img_ref.shape[1])] = 1
    return output

################################################################################
# functions process_img
################################################################################

def process_img(img):

    # undistort image
    img = cv2.undistort(img, mtx, dist, None, mtx)

    # process image and generate binary pixel of interets
    preprocess_img = np.zeros_like(img[:,:,0])
    gradx = abs_sobel_thresh(img, orient = 'x', thresh = (12, 255))
    grady = abs_sobel_thresh(img, orient = 'y', thresh = (12, 255))
    s_binary = color_select(img, color_space = 'HSV', channel = 1, thresh = (100, 255))
    v_binary = color_select(img, color_space = 'HSV', channel = 2, thresh = (100, 255))
    preprocess_img[((gradx == 1) & (grady == 1) | (s_binary == 1) & (v_binary == 1))] = 255
    # preprocess_img[((gradx == 1) & (grady == 1))] = 255
    # preprocess_img[((s_binary == 1) & (v_binary == 1))] = 255
    result = preprocess_img

    # preparation for transformation
    img_size = (img.shape[1], img.shape[0])                                         # 1280 x 720
    src = np.float32([[580, 462], [703, 462], [1093, 720], [220, 720]])             # values tuned for straight_line*.jpg
    offset = img_size[0]*.25                                                        # 320
    dst = np.float32([[offset, 0], [img_size[0]-offset, 0], \
                      [img_size[0]-offset, img_size[1]], [offset, img_size[1]]])

    # perform transformation
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(preprocess_img, M, img_size, flags = cv2.INTER_LINEAR)

    # tracker class for finding & storing information about lanes
    window_width = 40
    window_height = 80
    curve_centers = tracker(mywindow_width = window_width, mywindow_height = window_height, \
                            mymargin = 40, my_ym = 10./720, my_xm = 4./384, mysmooth_factor = 20)
    window_centroids = curve_centers.find_window_centroids(warped)

    # points used to draw all the left and right windows
    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)

    # points used to find the left and right lanes
    leftx = []
    rightx = []

    # go through each level and draw the windows
    for level in range(0,len(window_centroids)):
        # window_mask is a function to draw window areas
        l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
        r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
        # add center value found in frame to the list of lane points per left & right
        leftx.append(window_centroids[level][0])
        rightx.append(window_centroids[level][1])
        # add graphic points from window mask here to total pixels found
        l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
        r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

    # draw the results
    template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
    zero_channel = np.zeros_like(template) # create a zero color channel
    template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
    warpage= np.dstack((warped, warped, warped))*255 # making the original road pixels 3 color channels
    result = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results

    # fit the lane boundaries to the left,right center positions
    yvals = range(0,warped.shape[0])
    res_yvals = np.arange(warped.shape[0]-(window_height/2), 0, -window_height)

    left_fit = np.polyfit(res_yvals, leftx, 2)
    left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]
    left_fitx = np.array(left_fitx, np.int32)

    right_fit = np.polyfit(res_yvals, rightx, 2)
    right_fitx = right_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2]
    right_fitx = np.array(right_fitx, np.int32)

	# average the fitted points to improve stability
    recent_llanes.append(left_fitx)
    recent_rlanes.append(right_fitx)

    avg_left = np.average(recent_llanes[-10:], axis = 0)
    avg_right = np.average(recent_rlanes[-10:], axis = 0)

    left_lane = np.array(list(zip(np.concatenate((avg_left - window_width/2, avg_left[::-1] + window_width/2), axis = 0), \
                np.concatenate((yvals, yvals[::-1]), axis = 0))), np.int32)
    right_lane = np.array(list(zip(np.concatenate((avg_right - window_width/2, avg_right[::-1] + window_width/2), axis = 0), \
                 np.concatenate((yvals, yvals[::-1]), axis = 0))), np.int32)
    middle_marker = np.array(list(zip(np.concatenate((avg_left + window_width/2, avg_right[::-1] - window_width/2), axis = 0), \
                    np.concatenate((yvals, yvals[::-1]), axis = 0))), np.int32)

	# draw lanes and middle marker
    road = np.zeros_like(img)
    cv2.fillPoly(road, [left_lane], color = [255, 0, 0])
    cv2.fillPoly(road, [right_lane], color = [0, 0 , 255])
    cv2.fillPoly(road, [middle_marker], color = [0, 255, 0])

    # perform retransformation
    road_warped = cv2.warpPerspective(road, Minv, img_size, flags = cv2.INTER_LINEAR)

    #base = cv2.addWeighted(img, 1.0, road_warped_bkg, -1.0, 0.0)
    result = cv2.addWeighted(img, 1.0, road_warped, 1.0, 0.0)

    # parameter for calculations
    xm_per_pix = curve_centers.xm_per_pix
    ym_per_pix = curve_centers.ym_per_pix

    # calculate the offset of the car on the road
    camera_center = (left_fitx[-1] + right_fitx[-1])/2
    center_diff = (camera_center - warped.shape[1]/2) * xm_per_pix
    side_pos = 'left'
    if center_diff <= 0:
        side_pos = 'right'

    left_fit_cr = np.polyfit(np.array(res_yvals, np.float32)*ym_per_pix, np.array(leftx, np.float32)*xm_per_pix, 2)
    l_curverad = ((1 + (2*left_fit_cr[0]*yvals[-1]*ym_per_pix + left_fit_cr[1]) **2) **1.5) / np.absolute(2*left_fit_cr[0])
    right_fit_cr = np.polyfit(np.array(res_yvals, np.float32)*ym_per_pix, np.array(rightx, np.float32)*xm_per_pix, 2)
    r_curverad = ((1 + (2*right_fit_cr[0]*yvals[-1]*ym_per_pix + right_fit_cr[1]) **2) **1.5) / np.absolute(2*right_fit_cr[0])

    # draw the text showing curvature, offset and speed
    cv2.putText(result, 'Left Radius of Curvature = ' +str(round(l_curverad,3)) +'(m)', (25, 50), cv2.FONT_HERSHEY_PLAIN, 0.98, (255, 255, 255), 2)
    cv2.putText(result, 'Right Radius of Curvature = ' +str(round(r_curverad,3)) +'(m)', (25, 75), cv2.FONT_HERSHEY_PLAIN, 0.98, (255, 255, 255), 2)
    cv2.putText(result, 'Vehicle is ' +str(abs(round(center_diff,3))) +'m ' +side_pos +' of center', (25, 100), cv2.FONT_HERSHEY_PLAIN, 0.98, (255, 255, 255), 2)

    return result

################################################################################
# video pipeline
################################################################################

output_video = 'project_output.mp4'
input_video = 'project_video.mp4'

# global variable for found lane points
recent_llanes = []
recent_rlanes = []

clip1 = VideoFileClip(input_video)
video_clip = clip1.fl_image(process_img)
video_clip.write_videofile(output_video, audio = False)
