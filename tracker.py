################################################################################
# code sources:
#   * Udacity project lecture
#   * Udacity Q&A project video (https://www.youtube.com/watch?v=vWY8YUayf9Q)
################################################################################

# imports
import numpy as np
import cv2

# tracker class
class tracker():

    # when starting a new instance please be sure to specify all unassigned variables
    def __init__(self, mywindow_width, mywindow_height, mymargin, my_ym = 1, my_xm = 1, mysmooth_factor = 15):

        # list that stores all the past (left,right) center set values used for smoothing the output
        self.recent_centers = []

        # the window pixel width of the center values, used to count pixels inside center windows to determine curve values
        self.window_width = mywindow_width

        # the window pixel height of the center values, used to count pixels inside center windows to determine curve values
        # breaks the image into vertical levels
        self.window_height = mywindow_height

        # the pixel distance in both directions to slide (left_window + right_window) template for searching
        self.margin = mymargin
        self.ym_per_pix = my_ym # meters per pixel in vertical axis
        self.xm_per_pix = my_xm # meters per pixel in horizontal axis
        self.smooth_factor = mysmooth_factor

# the main tracking function for finding and storing lane segment positions
    def find_window_centroids(self, warped):

        window_width = self.window_width
        window_height = self.window_height
        margin = self.margin

        border = 200

        window_centroids = [] # store the (left,right) window centroid positions per level
        window = np.ones(window_width) # create our window template that we will use for convolutions

        # first find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template

        # sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(border):int(warped.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum))-window_width/2 + border

        r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):warped.shape[1]-border], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)

        # add what we found for the first layer
        window_centroids.append((l_center,r_center))

        # go through each layer looking for max pixel locations
        for level in range(1,int(warped.shape[0]/window_height)):

			# convolve the window into the vertical slice of the image
    	    image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
    	    conv_signal = np.convolve(window, image_layer)

    	    # find the best left centroid by using past left center as a reference
    	    # use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
			# if no point is found stay where you are
    	    offset = window_width/2
    	    l_min_index = int(max(l_center+offset-margin,0))
    	    l_max_index = int(min(l_center+offset+margin,warped.shape[1]))

    	    if(np.argmax(conv_signal[l_min_index:l_max_index]) == 0):
    	        l_center = l_min_index + offset
    	    else:
    	        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
				
    	    # find the best right centroid by using past right center as a reference
    	    r_min_index = int(max(r_center+offset-margin,0))
    	    r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
    	    if(np.argmax(conv_signal[r_min_index:r_max_index]) == 0):
    	        r_center = r_min_index + offset
    	    else:
    		    r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
    	    # add what we found for that layer
    	    window_centroids.append((l_center,r_center))

        self.recent_centers.append(window_centroids)
        # return averaged values of the line centers, helps to keep the markers from jumping around too much
        return np.average(self.recent_centers[-self.smooth_factor:], axis = 0)
