# Advanced Lane Finding Project #

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration1.jpg "Raw Image - Camera Calibration"
[image2]: ./output_images/calibration1_undistorted.jpg "Undistorted Image"
[image3]: ./output_images/test8.png "Example Image"
[image4]: ./output_images/test8_undistorted.jpg "Undistorted Image"
[image5]: ./output_images/test8_gradient.jpg "Gradient Example"
[image6]: ./output_images/test8_colorspace.jpg "Threshold Colorspace Example"
[image7]: ./output_images/test8_grad_color.jpg "Binary Example"
[image8]: ./output_images/straight_lines1.jpg "Binary Example"
[image9]: ./output_images/straight_lines1_warped.jpg "Binary Example"
[image10]: ./output_images/test8_warped.jpg "Warp Example"
[image11]: ./output_images/test8_lane_ident.jpg "Lane Identification"
[image12]: ./output_images/test8_final.jpg "Final Output"
[video1]: ./project_output.mp4 "Project Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file called `camera_cal/cam_cal.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

Camera raw image:

![][image1]

Applied distortion correction:

![][image2]

### Pipeline (single images)

I will show all my steps on this example image I took out of the project video.

![][image3]

#### 1. Provide an example of a distortion-corrected image.

After applying distortion-correction the image looks like this:

![][image4]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 109 through 122 in `images.py`).  Here's the my output for this step for the example image.

First image with only gradient thresholds applied:

![][image5]

Image with only color thresholds applied:

![][image6]

And combined.

![][image7]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes the functions `cv2.getPerspectiveTransform` and `cv2.warpPerspective` (steps at lines 124 through 135 in `images.py`). The last function that takes as inputs an (preprocessed) image (`preprocess_img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([[580, 462],
				  [703, 462],
				  [1093, 720],
				  [220, 720]])

offset = img_size[0]*.25

dst = np.float32([[offset, 0],
				  [img_size[0]-offset, 0],
				  [img_size[0]-offset, img_size[1]],
				  [offset, img_size[1]]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 580, 462      | 320, 0        |
| 703, 462      | 960, 0        |
| 1093, 720     | 960, 720      |
| 220, 720      | 320, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image (with straight lines) and its warped counterpart to verify that the lines appear parallel in the warped image.

Test image straight lines:

![][image8]

Warped image:

![][image9]

And for the example image (binary mode) it looks like this:

![][image10]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To identify the lane-line pixels I used the Sliding-Window-Search approach out of lesson 33 of the project slides in combination with fitting a polynomial. To get a more robust output I used the class `tracker` and two global variables to store the recent found lanes. I averaged over the last 10 to minimize the impact of single bad frames.

The code is in lines 137 through 188 in `images.py` (corresponding lines 158 through 182 in `video.py`) and in `tracker.py`.  

![][image11]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is calculated as stated in lesson 35 in the project slides.

For the position of the vehicle, I assumed the camera is mounted at the center of the car and the deviation of the midpoint of the lane from the center of the image is the offset.

I did this in lines 203 through 223 in my code in `images.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 190 through 201 in my code in `images.py`. Here is an example of my result on a test image:

![][image12]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

If no lines are found because of bad light conditions or missing lane markers the pipeline will fall short. To fix this you could try to implement an algorithm that varies colorspaces, if the standard approch does not identify any line points.

If the line curvature is too big in combination with a dashed line, it could be that the sliding window approach will have difficulties to identify the next line section. To avoid this you could implement a more advanced line search method.
