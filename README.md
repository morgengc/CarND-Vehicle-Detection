## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./output_images/hog_image.jpg
[image3]: ./examples/sliding_windows.png
[image4]: ./output_images/search_and_classify.jpg
[image5]: ./output_images/heatmap_and_bbox.jpg
[image6]: ./output_images/labels_map.jpg
[image7]: ./output_images/last_frame_with_bbox.jpg

Python files in this project:

* `utils.py`: Contains all utility functions
* `classifier.py`: SVM classifier, write all parameters to a pickle file after training
* `main.py`: Read frame from video stream, detect vehicle in the frame, and write to an output video stream
* `tracker.py`: Tracking heat map
* `detecting.py`: Test script to visualize various images

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in funtion `extract_features()` at line 129 of the file called `utils.py`. 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`(function `preview_hog_image()` in `detecting.py`):

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and I found these value made a better result:

* orientations=9
* pixels_per_cell=(8, 8)
* cells_per_block=(2, 2)
* hog_channel='ALL'

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG parameters listed in previous section, to see these settings please refer to line 25 to 34 in `classifier.py` for detail.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search a stripe region limited to [ystart=400, yend=656], because no car will appear in regions beyond this stripe. And I choose `overlap = 0.5` for test.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

![alt text][image4]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video.mp4), and a [link to video combined with lane line](./project_video_lane_output_2scales.mp4).


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

Here are six frames and their corresponding heatmaps:

![alt text][image5]

Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* Speed. Despite all the efforts in optimizing the algorithm, the detection is still far from real-time. I don't really know how to dramatically improve the performance with a CPU-based solution. However, given that most of the pipeline is easily parallelizable, I expect a significant performance gain using the same pipeline on a GPU.

* Overlapping vehicles. The current algorithm for drawing bounding boxes is very simple: look at the regions marked by scipy.ndimage.measurements.label() and find the box that encloses the whole region. When images of vehicles overlap, this results in a single large box that covers both vehicles. Perhaps more sophisticated / computationally intensive algorithms can handle this.

* Front of the vehicles, and vehicles in the opposite directions. The GTI dataset mostly captures the rear of vehicles, so the classifier trained on it does a better job detecting rear than front. This is not ideal because when vehicles enter the frame we typically see their front first. To improve this, we can try to collect more diverse images, as well as data augmentation.

* Classifier performance. Through validation, I decided on a model that achieved about 0.995 accuracy on the given dataset. However, given that we are making a lot of predictions over many frames, even a 0.5% error rate can be problematic. 
