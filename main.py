from moviepy.editor import VideoFileClip

from utils import *
from tracker import Tracker


# The pipeline function that calls helper function detect() to
# detect vehile-in boxes in the input image, and then calls track method of
# a tracker instance to merge multiple detections and remove false positives, 
# eventually calls another helper function to draw bounding boxes
# and returns the resulting image.
def process_image(img):
    model_file = "svc.p"
    ystart = 400
    ystop = 656
    scale_list = [1.5, 2]
    # Detect vehicle-in boxes in single image frame
    box_list = detect(img, model_file, ystart, ystop, scale_list)
    # Merge multiple detections and remove false positives
    labels = gtracker.track(img, box_list)
    # Draw bounding boxes
    if labels is not None:
        img = draw_bounding_boxes(img, labels)

    return img


# Set up a heat map tracker instance that will process
# a sequence of heat maps coming out from video frames
# and return labels of bounding boxes using its methods.
single_heat_thresh = 0
frame_factor = 10
multiple_heat_thresh = 7

global gtracker
gtracker = Tracker(single_heat_thresh, frame_factor, multiple_heat_thresh)

# Set iutput and output video
# input_video = "test_video.mp4"
# output_video = "test_video_output.mp4"
input_video = "project_video_lane.mp4"
output_video = "project_video_lane_output_2scales.mp4"

# Process the video
clip = VideoFileClip(input_video)
video_clip = clip.fl_image(process_image)
video_clip.write_videofile(output_video, audio=False)

