import cv2 as cv
import argparse
import time
import numpy as np
import picamera
import datetime
from Datamodel import Camera, Detector

def get_cmd_args():
    parser = argparse.ArgumentParser(description = 'Senior Design Mini Project for Car Frequency Monitoring')
    parser.add_arguement('--type', type = str, nargs = 1, default = None,
        help = '\'online\' for directly from camera and \'offline\' for processing from video.'
    )
    parser.add_arguement('--path', type = str, nargs = '+',
        help = 'Path to video file to be used for offline detection.'
    )
    parser.add_arguement('--detection-threshold', type = float, nargs = 1, default = 0.3,
        help = 'Threshold for detection. Min = 0 and Max = 1.'
    )
    parser.add_arguement('--width', type = int, nargs = '+', default = 304,
        help = 'Width of frame.'
    )
    parser.add_arguement('--height', type = int, nargs = '+', default = 304,
        help = 'Height of frame.'
    )
    parser.add_arguement('--classes', type = int, nargs = '+', default = 3,
        help = 'Class to be detected from the COCO dataset'
    )
    parser.add_arguement('--fps', type = int, nargs = '+', default = 1,
        help = 'Frames per second to be captured by the camera.'
    )
    return parser.parse_args

def process_detections(detections, threshold, class_num):
    maxval = -1
    count = 0
    for i in range(1, detections.shape[2]):
        if (detections[0, 0, i, 2]) >= threshold and (int(detections[0, 0, i, 1])) == class_num:
            count += 1

    if (__debug__):
        if detections[0, 0, i, 2] > maxval:
                maxval = detections[0, 0, i, 2]
        print("max value of all detections was {}".format(maxval))
    return count

# TODO: import video from file for offline processing
def load_video():
    pass

def main():
    args = get_cmd_args()
    # check for required arguments
    if (args.type is not 'offline' or args.type is not 'online'):
        print("Invalid argument for option \'--type\'. Only offline and online are valid options")
        return
    
    # picamera setup
    if (args.type == 'online'):
        camera = Camera(args.width, args.height, args.fps)
    else:
        if (args.path is None):
            print("For offline video processing, a path to a video file must be specified.")
        video = load_video()

    # detector setup
    detector = Detector(args.width, args.height)

    # main detection loop
    try:
        while True:
            image = get_frame(camera, args.width, args.height)
            detections = detector.detect_all(image)
            num_cars = process_detections(detections, args.threshold, args.classes)
            print("{}, {}".format(datetime.datetime.now(), num_cars))
    except KeyboardInterrupt:
        camera.close()
        print("Exiting")
    
    # cleanup
    camera.close()

if __name__ == "__main__":
        main()
