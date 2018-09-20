import cv2 as cv
import argparse
import time
import numpy as np
import picamera
import datetime
from Datamodel import Camera, Detector

def get_cmd_args():
    parser = argparse.ArgumentParser(description = 'Senior Design Mini Project for Car Frequency Monitoring')
    parser.add_argument('--type', type = str, default = None,
        help = '\'online\' for directly from camera and \'offline\' for processing from video.'
    )
    parser.add_argument('--path', type = str, nargs = '+', default = None,
        help = 'Path to video file to be used for offline detection.'
    )
    parser.add_argument('--threshold', type = float, default = 0.3,
        help = 'Threshold for detection. Min = 0 and Max = 1.'
    )
    parser.add_argument('--width', type = int, nargs = '+', default = 304,
        help = 'Width of frame.'
    )
    parser.add_argument('--height', type = int, nargs = '+', default = 304,
        help = 'Height of frame.'
    )
    parser.add_argument('--classes', type = int, nargs = '+', default = 3,
        help = 'Class to be detected from the COCO dataset'
    )
    parser.add_argument('--fps', type = int, default = 1,
        help = 'Frames per second to be captured by the camera.'
    )
    return parser.parse_args()

def process_detections(detections, threshold, class_num):
    count = 0
    for i in range(1, detections.shape[2]):
        if (detections[0, 0, i, 2]) >= threshold and (int(detections[0, 0, i, 1])) == class_num:
            count += 1

    return count

def load_video(path):
    return cv.VideoCapture(path)

def main():
    args = get_cmd_args()
    # check for required arguments
    if (args.type != 'offline' and args.type != 'online'):
        print("Invalid argument for option \'--type\'. Only offline and online are valid options")
        return
    
    # picamera setup
    if (args.type == 'online'):
        camera = Camera(args.width, args.height, args.fps)
    else:
        if (args.path is None):
            print("For offline video processing, a path to a video file must be specified.")
        video = load_video(args.path[0])

    # detector setup
    detector = Detector(args.width, args.height)

    # main detection loop
    try:
        img_number = 0
        while True:
            
            # run ssd on capture from camera
            if (args.type == 'online'):
                image = camera.get_frame(args.width, args.height)
                cv.imwrite('img{}.jpg'.format(img_number), image)
                img_number += 1
            
            # run ssd on a frame from video
            else:
                if video.isOpened():
                    ret, image = video.read()
                    if not ret:
                        print('Error reading video.')
                        break
                else:
                    print('Reached end of video.')
                    break
            detections = detector.detect_all(image)
            num_cars = process_detections(detections, args.threshold, args.classes)
            print("{}, {}".format(datetime.datetime.now(), num_cars))
    except KeyboardInterrupt:
        
        if args.type == 'offline':
            video.release()
        else:
            camera.close()
    
    # cleanup
    print("Exiting")
    if args.type == 'offline':
        video.release()
    else:
        camera.close()

if __name__ == "__main__":
        main()
