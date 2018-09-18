import cv2 as cv
import argparse
import time
import numpy as np
import picamera

class Detector(object):
    def __init__(
        self,
        width,
        height
    ):
        self.width = width
        self.height = height
        self.net = cv.dnn.readNetFromTensorflow('ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb', 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt')
    
    def detect_all(self, image):
        rows = image.shape[0]
        cols = image.shape[1]
        self.net.setInput(
            cv.dnn.blobFromImage(
                image,
                size = (self.width, self.height),
                swapRB = False,
                crop=False
            )
        )
        return self.net.forward()

def get_frame(camera, width, height):
    image = np.empty((width * height * 3,), dtype=np.uint8)
    camera.capture(image, format='bgr', use_video_port=True)
    image = image.reshape((width, height, 3))
    return image

def process_detections(detections):
    maxval = -1
    for i in range(1, detections.shape[2]):
        if (detections[0, 0, i, 2]) > 0.5:
            print(detections[0, 0, i, 1])

        if detections[0, 0, i, 2] > maxval:
                maxval = detections[0, 0, i, 2]
    
    print("max value of all detections was {}".format(maxval))
    
def main():

    # image capture metadata
    fps = 1
    frame_width = 300
    frame_height = 300

    # picamera setup
    camera = picamera.PiCamera()
    camera.resolution = (frame_width, frame_height)
    camera.framerate = fps
    camera.start_preview()
    time.sleep(2)

    # detector setup
    detector = Detector(frame_width, frame_height)

    while True:
        image = get_frame(camera, frame_width, frame_height)
        detections = detector.detect_all(image)
        num_cars = process_detections(detections)
        break

    # cleanup
    camera.close()


if __name__ == "__main__":
        main()
