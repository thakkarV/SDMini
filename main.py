import cv2 as cv
import argparse
import time
import numpy as np
import picamera
import io

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
                swapRB = True,
                crop=False
            )
        )
        return self.net.forward()


def get_frame():
    with picamera.PiCamera() as camera:
        stream = io.BytesIO()
        camera.resolution = (300, 300)
        camera.start_preview()
        time.sleep(2)
        camera.capture(stream, format='jpeg')
        # Construct a numpy array from the stream
        data = np.fromstring(stream.getvalue(), dtype = np.uint8)
        # "Decode" the image from the array, preserving colour
        image = cv.imdecode(data, 1)
        cv.imwrite("capture.jpeg", image)
        # OpenCV returns an array with data in BGR order. If you want RGB instead
        # use the following...
        image = image[:, :, ::-1]

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
    detector = Detector(300, 300)

    while True:
        image = get_frame()
        detections = detector.detect_all(image)
        num_cars = process_detections(detections)
        break

if __name__ == "__main__":
        main()







