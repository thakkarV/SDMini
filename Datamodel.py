import cv2 as cv
import time
import numpy as np
import picamera
import datetime

class Detector(object):
    def __init__(
        self,
        width,
        height,
    ):
        self.width = width
        self.height = height
        self.net = cv.dnn.readNetFromTensorflow('mobilenetv2_frozen_graph.pb', 'mobilenetv2_ssdlite.pbtxt')
    
    def detect_all(self, image):
        rows = image.shape[0]
        cols = image.shape[1]
        self.net.setInput(
            cv.dnn.blobFromImage(
                image,
                size = (self.width, self.height),
                swapRB = False,
                crop = False
            )
        )
        return self.net.forward()

class Camera(object):
    def __init__(self, frame_width, frame_height, fps):
        self.camera = picamera.PiCamera()
        self.resolution = (frame_width, frame_height)
        self.framerate = fps
        self.start_preview()
        time.sleep(2)

    def get_frame(self, width, height):
        image = np.empty((width * height * 3,), dtype = np.uint8)
        self.camera.capture(image, format = 'bgr', use_video_port = True)
        image = image.reshape((width, height, 3))
        return image

    def close(self):
        self.camera.close()
