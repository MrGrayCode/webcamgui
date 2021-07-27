from tkinter import *
import cv2
from threading import Thread
from imutils.video import VideoStream
from imutils.video import FPS
from imutils import resize
import PIL.Image, PIL.ImageTk
import time
from datetime import datetime

class VideoFeed:
    '''
    Class for reading video stream
    Uses a seperate thread for the same by using the imutils library
    '''
    def __init__(self, video_source = 0):
        self.vid = VideoStream(usePiCamera = False).start()
        time.sleep(5)
        if not self.vid.stream.isOpened():
            raise ValueError("Unable to Open Video Source", video_source)
        self.width = self.vid.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def _isOpened(self):
        #check whether the camera is opened
        return self.vid.stream.isOpened()
    
    

    def get_frame(self):
        '''
        Reads image from camera and returns in RGB format
        '''
        if self._isOpened():
            frame = self.vid.read()
            frame = resize(frame, width = 640, height = 480)
            return (True, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            return (False, None)
        
    def close(self):
        if self._isOpened():
            self.vid.stream.release()