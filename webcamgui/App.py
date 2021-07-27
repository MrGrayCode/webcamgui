from tkinter import *
import cv2
from threading import Thread
from imutils.video import VideoStream
from imutils import resize
import PIL.Image, PIL.ImageTk
import time
from datetime import datetime
import os
from pathlib import Path

import torch

from webcamgui.VideoFeed import VideoFeed

class App:
    def __init__(self, window, fps, model, transforms, window_title, video_source = 0, class_names = None):
        # torch variables
        self.model = model
        self.transforms = transforms

        # class names
        self.class_names = class_names

        #Creating the Window
        self.window = window
        self.window.title(window_title)
        self.window.config(bg = "#eaeae1")
        
        #Video source
        self.video_source = video_source
        self.vid = VideoFeed()
        self.center = [200, self.vid.height//2]

        #path to save snapshots
        self.path = "snapshots/snapshot-"
        Path('snapshots').mkdir(exist_ok=True)

        #Layout
        self.left = Frame(self.window, borderwidth = 2, relief = "solid", bg = "#ccccb3")
        self.right = Frame(self.window, borderwidth = 2, relief = "solid", bg = "#ccccb3")
        self.buttons_left = Frame(self.window, borderwidth = 2, relief = "solid", bg = "#e0e0d1")
        self.buttons_right = Frame(self.window, borderwidth = 2, relief = "solid", bg = "#e0e0d1")

        self.box_left = Frame(self.left, borderwidth = 2, relief = "solid", bg = "#e0e0d1")
        self.box_right = Frame(self.right, borderwidth = 2, relief = "solid", bg = "#e0e0d1")
        self.camera_buttons = Frame(self.buttons_left, borderwidth = 2, relief = "solid", bg = "#e0e0d1")
        self.app_buttons = Frame(self.buttons_right, borderwidth = 2, relief = "solid", bg = "#e0e0d1")

        self.canvas_left = Canvas(self.box_left, width = self.vid.width, height = self.vid.height, bg = "#e0e0d1")
        self.label_left = Label(self.left, text = "Camera Feed", font = ("MS Sans", 12, "bold"), bg = "#ccccb3")
        self.canvas_right = Canvas(self.box_right, height = self.vid.height, bg = "#e0e0d1")
        self.label_right = Label(self.right, text = "Output", font = ("MS Sans", 12, "bold"), bg = "#ccccb3")
        
        #Buttons
        self.snap_button = Button(self.camera_buttons, text = "Snapshot", command = self.snapshot, font = ("MS Sans", 12,"bold"), bg = "#004080", fg = "#ffffff")
        self.exit_button = Button(self.app_buttons, text = "Exit", command = self.exit, font = ("MS Sans", 12,"bold"), bg = "#ff3333", fg = "#ffffff")

        self.left.grid(row = 0, column = 0, padx = 2, pady = 2 )
        self.right.grid(row = 0, column = 1, padx = 2, pady = 2)
        self.buttons_left.grid(row = 1, column = 0, padx = 2, pady = 2)
        self.buttons_right.grid(row = 1, column = 1, padx = 2, pady = 2)
        self.box_left.grid(row = 0, column = 0, padx = 10, pady = 10)
        self.box_right.grid(row = 0, column = 0, padx = 10, pady = 10)
        
        self.camera_buttons.grid(row = 0, column = 0, padx = 278, pady = 10)
        self.app_buttons.grid(row = 0, column = 1, padx = 170, pady = 10)

        self.snap_button.grid(row = 0, column = 4)
        self.exit_button.grid(row = 0, column = 0)

        self.canvas_left.grid(row = 0, column = 0)
        self.label_left.grid(row = 1, column = 0)
        self.canvas_right.grid(row = 0, column = 0)
        self.label_right.grid(row = 1, column = 0)

        self.delay = 5
        self.update(fps)

    def current_frame(self):
        ret, frame = self.vid.get_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def snapshot(self):
        '''
        Callback function for the snapshot button
        Saves the image to path
        '''
        snap = self.current_frame()
        timestamp = "{:%Y%m%d-%H%M%S}".format(datetime.now())
        snap_path = self.path + timestamp + ".png"
        cv2.imwrite(snap_path, snap)
        print("[INFO] Saved Snapshot", snap_path)
    
    def handle_output_screen(self, canvas, text="Model output here"):
        height = canvas.winfo_height()
        width = canvas.winfo_width()
        canvas.create_text(width//2,height//2,font="Times 20 italic bold",text=text)
    
    def get_predictions(self, input):
        input = self.transforms(input)
        input = input.unsqueeze(0)

        predictions = self.model(input)
        class_id = torch.argmax(predictions).item()
        if self.class_names:
            class_id = self.class_names[class_id]
            # class_id = class_id.split(',')
        else:
            class_id = str(class_id)
        return class_id

    def update(self,fps):
        '''
        Updates the canvas in the window
        Also the FPS is updated after each execution of this function
        '''
        ret, frame = self.vid.get_frame()
        #sensor_data = self.sensor.get_data()
        self.canvas_left.delete("all")
        self.canvas_right.delete("all")
        if ret:
            fps.update()
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas_left.create_image(0, 0, image = self.photo, anchor = NW)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            class_id = self.get_predictions(frame)
            self.handle_output_screen(self.canvas_right, text=class_id)

        self.window.after(self.delay, self.update, fps)
    
    def exit(self):
        self.vid.close()
        time.sleep(2.0)
        self.window.destroy()
