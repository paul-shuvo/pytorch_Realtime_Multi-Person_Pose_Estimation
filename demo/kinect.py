from freenect2 import Device, FrameType
import cv2
import numpy as np
import time
def callback(type_, frame):
    print(f'{type_}, {frame.format}') 
    if type_ is FrameType.Color: # FrameFormat.BGRX
        rgb = frame.to_array().astype(np.uint8)
        cv2.imshow('rgb', rgb[:,:,0:3])
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
            # device.stop()
                break

device = Device()
device.start()
t = True
while True:
    if t:
        t = False
        time.sleep(2)
    type_, frame = device.get_next_frame()
    if type_ is FrameType.Color: # FrameFormat.BGRX
        rgb = frame.to_array().astype(np.uint8)
        cv2.imshow('rgb', rgb[:,:,0:3])
        # device.stop()
        # device.start()
# while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        device.stop()
        break