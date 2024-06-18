import cv2 
import mediapipe as mp 
from math import hypot 
import screen_brightness_control as sbc 
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from google.protobuf.json_format import MessageToDict

def get_available_cameras():
    camera_list = []
    for i in range(10):  # Check up to 10 devices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            camera_list.append(i)
            cap.release()

    return camera_list

def set_volume(volume):
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume_object = cast(interface, POINTER(IAudioEndpointVolume))
    volume_object.SetMasterVolumeLevelScalar(volume, None)

mpHands = mp.solutions.hands 

hands = mpHands.Hands( 
	static_image_mode=False, 
	model_complexity=1, 
	min_detection_confidence=0.75, 
	min_tracking_confidence=0.75, 
	max_num_hands=2) 

Draw = mp.solutions.drawing_utils 
 
cap = cv2.VideoCapture(get_available_cameras()[0])

while True: 
	_, frame = cap.read() 
	frame = cv2.flip(frame, 1) 
	frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
	Process = hands.process(frameRGB) 
	landmarkList_l = []
	landmarkList =[]

	if Process.multi_hand_landmarks:
 
		for handlm in Process.multi_hand_landmarks:
			for i in Process.multi_handedness:
				label = MessageToDict(i)['classification'][0]['label']
				if label == 'Left':
					# Display 'Left Hand' 
					# on Left side of window
						cv2.putText(frame, label+' Hand', (20, 50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)
						for _id, landmarks in enumerate(handlm.landmark): 
							# store height and width of image 
							height_l, width_l, color_channels_l = frame.shape 
							# calculate and append x, y coordinates 
							# of handmarks from image(frame) to lmList 
							x_l, y_l = int(landmarks.x*width_l), int(landmarks.y*height_l) 
							landmarkList_l.append([_id, x_l, y_l]) 
							# draw Landmarks 
							Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS) 
						# If landmarks list is not empty 
						if landmarkList_l != []: 
								# store x,y coordinates of (tip of) thumb 
							x_1_l, y_1_l = landmarkList_l[4][1], landmarkList_l[4][2] 

								# store x,y coordinates of (tip of) index finger 
							x_2_l, y_2_l = landmarkList_l[8][1], landmarkList_l[8][2] 

								# draw circle on thumb and index finger tip 
							cv2.circle(frame, (x_1_l, y_1_l), 7, (0, 255, 0), cv2.FILLED) 
							cv2.circle(frame, (x_2_l, y_2_l), 7, (0, 255, 0), cv2.FILLED) 

								# draw line from tip of thumb to tip of index finger 
							cv2.line(frame, (x_1_l, y_1_l), (x_2_l, y_2_l), (0, 255, 0), 3) 

								# calculate square root of the sum of 
								# squares of the specified arguments. 
							L = hypot(x_2_l-x_1_l, y_2_l-y_1_l) 

								# 1-D linear interpolant to a function 
								# with given discrete data points 
								# (Hand range 15 - 220, Brightness 
								# range 0 - 100), evaluated at length. 
							b_level_left = np.interp(L, [15, 220], [0, 100])

								# set brightness 
							sbc.set_brightness(int(b_level_left)) 					

				if label == 'Right': 
					
					# Display 'Right Hand' 
					# on Right side of window 
					cv2.putText(frame, label+' Hand', (460, 50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)
					for _id, landmarks in enumerate(handlm.landmark): 
							height, width, color_channels = frame.shape 
							x, y = int(landmarks.x*width), int(landmarks.y*height) 
							landmarkList.append([_id, x, y])  
							Draw.draw_landmarks(frame, handlm,mpHands.HAND_CONNECTIONS) 

					if landmarkList != []: 
							x_1, y_1 = landmarkList[4][1], landmarkList[4][2] 
							x_2, y_2 = landmarkList[8][1], landmarkList[8][2] 
							cv2.circle(frame, (x_1, y_1), 7, (0, 255, 0), cv2.FILLED) 
							cv2.circle(frame, (x_2, y_2), 7, (0, 255, 0), cv2.FILLED) 
							cv2.line(frame, (x_1, y_1), (x_2, y_2), (0, 255, 0), 3) 
							R = hypot(x_2-x_1, y_2-y_1) 
							b_level_right = np.interp(R, [15, 220], [0, 100])
       
							# Set volume
							set_volume(b_level_right/100)   
	# Display Video and when 'q' is entered, destroy 
	# the window 
	cv2.imshow('Image', frame) 
	if cv2.waitKey(1) & 0xff == ord('q'): 
		break
