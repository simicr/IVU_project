import torch
import mediapipe as mp
import numpy as np
import cv2
import datetime
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
from torchvision import models
import torch.nn as nn

MARGIN = 10 
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)

camera = cv2.VideoCapture(0)
start_time = datetime.datetime.now()

mediapipe_result = None
hand_detected = False

def landmark_callback(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global mediapipe_result
    global hand_detected
    if len(result.handedness) > 0:
        hand_detected = True
        mediapipe_result = result
    else:
        hand_detected = False

options =mp.tasks.vision.HandLandmarkerOptions(
  base_options= mp.tasks.BaseOptions(model_asset_path='hand_landmarker.task'),
  running_mode= mp.tasks.vision.RunningMode.LIVE_STREAM,
  result_callback=landmark_callback,
  num_hands=1)

landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options) 


def crop_image(image, mediapipe_result):

    hand_position = mediapipe_result.hand_landmarks[0] if mediapipe_result.hand_landmarks else None

    if not hand_position:
        return image

    x = max(int(min(hand_position, key=lambda item: item.x).x * image.shape[1])-100, 0)
    y = max(int(min(hand_position, key=lambda item: item.y).y * image.shape[0])-100, 0)
    w = int((max(hand_position, key=lambda item: item.x).x - min(hand_position, key=lambda item: item.x).x) * image.shape[1]) + 100
    h = int((max(hand_position, key=lambda item: item.y).y - min(hand_position, key=lambda item: item.y).y) * image.shape[0]) + 100

    roi = np.copy(image[y:y+h, x:x+w])
    roi_resized = cv2.resize(roi, (224, 224))
    return roi_resized
    

def load_model(save_path):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    
    model.fc = nn.Linear(in_features, 26)

    for param in model.parameters():
        param.requires_grad = False
        
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(save_path, map_location=torch.device('cuda')))
    else:
        model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
    return model

model = load_model('asl_model.pth')
while True:
    # Read each frame from the webcam
    _, frame = camera.read()
    x , y, c = frame.shape

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    timestamp = datetime.datetime.now()
    timestamp_ms = timestamp - start_time
    timestamp_ms = int(timestamp_ms.total_seconds() * 1000)
    landmarker.detect_async(mp_image, timestamp_ms) 

    if hand_detected: 
        cropped_image = crop_image(mp_image.numpy_view(), mediapipe_result)
        handedness = mediapipe_result.handedness[0]
        hand = handedness[0].category_name
        landmarks = mediapipe_result.hand_world_landmarks[0]

        if hand == "Left":
           for i, lm in enumerate(landmarks):
              landmarks[i].x = -1 * lm.x

        with torch.no_grad():
            input = torch.flip(torch.tensor(cropped_image).permute(2,0,1).float().unsqueeze(0), dims=(0, 1, 2)) / 255
            logits = model(input)
            prediction = chr(ord("A") + torch.argmax(logits))

        (text_width, text_height), baseline = cv2.getTextSize(prediction[0],  cv2.FONT_HERSHEY_PLAIN, 10, 1)
        cv2.rectangle(frame, (0, 0), (text_width, text_height + baseline), (0,0,0), -1)
        frame = cv2.putText(frame, prediction[0], (0,text_height + baseline), cv2.FONT_HERSHEY_PLAIN, 10, (0,0,255), 4)
        
        if cv2.waitKey(1) == ord('s'): 
            no_points = cv2.rectangle(np.array(mp_image.numpy_view()), (0, 0), (text_width, text_height + baseline), (0,0,0), -1) 
            cv2.putText(no_points, prediction[0], (0,text_height + baseline), cv2.FONT_HERSHEY_PLAIN, 10, (0,0,255), 4)
            cv2.imwrite("{0}.png".format(prediction[0]), cropped_image)

        cv2.imshow("Output", frame)
    else:
        cv2.imshow("Output", frame)

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()