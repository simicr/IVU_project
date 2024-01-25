import pickle
import numpy as np
import cv2
import os
import datetime
import mediapipe as mp

from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class SVMModel:
    
    def __init__(self):
      super()
      self.cwd = os.getcwd()
      self.ok = False
      self.test_dir = os.path.join('archive', 'asl_alphabet_test', 'asl_alphabet_test')
      self.train_dir = os.path.join('archive', 'asl_alphabet_train', 'asl_alphabet_train')
      self.letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
      
      self.model = svm.LinearSVC(dual="auto", max_iter=10000, C=0.01)
      self.scaler = StandardScaler()


    def load_model(self, path):
      with open(path, 'rb') as inp:
        modelFile = pickle.load(inp)
        self.model = modelFile[0]
        self.scaler = modelFile[1]

    def save_model(self, path):
      with open(path, 'wb') as outp:
        pickle.dump([self.model, self.scaler], outp, pickle.HIGHEST_PROTOCOL)

    def open_features(self, path):
      all_features = None
      with open(path, 'rb') as inp:
        all_features = pickle.load(inp)
      return all_features

    def extract_features_from_landmarks(self, landmarks):
      data = landmarks 
      total_joints = 21
      features = []
      joint1_indices = [3,4,6,7,8,10,11,12,14,15,16,18,19,20,21]
      joint1_point = np.array([data[0].x, data[0].y, data[0].z])
      for idx in joint1_indices:
        landmark = data[idx-1]
        point3d = np.array([landmark.x, landmark.y, landmark.z])
        distance = np.linalg.norm(point3d - joint1_point)
        features.append(distance)
      
      starting_indices_of_subsequent_joints = [4,5,6,6,8,9,10,10,12,13,14,14,16,17,18,18,20,21]
      next_joint_idx = 1
      
      for start_idx in starting_indices_of_subsequent_joints:
        #we skip the last 2 joints as their distances would have already been calculated by then
        if next_joint_idx < 19:
          joint_landmark = data[next_joint_idx]
          joint_point = np.array([joint_landmark.x, joint_landmark.y, joint_landmark.z])
          for idx in range(start_idx, total_joints + 1):
            landmark = data[idx-1]
            point3d = np.array([landmark.x, landmark.y, landmark.z])
            distance = np.linalg.norm(point3d - joint_point)
            features.append(distance)
          next_joint_idx = next_joint_idx + 1
      
      
      #Angle features
      x_axis = np.array([1,0,0]) 
      y_axis = np.array([0,1,0]) 
      z_axis = np.array([0,0,1]) 
      for start_idx in range(0, 21):
        for idx in range(start_idx + 1, 21):
          from_joint_landmark = data[start_idx]
          from_joint_point = np.array([from_joint_landmark.x, from_joint_landmark.y, from_joint_landmark.z])
          to_joint_landmark = data[idx]
          to_joint_point = np.array([to_joint_landmark.x, to_joint_landmark.y, to_joint_landmark.z])
          to_joint_vec = to_joint_point - from_joint_point
          
          to_joint_vec /= np.linalg.norm(to_joint_vec)
          cos_x = np.dot(to_joint_vec, x_axis)
          cos_y = np.dot(to_joint_vec, y_axis)
          cos_z = np.dot(to_joint_vec, z_axis)
          features.extend([cos_x, cos_y, cos_z])
      return features
    
    def scale_features(self, features):
      scaled_features = self.scaler.transform(features[:][:, 0 : 190]) # First 190 features are distance features - need to normalize
      scaled_features = np.hstack((scaled_features, features[:][:, 190 : ])) # Rest are angle based features, no normalization needed since they are no affected by hand size
      return scaled_features

    def fit_and_scale_features(self, features):
      scaled_features = self.scaler.fit_transform(features[:][:, 0 : 190]) # First 190 features are distance features - need to normalize
      scaled_features = np.hstack((scaled_features, features[:][:, 190 : ])) # Rest are angle based features, no normalization needed since they are no affected by hand size
      return scaled_features
       
    
    def train_model(self, train_path = "train"):
      all_features = []
      all_targets = []

      # Load features
      for letter in self.letters:
        feature_dir = os.path.join(train_path, f'{letter}_vec_features.pkl')
        features = self.open_features(feature_dir)
        target = np.full(len(features), letter)
        if letter == 'A':
          all_features = features
          all_targets = target
        else:    
          all_features = np.concatenate((all_features, features), axis=0)
          all_targets = np.concatenate((all_targets, target), axis=0)
    
      
      scaled_features = self.fit_and_scale_features(all_features)
      start_time = datetime.datetime.now()
      self.model.fit(scaled_features, all_targets)
      timestamp_ms = datetime.datetime.now() - start_time
      timestamp_ms = int(timestamp_ms.total_seconds() * 1000)
      print(f"Training took {timestamp_ms}ms")

    def test_model(self, test_path = "test"):
      all_features = []
      all_targets = []

      # load features
      for letter in self.letters:
        feature_dir = os.path.join(test_path, f'{letter}_vec_features.pkl')
        features = self.open_features(feature_dir)
        target = np.full(len(features), letter)
        if letter == 'A':
          all_features = features
          all_targets = target
        else:    
          all_features = np.concatenate((all_features, features), axis=0)
          all_targets = np.concatenate((all_targets, target), axis=0)
    
      test_data = self.scale_features(all_features)
      y_pred = self.model.predict(test_data)
      accuracy = accuracy_score(all_targets, y_pred)
      print("Accuracy: ", accuracy)
      

    def predict(self, landmarks):
      features = self.extract_features_from_landmarks(landmarks)
      scaled_features = self.scale_features(np.array(features, ndmin=2))
      return self.model.predict(scaled_features)
      



'''
video
'''
MARGIN = 10 
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) 

def landmark_callback(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
  global mediapipe_result
  global hand_detected
  if len(result.handedness) > 0:
    hand_detected = True
    mediapipe_result = result
  else:
    hand_detected = False


def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(1):
    hand_landmarks = hand_landmarks_list[idx]
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

camera = cv2.VideoCapture(0)
start_time = datetime.datetime.now()
mediapipe_result = None
hand_detected = False

options =mp.tasks.vision.HandLandmarkerOptions(
  base_options= mp.tasks.BaseOptions(model_asset_path='hand_landmarker.task'),
  running_mode= mp.tasks.vision.RunningMode.LIVE_STREAM,
  result_callback=landmark_callback,
  num_hands=1)
landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options) 




trainModel = False
if trainModel:
  model = SVMModel()
  model.train_model(os.path.join(os.getcwd(), 'train_LM'))
  model.test_model(os.path.join(os.getcwd(), 'test_LM'))
  model.save_model('awesome_model.pkl')


model = SVMModel()
model.load_model('awesome_model.pkl')
while True:
    # Read each frame from the webcam
    _, frame = camera.read()
    x , y, c = frame.shape

    # Flip the frame vertically
    #frame = cv2.flip(frame, 1)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    timestamp = datetime.datetime.now()
    timestamp_ms = timestamp - start_time
    timestamp_ms = int(timestamp_ms.total_seconds() * 1000)
    landmarker.detect_async(mp_image, timestamp_ms) 

    if hand_detected: 
      annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), mediapipe_result)
      handedness = mediapipe_result.handedness[0]
      hand = handedness[0].category_name
      landmarks = mediapipe_result.hand_world_landmarks[0]
      
      if hand == "Left":
        for i, lm in enumerate(landmarks):
          landmarks[i].x = -1 * lm.x
      
      prediction = model.predict(landmarks)
      (text_width, text_height), baseline = cv2.getTextSize(prediction[0],  cv2.FONT_HERSHEY_PLAIN, 10, 1)
      cv2.rectangle(annotated_image, (0, 0), (text_width, text_height + baseline), (0,0,0), -1)
      annotated_image = cv2.putText(annotated_image, prediction[0], (0,text_height + baseline), cv2.FONT_HERSHEY_PLAIN, 10, (0,0,255), 4)
      
      if cv2.waitKey(1) == ord('s'): 
        no_points = cv2.rectangle(np.array(mp_image.numpy_view()), (0, 0), (text_width, text_height + baseline), (0,0,0), -1) 
        cv2.putText(no_points, prediction[0], (0,text_height + baseline), cv2.FONT_HERSHEY_PLAIN, 10, (0,0,255), 4)
        cv2.imwrite("{0}.png".format(prediction[0]), annotated_image)
        cv2.imwrite("{0}_no_points.png".format(prediction[0]), no_points)
      cv2.imshow("Output", annotated_image)
    else:
      cv2.imshow("Output", frame)

    if cv2.waitKey(1) == ord('q'):
      break

camera.release()
cv2.destroyAllWindows()
