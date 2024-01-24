import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import os
import pickle


options = vision.HandLandmarkerOptions(
  base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
  running_mode=mp.tasks.vision.RunningMode.IMAGE,
  num_hands=1)
landmarker = vision.HandLandmarker.create_from_options(options)
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


class LandMarkExtractor():

  def __init__(self) -> None:
    super().__init__()

  def load_and_detect_image(self, path):
    mp_image = mp.Image.create_from_file(path)
    detection_result = landmarker.detect(mp_image)
    return detection_result


  def extract_features(self, data):
    
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
        #We skip the last 2 joints as their distances would have already been calculated by then
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
          
          #Normalize vector so that we can use the dot product to calculate cos theta
          to_joint_vec /= np.linalg.norm(to_joint_vec)
          cos_x = np.dot(to_joint_vec, x_axis)
          cos_y = np.dot(to_joint_vec, y_axis)
          cos_z = np.dot(to_joint_vec, z_axis)
          features.extend([cos_x, cos_y, cos_z])
    return features


  def store_in_one_file(self, out_path, dir_path):
    results = []
    path_list = os.listdir(dir_path)
    num_processed = 0
    skipped = 0
    for path in path_list:
      try:
        
        detection_result = self.load_and_detect_image(os.path.join(dir_path, path))
        detection_result.hand_world_landmarks[0] 
        features = self.extract_features(detection_result.hand_world_landmarks[0])
        results.append(features)
      except IndexError as ie:
        print("Failed to process: " , path)
        skipped = skipped + 1
      num_processed = num_processed + 1
      if num_processed % 50 == 0:
        print("Processed {0} images out of {1}".format(num_processed, len(path_list)))
    print("Successfully processed {0} out of {1}".format(num_processed - skipped, len(path_list)))
    with open(out_path, 'wb') as outp:
        pickle.dump(results, outp, pickle.HIGHEST_PROTOCOL)



  def generate_dataset(self, data_path, save_path):
  
    for letter in letters:
      feature_data_path = os.path.join(data_path, letter)
      if not os.path.exists(save_path):
        os.makedirs(save_path)
      feature_path = os.path.join(save_path,f"{letter}_vec_features.pkl")

      self.store_in_one_file(feature_path, feature_data_path)


lme = LandMarkExtractor()
lme.generate_dataset(os.path.join(os.getcwd(), 'train'), os.path.join(os.getcwd(), 'train_LM'))
lme.generate_dataset(os.path.join(os.getcwd(), 'test'), os.path.join(os.getcwd(), 'test_LM'))
print("Done")
