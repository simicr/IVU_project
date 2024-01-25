# Project for Image and Video Understanding

## Dataset

First download the dataset from [Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet/).
Then extract the dataset into the root folder of the project. The topmost folder should be **archive**.

## Installing Python Packages

The project was tested on **Python 3.10**. Mediapipe requires a Python version between 3.8 to 3.11, but be aware that versions other than 3.10 were not tested.
To install the required packages, first make sure you have [pip](https://pypi.org/project/pip/) installed and then run the following command:

>python -m pip install -r pip_requirements.txt
 
This should install all the necesery packages. Contact if problems are noticed.
## Generating a split

Once the packages and dataset have been downloaded, the next step is to generate a training/test split from the dataset. To do so, please run the <strong>dataset\.py </strong> script. 
If this is the first time running the script, you will need to modify the last line of the script and change *make_data=True*. This will consolidate all of the training and test dataset into one folder from which the split will then be generated. Additionally, the *data_ratio* parameter can be changed to specify how large the split should be from the dataset.

## Hand Landmark Generation

After generating a split, you will need to run the **dataset_generation\.py** script. This will process the images from the split and generate the respective hand landmark representation for each class by feeding the images into Mediapipe, and extracting the necessary features as specified in the report. This is the data that the SVM model will be trained and tested on. As Mediapipe will not be able to correctly process all of the images, a number of errors might show up saying the some images couldn't be processed. These errors can be ignored and will not impact performance if the data split is large enough. Around 400 examples per class worked quite well for us. The output features should be stored inside two folders called **train_LM** and **test_LM**.

**Important:** For the SVM model and dataset generation to work, the **hand_landmarker.task** file will need to be downloaded from [Mediapipe's website](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/index#models). Place this in the root folder alongside the **svm_model.py** script.

## SVM Model
To run the SVM model, the **svm_model.py** script needs to be executed. To train the model, the *train_model* parameter on line 225 will need to be set to *True*. This will train the SVM model on the previously generated split and output a model called **awesome_model.pkl**. If *train_model* is set to *False* then the script will use the model saved on disk. Afterwards it will then try to use the user's webcam using OpenCV to read video frames, which are then fed to Mediapipe, and features are extracted from the resulting hand landmarks. Finally, these are used to generate a classification which is shown on-screen. This process is done every frame. To kill process pres 'q' and to make a screenshot press 's'. 

## CNN Model
To run the CNN model, first make the data split that you want to use for training. The training is done in **cnn_model.py** locally or **cnn_model.ipynb** on Drive. If it is *True* upload the data on the root of your Drive.

*train_model*  needs to be set to *True* for training and *False* otherwise. *define_model* can be changed as wanted.

the gotten **asl_model.pth** should then be moved to the local root folder. To test out this model run **cnn_realtime.py**. 