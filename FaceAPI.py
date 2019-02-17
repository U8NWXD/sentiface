
# coding: utf-8

# # Face detection using Cognitive Services
# This walkthrough shows you how to use the cognitive services [Face API](https://azure.microsoft.com/services/cognitive-services/face/) to detect faces in an image. The API also returns various attributes such as the gender and age of each person. The sample images used in this walkthrough are from the [How-Old Robot](http://www.how-old.net) that uses the same APIs.
#
# You can run this example as a Jupyter notebook on [MyBinder](https://mybinder.org) by clicking on the launch Binder badge:
#
# [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/Microsoft/cognitive-services-notebooks/master?filepath=FaceAPI.ipynb)
#
# For more information, see the [REST API Reference](https://westus.dev.cognitive.microsoft.com/docs/services/563879b61984550e40cbbe8d/operations/563879b61984550f30395236).
#
# ## Prerequisites
# You must have a [Cognitive Services API account](https://docs.microsoft.com/azure/cognitive-services/cognitive-services-apis-create-account) with **Face API**. The [free trial](https://azure.microsoft.com/try/cognitive-services/?api=face-api) is sufficient for this quickstart. You need the subscription key provided when you activate your free trial, or you may use a paid subscription key from your Azure dashboard.
#
# ## Running the walkthrough
# To continue with this walkthrough, replace `subscription_key` with a valid subscription key.

# In[10]:


import os
import pickle
# from io import BytesIO

import argparse
import cv2
import dlib
import h5py
import keras
import matplotlib.pyplot as plt
import numpy as np
import openface
import pandas as pd
import requests
import tensorflow as tf
# from IPython.display import HTML
# from keras import backend as K
# from keras import optimizers
# from keras.engine.topology import Layer
# from keras.layers import Activation, Conv2D, Input, ZeroPadding2D, concatenate
# from keras.layers.core import Dense, Flatten, Lambda
# from keras.layers.merge import Concatenate
# from keras.layers.normalization import BatchNormalization
# from keras.layers.pooling import AveragePooling2D, MaxPooling2D
# from keras.models import Model, Sequential, load_model
# from keras.utils import CustomObjectScope
# from matplotlib import patches
# from numpy import genfromtxt
# from PIL import Image

import cognitive_face as CF
# import utils
# from utils import LRN2D

import secrets
# In[5]:


# subscription_key = "ca1c98d3e8ab4862b4115ba63e5a205b"
# assert subscription_key


# Next, verify `face_api_url` and make sure it corresponds to the region you used when generating the subscription key. If you are using a trial key, you don't need to make any changes.

# In[6]:


# face_api_url = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0/detect'


# Here is the URL of the image. You can experiment with different images  by changing ``image_url`` to point to a different image and rerunning this notebook.

# In[7]:


# image_url = 'https://how-old.net/Images/faces2/main007.jpg'


# In[65]:


def setup_CF():
    # KEY = '3b185944248044f78dce243df63b6d94'  # Replace with a valid subscription key (keeping the quotes in place).
    # CF.Key.set(KEY)

    # BASE_URL = 'https://westus.api.cognitive.microsoft.com/face/v1.0/'  # Replace with your regional Base URL
    # CF.BaseUrl.set(BASE_URL)
    key = KEY# Replace with a valid subscription key (keeping the quotes in place).
    CF.Key.set(key)

    base_url = BASE_URL  # Replace with your regional Base URL
    CF.BaseUrl.set(base_url)

# # # You can use this example JPG or replace the URL below with your own URL to a JPEG image.
# image_url = 'https://how-old.net/Images/faces2/main007.jpg'
# img_url = 'https://raw.githubusercontent.com/Microsoft/Cognitive-Face-Windows/master/Data/detection1.jpg'
# faces = CF.face.detect('temp.jpg', attributes='emotion,age,gender,smile')
# print(faces)
# img = cv2.imread('temp.jpg')
# bb = faces[0]['faceRectangle']
# emotion_scores = faces[1]['faceAttributes']['emotion']
# x, y, w, h = bb['left'], bb['top'], bb['width'], bb['height']
# print(x, y, w, h)
# print(img.shape)
# plt.imshow(img[y:y+h,x:x+w])


# In[1]:



# get_ipython().magic(u'matplotlib inline')



# In[11]:


# keras.__version__


# In[3]:


# In[76]:


# df = pd.DataFrame([[1, 2, 3]],columns=['a', 'b', 'c'])
# df.append(pd.DataFrame([[1, 2, 3]],columns=['a', 'b', 'c']),ignore_index=True)
# df = df.append([{'a':[1,2,3,4.6],'b':2,'c':3}, {'a':[1,2,2,4],'b':2,'c':3}],ignore_index=True)
# type(df['a'].values[0])


# In[9]:


# base_model = VGG19(weights='imagenet')


# In[16]:



# get_ipython().magic(u'load_ext autoreload')
# get_ipython().magic(u'autoreload 2')

# np.set_printoptions(threshold=np.nan)


# In[58]:


# get_embedding.model.save('openface.h5')


# In[63]:




# In[61]:


#from https://github.com/obieda01/Deep-Learning-Specialization-Coursera/blob/master/Course%204%20-%20Convolutional%20Neural%20Networks/Week%204/Face%20Recognition/Face%20Recognition%20for%20the%20Happy%20House%20-%20%20v1.ipynb
def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    ### START CODE HERE ### (≈ 4 lines)
    # Step 1: Compute the (encoding) distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.squared_difference(anchor, positive))
    # Step 2: Compute the (encoding) distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.squared_difference(anchor, negative))
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = pos_dist - neg_dist + alpha
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    maxi = tf.maximum(basic_loss, 0.0)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    ### END CODE HERE ###

    return loss

def get_embedding(img, box):
#     start = time.time()
#     img1 = cv2.imread(img_path, 1)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x, y, w, h = box
#     if (x, y, w, h) == (0, 0, 0, 0):
#         return None
#     print "Face detection took %s secs" % (time.time() - start)

#     start = time.time()
#     cv2.imshow('largest face', img1[y:y+h, x:x+w])
#     cv2.waitKey()
#     if img_to_encoding.align is None:
#         facePredictor = '/Users/ketanagrawal/openface/models/dlib/shape_predictor_68_face_landmarks.dat'
#         img_to_encoding.align = openface.AlignDlib(facePredictor)
#     print "Face alignment part 1 took %s secs" % (time.time() - start)
#     s = time.time()
    bb = dlib.rectangle(x, y, x + w, y + h)
    img1 = get_embedding.align.align(96, img, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
#     print "Face alignment part 2 took %s secs" % (time.time() - s)
#     print "Face alignment took %s secs" % (time.time() - start)

    img = img1[...,::-1]
    img = np.around(img/255.0, decimals=12)
    x_train = np.array([img])
    embedding = get_embedding.model.predict_on_batch(x_train)
#     print "Forward pass took %s secs" % (time.time() - start)
    return embedding

facePredictor = '/Users/ketanagrawal/openface/models/dlib/shape_predictor_68_face_landmarks.dat'
get_embedding.align = openface.AlignDlib(facePredictor)

def load_model():
    with CustomObjectScope({'tf': tf}):
        get_embedding.model = load_model('openface.h5', custom_objects={'triplet_loss':triplet_loss})
    get_embedding.model.compile(optimizer='adam', loss=triplet_loss, metrics = ['accuracy'])


# In[68]:


def create_row(face, vid_name, frame_num, frame):
    row = {}
    row['vid_name'] = vid_name
    row['frame'] = frame_num

    bb = face['faceRectangle']
    row['x'], row['y'], row['w'], row['h'] = bb['left'], bb['top'], bb['width'], bb['height']

    attr = face['faceAttributes']
    row['gender'] = attr['gender']
    row['age'] = attr['age']
    row['smile'] = attr['smile']

    for emotion in ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']:
        row[emotion] = attr['emotion'][emotion]
#     row['embedding'] = get_embedding(frame, box)

    return pd.DataFrame(row,index=[0])

def annotate_vid(fname, skip=30):
    embeddings = []
    vid_name = fname.split('.')[0]
    cap = cv2.VideoCapture(fname)
    info = []
    face_info = pd.DataFrame(columns = ['vid_name',
                                        'frame',
                                        'x', 'y', 'w', 'h', #bounding box
                                        'gender',
                                        'age',
                                        'smile',
                                        'anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise', #emotions
                                        ]) #Openface embedding
    frame_num = 0
    while True:
        # Capture frame-by-frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        if frame_num % (skip*10) == 0:
            print('Frame {0} of {1}'.format(frame_num,cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite('temp.jpg', frame)
        faces = CF.face.detect('temp.jpg', attributes='emotion,age,gender,smile')
        rows = [create_row(face, vid_name, frame_num, frame) for face in faces]
        for face in faces:
            bb = face['faceRectangle']
            box = (bb['left'], bb['top'], bb['width'], bb['height'])
            embeddings.append(get_embedding(frame, box))
        if len(rows):
            face_info = face_info.append(rows, ignore_index=True)
        frame_num += skip #skip over designated # of frames

    cap.release()
    cv2.destroyAllWindows()
    # face_info.to_csv('{0}.csv'.format(vid_name))
    embeddings = np.array(embeddings)
    embeddings = embeddings.reshape(embeddings[0], embeddings[2])
    # np.save('{0}_embeddings.npy'.format(vid_name), embeddings)
    return face_info, embeddings

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="path to video to be analyzed", type=str)
    parser.add_argument("skip", help="amount of frames to skip in between analysis frames", type=str, default=30)
    args = parser.parse_args()
    vid_path = args.video
    annotate_vid(vid_path, skip=24)

# When everything done, release the capture


# In[69]:


# df = pd.read_csv('sentiface/reactionvideo.csv')
# print(len(df))
# df


# The next few lines of code call into the Face API to detect the faces in the image. In this instance, the image is specified via a publically visible URL. You can also pass an image directly as part of the request body. For more information, see the [API reference](https://westus.dev.cognitive.microsoft.com/docs/services/563879b61984550e40cbbe8d/operations/563879b61984550f30395236).

# In[23]:



# headers = { 'Ocp-Apim-Subscription-Key': subscription_key }

# params = {
#     'returnFaceId': 'true',
#     'returnFaceLandmarks': 'false',
#     'returnFaceAttributes': 'age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise',
# }

# response = requests.post(face_api_url, params=params, headers=headers, json={"url": image_url})
# faces = response.json()
# HTML("<font size=5>Detected <font color='blue'>%d</font> faces in the image</font>"%len(faces))


# Finally, the face information can be overlaid of the original image using the `matplotlib` library in Python.

# In[16]:


# get_ipython().magic(u'matplotlib inline')


# response = requests.get(image_url)
# image = Image.open(BytesIO(response.content))

# plt.figure(figsize=(8,8))
# ax = plt.imshow(image, alpha=0.6)
# for face in faces:
#     fr = face["faceRectangle"]
#     fa = face["faceAttributes"]
#     origin = (fr["left"], fr["top"])
#     p = patches.Rectangle(origin, fr["width"], fr["height"], fill=False, linewidth=2, color='b')
#     ax.axes.add_patch(p)
#     plt.text(origin[0], origin[1], "%s, %d"%(fa["gender"].capitalize(), fa["age"]), fontsize=20, weight="bold", va="bottom")
# _ = plt.axis("off")


# Here are more images that can be analyzed using the same technique.
# First, define a helper function, ``annotate_image`` to annotate an image given its URL by calling into the Face API.

# In[6]:


# def annotate_image(image_url):
#     response = requests.post(face_api_url, params=params, headers=headers, json={"url": image_url})
#     faces = response.json()

#     image_file = BytesIO(requests.get(image_url).content)
#     image = Image.open(image_file)

#     plt.figure(figsize=(8,8))
#     ax = plt.imshow(image, alpha=0.6)
#     for face in faces:
#         fr = face["faceRectangle"]
#         fa = face["faceAttributes"]
#         origin = (fr["left"], fr["top"])
#         p = patches.Rectangle(origin, fr["width"],                               fr["height"], fill=False, linewidth=2, color='b')
#         ax.axes.add_patch(p)
#         plt.text(origin[0], origin[1], "%s, %d, %s"%(fa["gender"].capitalize(), fa["age"], fa["emotion"]),                  fontsize=20, weight="bold", va="bottom")
#     plt.axis("off")


# You can then call ``annotate_image`` on other images. A few examples samples are shown below.

# In[7]:


# annotate_image("https://how-old.net/Images/faces2/main001.jpg")


# # In[11]:


# annotate_image("https://how-old.net/Images/faces2/main002.jpg")


# # In[12]:


# annotate_image("https://how-old.net/Images/faces2/main004.jpg")


# # In[13]:


# annotate_image("https://previews.123rf.com/images/kozzi/kozzi1301/kozzi130100010/17106684-close-up-image-of-angry-face-of-man-isolated-on-a-white-background.jpg")


# # In[14]:


# annotate_image("https://petapixel.com/assets/uploads/2014/04/faces1.jpg")
