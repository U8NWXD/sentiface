
# coding: utf-8

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
from keras.utils import customobjectscope

import cognitive_face as CF
from secrets import * #You need to supply KEY and BASE_URL variables from this file!

def setup_CF():
    key = KEY# Replace with a valid subscription key (keeping the quotes in place).
    CF.Key.set(key)

    base_url = BASE_URL  # Replace with your regional Base URL
    CF.BaseUrl.set(base_url)

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
    """/

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
    x, y, w, h = box
    bb = dlib.rectangle(x, y, x + w, y + h)
    img1 = get_embedding.align.align(96, img, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

    img = img1[...,::-1]
    img = np.around(img/255.0, decimals=12)
    x_train = np.array([img])
    embedding = get_embedding.model.predict_on_batch(x_train)
    return embedding

facePredictor = '/Users/ketanagrawal/openface/models/dlib/shape_predictor_68_face_landmarks.dat'
get_embedding.align = openface.AlignDlib(facePredictor)

def load_model():
    with CustomObjectScope({'tf': tf}):
        get_embedding.model = keras.models.load_model('openface.h5', custom_objects={'triplet_loss':triplet_loss})
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

def annotate_vid(fname, skip=30, max_frames=None):
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
    i = 0
    while not max_frames or i < max_frames:
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
        i += 1

    cap.release()
    cv2.destroyAllWindows()
    # face_info.to_csv('{0}.csv'.format(vid_name))
    embeddings = np.array(embeddings)
    embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[2])
    # np.save('{0}_embeddings.npy'.format(vid_name), embeddings)
    return face_info, embeddings

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="path to video to be analyzed", type=str)
    parser.add_argument("skip", help="amount of frames to skip in between analysis frames", type=str, default=30)
    args = parser.parse_args()
    vid_path = args.video
    annotate_vid(vid_path, skip=24)

