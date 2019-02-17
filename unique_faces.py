import os
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import cv2
import matplotlib.pyplot as plt
from collections import Counter

def identify_faces(embeddings):
    clustering = DBSCAN().fit(embeddings)
    return clustering.labels_

def load_embeddings(filepath):
    embeddings = np.load(filepath)
    return embeddings

def load_face_info(filepath):
    df = pd.read_csv(filepath)
    return df

def match_labels(labels, num_faces):
    matches = defaultdict(list)
    for i,x in enumerate(labels):
        matches[x].append(i)
    return matches

def show_faces(video_filepath, matches, faces_data):
    cap = cv2.VideoCapture(video_filepath)
    count = 0
    for label in matches.keys():
        if label == -1:
            continue
        print("Class : {}".format(label))
        for face in matches[label]:
            count += 1
            if count > 4:
                count = 0
                break
            row = faces_data.iloc[face]
            x, y, w, h = row['x'], row['y'], row['w'], row['h']
            frame = row['frame'] - 24
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, img = cap.read()
            if ret:
                crop = img[y:y+h, x:x+w]
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                plt.imshow(crop)
                plt.show()

def label_persons(embeddings, frame_data, output_filepath=None):
    labels = identify_faces(embeddings)
    matches = match_labels(labels, frame_data.shape[0])
    frame_data['person'] = labels
    if output_filepath is not None:
        frame_data.to_csv(output_filepath, index=False)
    return frame_data

def main():
    # embeddings_filepath = 'embeddings.npy'
    embeddings_filepath = 'reactionvideo_embeddings.npy'

    embeddings = load_embeddings(embeddings_filepath)
    embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[2])
    # faces_filepath = 'sotu.csv' 
    faces_filepath = 'reactionvideo.csv'
    faces_info = load_face_info(faces_filepath)
    faces_info = faces_info.drop(['Unnamed: 0'], axis=1)
     
    labels = identify_faces(embeddings)
    print(Counter(labels))    

    matches = match_labels(labels, faces_info.shape[0])

    faces_info['person'] = labels

    faces_info.to_csv('reactionvideo_identified.csv', index=False)
    print(faces_info.head())

    # video_dir = '/Users/Tomas/Downloads/'
    # video_file = '/Users/Tomas/Downloads/sotu.mp4'
    # video_file = 'reactionvideo.mp4'
    # show_faces(os.path.join(video_dir, video_file), matches, faces_info)
    
if __name__ == '__main__':
    main()
