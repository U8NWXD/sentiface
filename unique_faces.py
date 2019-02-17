from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import cv2
import matplotlib.pyplot as plt
from collections import Counter

def identify_faces(embeddings):
    clustering = DBSCAN(min_samples=15).fit(embeddings)
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
    for label in matches.keys():
        if label == -1:
            continue
        print("Class : {}".format(label))
        for face in matches[label]:
            row = faces_data.iloc[face]
            x, y, w, h = row['x'], row['y'], row['w'], row['h']
            frame = row['frame'] - 1000
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, img = cap.read()
            if ret:
                crop = img[y:y+h, x:x+w]
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                plt.imshow(crop)
                plt.show()


def main():
    embeddings_filepath = 'embeddings.npy'
    embeddings = load_embeddings(embeddings_filepath)
    embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[2])
    faces_filepath = 'sotu.csv' 
    faces_info = load_face_info(faces_filepath)
    
    labels = identify_faces(embeddings)

    matches = match_labels(labels, faces_info.shape[0])

    video_file = '/Users/Tomas/Downloads/sotu.mp4'
    show_faces(video_file, matches, faces_info)
    
if __name__ == '__main__':
    main()
