from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
from os import listdir
from os.path import isfile, join
import scipy.spatial.distance as distance

import matplotlib.pyplot as plt

class Face():
    def __init__(self, img, vec):
        """
            img: some (CHANNEL_SIZExHEIGHTxWIDTH) array representing the image of this face)
            vec: the vector representation of this face
        """

        self.img = img
        self.vec = vec

    def similar(self, other_face, threshold = 0.2):
        """
        Returns true if the other face is within the similarity threshold, false otherwise.
        """
        sim = (self.vec - other_face.vec).norm().item()

        return sim <= threshold

    def display(self):
        cv2.imshow('face', self.img)
        cv2.waitKey(0)

class FaceRecognizer():
    """https://github.com/timesler/facenet-pytorch/blob/master/examples/face_tracking.ipynb

    cite yo sources. also a useful repo in general  
    """
    def __init__(self, crop_size=160, min_face_size=20, memory_length=20, debug=False):
        """Initialize our facial recognition object"""

        # i dont have a GPU but maybe you do :)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if debug: print('Running on device: {}'.format(self.device))

        self.det_mtcnn = MTCNN(keep_all=True, device=self.device)

        self.min_face_size = min_face_size
        self.crop_size = crop_size
        self.emb_mtcnn = MTCNN(
            image_size=self.crop_size, margin=0, min_face_size=self.min_face_size,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=self.device
        )

        self.resnet = InceptionResnetV1(pretrained='casia-webface').eval().to(self.device)

        self.faces = []

    def detect_faces(self, frame):
        """
            Given a frame (some `HEIGHTxWIDTHxCHANNELS` np-array), return a list of tuples corresponding to the `(tlx,tly,brx,bry)` coordinates of each bounding box for each detected face.
        """

        bb_list = []

        # Detect faces
        boxes, _ = self.det_mtcnn.detect(frame)
        faces_tmp = []

        if boxes is not None:
            for box in boxes:
                l = box.tolist()
                tlx,tly,brx,bry = l
                # fix dims
                tlx = max(0, int(tlx))
                tly = max(0, int(tly))
                brx = max(0, int(brx))
                bry = max(0, int(bry))

                bb_list.append((tlx,tly,brx,bry))

                continue # TODO make this take less memory!
                
                img = frame[tly:bry,tlx:brx,:] # crop frame

                if img.shape[0] < self.min_face_size or img.shape[1] < self.min_face_size: continue # skip if its not big enough

                # res = cv2.resize(img, (self.crop_size,self.crop_size))
                x_aligned = self.emb_mtcnn(img)
                if x_aligned is not None:
                    # add a dimension
                    x_aligned = x_aligned.unsqueeze(0)
                    emb = self.resnet(x_aligned)
                    new_face = Face(img, emb)
                    faces_tmp.append(new_face)
        else:
            boxes = []

        for face in faces_tmp:
            unique = True
            for face2 in self.faces:
                if face.similar(face2):
                    unique = False
                    break

            if unique:
                self.faces.append(face)

        return boxes

    def get_faces(self):
        return self.faces

    def get_face_vector(self, face_img):
        if face_img is None: return None
        x_aligned = self.emb_mtcnn(face_img)
        # print(x_aligned)
        # plt.imshow(x_aligned.permute(1,2,0).long())
        # plt.show()
        if x_aligned is not None:
            # add a dimension
            x_aligned = x_aligned.unsqueeze(0)
            emb = self.resnet(x_aligned)

            return emb
        return None


    # TODO the bounding boxes seem real off here...
    def detect_one_face(self, frame):
        """
            Given a frame (some `HEIGHTxWIDTHxCHANNELS` np-array), return a tuple corresponding to the `(tlx,tly,brx,bry)` coordinates of the most probable face.

            if it can't find one, it returns None
        """

        bb_list = []

        # Detect faces
        boxes, probs = self.det_mtcnn.detect(frame)
        faces_tmp = []

        max_prob = max(probs)

        if boxes is not None:
            for i,box in enumerate(boxes):
                if probs[i] == max_prob:
                    l = box.tolist()
                    tlx,tly,brx,bry = l
                    # fix dims
                    tlx = max(0, int(tlx))
                    tly = max(0, int(tly))
                    brx = max(0, int(brx))
                    bry = max(0, int(bry))

                    return (tlx,tly,brx,bry)

        return None

    def get_faces(self):
        return self.faces

def associate_name(face_rec, face_img, face_db='faces/face_db', threshold=0.45):
    
    # TODO
    
    face_refs = [f for f in listdir(face_db) if isfile(join(face_db, f))]

    anchor_vec = face_rec.get_face_vector(face_img)
    if anchor_vec is not None:
        anchor_vec = anchor_vec.detach().numpy()
        for face_ref in face_refs:
            vec = np.load('{}/{}'.format(face_db, face_ref), allow_pickle=True)

            sim = distance.cosine(anchor_vec, vec)

            if sim < threshold: return face_ref.split('/')[-1]
    return None


if __name__ == "__main__":
    f = FaceRecognizer()
    # frame = cv2.imread('faces/test.jpeg')
    # boxes = f.detect_one_face(frame)
    # cv2.rectangle(frame, (boxes[0],boxes[1]), (boxes[2],boxes[3]), (255,0,0), 1)
    # cv2.imshow('a', frame)
    # cv2.waitKey(0)

    f1 = cv2.imread('faces/face_db/patrick.jpeg')
    # f2 = cv2.imread('faces/face_db/michael.jpeg')
    # f2 = cv2.imread('faces/found_faces/found.jpg')

    v1 = f.get_face_vector(f1).detach().numpy()
    # v2 = f.get_face_vector(f2).detach().numpy()

    # print(distance.cosine(v1, v2))
    np.save('faces/face_db/patrick.npy', v1)

