import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
from PIL import Image

class Person:

    def __init__(self, detection, face=None, name=None, distance=None):
        """
            Parameters
            ----------
            detection :: Detection :
                the detection associated with this person
        """
        self.detection = detection

        self.face = face
        self.name = name
        self.distance = distance

        self.face_counter=50

        self.relevance = 1.0 # how relevant is this detection

    # DETECTION METHODS
    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        return self.detection.to_tlbr().astype('int')

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        return self.detection.to_xyah()

    def _get_subset_from_frame(self, frame, has_channels=True):
        """Returns the sub-image containing this person from a given frame."""

        tlx,tly,brx,bry = self.to_tlbr()

        # TODO problems here?

        return frame[tly:bry, tlx:brx, :] if has_channels else frame[tly:bry,tlx:brx]

    def _get_face_from_frame(self, frame, has_channels=True):
        """Returns the sub-image containing this face from a given frame."""
        if self.face is not None:
            tlx,tly,brx,bry = self.face
            return frame[tly:bry, tlx:brx, :] if has_channels else frame[tly:bry,tlx:brx]
        else:
            return None

    def set_face(self, frame, face_recognizer):
        """
            Searches this frame for a face and sets it, if applicable.
        """
        self.face_counter = 50

        tlx,tly,brx,bry = self.to_tlbr()
        f_tlx,f_tly,f_brx,f_bry = face_recognizer.detect_one_face(self._get_subset_from_frame(frame))


        self.face = tlx + f_tlx, tly + f_tly, tlx + f_brx, tly + f_bry

        cv2.rectangle(frame, (self.face[0], self.face[1]), (self.face[2], self.face[3]), (255,0,0), 1)

    def set_name(self, frame, face_recognizer, face_ref='faces/face_db'):
        self.face_counter = 50

        my_vec = face_recognizer.get_face_vector(Image.fromarray(self._get_subset_from_frame(frame)))

        if my_vec is not None:
            face_refs = [join(face_ref, f) for f in listdir(face_ref) if isfile(join(face_ref, f)) and f != '.DS_Store']

            for face_ref in face_refs:
                face = Image.open(face_ref)
                vec = face_recognizer.get_face_vector(face)

                sim = (my_vec - vec).norm().item()

                if sim < 0.2:
                    self.name = face_ref.split('/')[-1]
                    break

    def set_distance(self, depth_frame, sophisticated=True):
        if sophisticated:
            depth_subs = self._get_subset_from_frame(depth_frame, False)

            avg_depth = np.mean(depth_subs)

            self.distance = avg_depth
        else:
            tlx,tly,brx,bry = self.to_tlbr()

            width = ((brx - tlx) * 0.9) // 1

            # average male shoulder width is 16 inches = 0.4064m
            # TODO

            width_inches = width * 16
            self.distance = width_inches / 39.3701 # num inches in meter

    def is_identified(self):
        return self.name is not None

    def should_remove(self):
        return self.relevance <= 0.0

    def should_retry(self):
        return self.name is None and self.face_counter <= 0

    def absorb(self, other_person):
        if other_person.detection is not None:
            self.detection = other_person.detection
        
        if other_person.face is not None:
            self.face = other_person.face
        
        if other_person.name is not None:
            self.name = other_person.name

        if other_person.distance is not None:
            self.distance = other_person.distance

    def update(self, people, min_iou=0.5):
        matched = False
        max_iou = 0
        argmax_iou = None
        for i,person in enumerate(people):
            iou = self.iou(person)
            if iou < min_iou: continue

            if iou > max_iou:
                argmax_iou = i
                max_iou = iou
                matched = True

        if matched:
            self.absorb(people.pop(argmax_iou))
            self.relevance = 1.0
        else:
            self.relevance -= .25

        self.face_counter -= 1


    def iou(self, other_person):
        """
            Calculates the intersection over union of this bounding box
            and another bounding box

            other_person : BoundingBox :
                another bounding box
        """
        op = other_person

        return self._intersection(op) / self._union(op)

    def _intersection(self, other_person):
        """Calculates the intersection area of this bounding box and another bounding box."""

        tlx,tly,brx,bry = self.to_tlbr()
        o_tlx,o_tly,o_brx,o_bry = other_person.to_tlbr()

        rightest_left = max(tlx, o_tlx)
        leftest_right = min(brx, o_brx)

        if leftest_right < rightest_left:
            return 0 # no overlap
        
        bottomest_top = max(tly, o_tly)
        toppest_bottom = min(bry, o_bry)

        if bottomest_top > toppest_bottom:
            return 0

        return (leftest_right - rightest_left) * (toppest_bottom - bottomest_top)

    def _union(self, other_person):
        """Calculates the union area of this bounding box and another bounding box."""

        tlx,tly,brx,bry = self.to_tlbr()
        width,height = brx-tlx,bry-tly

        o_tlx,o_tly,o_brx,o_bry = other_person.to_tlbr()
        o_width,o_height = o_brx-o_tlx, o_bry - o_tly

        total_area = width * height + o_width * o_height

        return total_area - self._intersection(other_person)

    def draw(self, frame):
        tlx,tly,brx,bry = self.to_tlbr()
        cv2.rectangle(frame, (tlx,tly), (brx,bry), (0,0,255) if self.name is None else (0, 255, 0), int(4 * self.relevance))
        if self.face is not None:
            tlx,tly,brx,bry = self.face
            cv2.rectangle(frame, (tlx,tly), (brx,bry), (255,0,0), 1)

        y = tly - 15 if tly - 15 > 15 else tly + 15
        cv2.putText(frame, self.get_label_string(), (tlx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    def get_label_string(self):
        to_ret = 'person'
        if self.name is not None: to_ret = self.name
        if self.distance: to_ret += ' - {}m'.format(round(self.distance, 3))

        return to_ret

    def __str__(self):
        return 'Person | Name: {} | Bounding Box: {} | Face_BoundingBox: {} | Distance: {} | Relevance: {}'.format(self.name, self.to_tlbr(), self.face, self.distance, self.relevance)

if __name__ == "__main__":
    a = Person(None)
    b = Person('a')
    c = [a,b]

    for thing in c:
        if thing.detection == 'a':
            c.remove(thing)
