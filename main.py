import cv2
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np

import colorsys

from tracking.deepSORT.application_util import preprocessing
from tracking.deepSORT.application_util import visualization
from tracking.deepSORT.deep_sort import nn_matching
from tracking.deepSORT.deep_sort.detection import Detection
from tracking.deepSORT.deep_sort.tracker import Tracker

def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)

class AUGR():
    """AUGR: Autonomous Ubiquitous Gathering Relay

    TODO
    """

    def __init__(self):
        super().__init__()

    def run(self):

        # Parameters
        nn_budget = 100
        max_cosine_distance = 0.2
        nms_max_overlap = 1.0

        # We use Cosine Distance for nearest neighbors
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        tracker = Tracker(metric)

        # stream through each frame and its associated detections
        for detections, frame in self.detection_stream():

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(
                boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            # Update tracker.
            tracker.predict()
            tracker.update(detections)

            # Visualize the Tracks
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 0:
                    continue
                startX,startY,endX,endY = track.to_tlwh().astype(np.int) # top, left, bottom, right
                color = create_unique_color_uchar(track.track_id)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, str(track.track_id), (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.imshow("Frame", frame)



    def detection_stream(self):
        """
            Starts an infinite detection stream that collects information
            from a frame, detects objects in it, and then yields the detections
            for other pieces of AUGR to operate on before displaying frames

            Parameters
            ----------
            self : AUGR object

            Yields
            -------
            Tuple[List[Detection], VideoStream.Frame]
                yields a detection a two-element tuple containing a List of Detections
                and the Frame they were detected in
        """
        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]
        COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

        model = 'tracking/mobilenet.caffemodel'
        prototxt = 'tracking/mobilenet.txt'
        confidence = 0.3

        net = cv2.dnn.readNetFromCaffe(prototxt, model)
        vs = VideoStream(src=0).start()

        count = 0

        while True:
            frame = vs.read()
            # frame = imutils.resize(frame, width = 300)

            # grab frame dimensions and convert frame to blob
            (h, w) = frame.shape[:2]
            # blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

            #pass blob through network and obtain the detections
            net.setInput(blob)
            detections = net.forward()

            # current frame detections
            cur_dets = []

            for i in np.arange(0, detections.shape[2]):

                # extract the confidence (i.e., probability) associated with
                # the prediction
                detectionConfidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if detectionConfidence > confidence:
                    # extract the index of the class label from the
                    # `detections`, then compute the (x, y)-coordinates of
                    # the bounding box for the object
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    (x,y,z) = -1,-1,-1
                    id = -1

                    det_tuple = (count, id, startX, startY, endX, endY, detectionConfidence, x, y, z)

                    cur_dets.append(det_tuple)

                    # draw the prediction on the frame
                    label = "{}: {:.2f}%".format(CLASSES[idx], detectionConfidence * 100)
                    print(label)

                    # DISPLAY ORIGINAL DETECTION BOUNDING BOX AND LABEL

                    # cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                    # y = startY - 15 if startY - 15 > 15 else startY + 15
                    # cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

            yield [self._create_detections_from_tuple(x) for x in cur_dets],frame

            key = cv2.waitKey(1) & 0xFF
            count += 1

        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.stop()

    def _create_detections_from_tuple(self, det_tuple):
        """
            Creates a tracking.deepSORT.deep_sort.tracking.Detection from a tuple of raw
            detection information

            Parameters
            ----------
            self : AUGR object

            det_tuple : tuple
                a tuple of `(frame_id, detect_id, bb_left, bb_top, bb_width, bb_height, confidence, x, y ,z)`

            Returns
            -------
            Detection
                a deepSORT detection object with the information encoded
        """
        bbox, confidence, feature = det_tuple[2:6], det_tuple[6], det_tuple[10:]
        
        return Detection(bbox, confidence, feature)




if __name__ == "__main__":
    augr = AUGR()
    augr.run()