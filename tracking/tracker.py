from tracking.deepSORT.application_util import preprocessing
from tracking.deepSORT.application_util.visualization import create_unique_color_float, create_unique_color_uchar

import numpy as np
import cv2

def run_tracking(tracker, detections, frame, nms_max_overlap=1.0, visualize=True):

    # TODO some better logging here?

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
    if visualize:
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            startX,startY,endX,endY = track.to_tlwh().astype(np.int) # top, left, bottom, right
            color = create_unique_color_uchar(track.track_id)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, str(track.track_id), (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)