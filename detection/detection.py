import numpy as np
import cv2
from tracking.deepSORT.deep_sort.detection import Detection
import logging

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

model = 'detection/mobilenet.caffemodel'
prototxt = 'detection/mobilenet.txt'

def load_model(model='detection/mobilenet.caffemodel', prototxt='detection/mobilenet.txt'):
    """
        Given a model path and a prototxt, load a model.

        Returns the loaded model.
    """

    return cv2.dnn.readNetFromCaffe(prototxt, model)

def raw_detect_from_frame(frame, net=None):
    """
        Given a frame, return a list of detections and the original frame, in a tuple.

        Parameters
        ----------
        frame :: `height x width x channels` numpy array :
            the frame to detect on


        Return
        ------
        tuple of `(np array of detections, frame)`

        detections is np array of dimension:
            detection_num, class, 
    """

    if net is None:
        net = load_model()

    # what does this do?
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    #pass blob through network and obtain the detections
    net.setInput(blob)
    detections = net.forward()

    return detections, frame

def _create_detections_from_tuple(det_tuple):
    """
        Creates a tracking.deepSORT.deep_sort.tracking.Detection from a tuple of raw
        detection information

        Parameters
        ----------

        det_tuple : tuple
            a tuple of `(frame_id, detect_id, bb_left, bb_top, bb_width, bb_height, confidence, x, y ,z, label)`
            

        Returns
        -------
        Detection
            a deepSORT detection object with the information encoded
    """
    bbox, confidence, label, feature = det_tuple[2:6], det_tuple[6], det_tuple[10], det_tuple[11:]
    
    return Detection(bbox, confidence, feature, label)

def get_detections_from_frame(frame, min_conf=0.6, net=None, label_filter=['person'], frame_num=0):
    """
        Given a frame, return the formatted `Detections` from this frame.
    """
    detections,_ = raw_detect_from_frame(frame, net=net)

    # current frame detections
    cur_dets = []

    (h, w) = frame.shape[:2]


    # iterate through our detections and prune them
    for i in np.arange(0, detections.shape[2]):

        # extract the confidence (i.e., probability) associated with
        # the prediction
        detectionConfidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if detectionConfidence > min_conf:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (x,y,z) = -1,-1,-1
            id = -1

            det_tuple = (frame_num, id, startX, startY, endX, endY, detectionConfidence, x, y, z, CLASSES[idx])

            if det_tuple[-1] in label_filter:
                logging.info('{} located with bounding box {} - conf: {}'.format(det_tuple[-1], det_tuple[2:6], det_tuple[6]))
            else:
                logging.debug('{} located (bad label) with bounding box {} - conf: {}'.format(det_tuple[-1], det_tuple[2:6], det_tuple[6]))

            cur_dets.append(det_tuple)

            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx], detectionConfidence * 100)

            # DISPLAY ORIGINAL DETECTION BOUNDING BOX AND LABEL

            # cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            # y = startY - 15 if startY - 15 > 15 else startY + 15
            # cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
        else:
            logging.debug('{} located (low conf) with bounding box {} - conf: {}'.format(CLASSES[int(detections[0, 0, i, 1])], (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype('int'), detectionConfidence))

    return [_create_detections_from_tuple(x) for x in cur_dets if _create_detections_from_tuple(x).label in label_filter],frame

def get_detections_from_stream(video_stream, min_conf=0.6, net=None, stream_has_ret=False, label_filter=['person']):
    """
        Given a cv2 video stream, yield the detections from it until the stream closes out
        or the end of time, whichever comes first.

        Parameters
        ----------
        video_stream :: cv2 Video Stream :
            the stream from which to pull frames
        min_conf :: float :
            number between 0 and 1 expressing the minimum confidence threshold needed to accept a detection
        net :: cv2.dnn Network :
            the network to use with detections, if None, will use the default network
        stream_has_ret :: bool :
            whether or not video_stream yields a tuple `ret,frame` or just frame
        label_filter :: list of str :
            a list of labels that we want to keep, will not yield any detections of other types
    """

    ret,Frame = None,None
    count=0
    while True:
        if stream_has_ret:
            ret,frame = video_stream.read()

            if not ret:
                video_stream.release()
                yield None,None # yield None,None when done,done
        else:
            frame = video_stream.read()
            
        yield get_detections_from_frame(frame, min_conf=min_conf, net=net, label_filter=label_filter, frame_num=count)
        count += 1

def draw_detections(detections, frame):
    for det in detections:
        startX,startY,endX,endY = det.to_tlbr().astype('int')
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0,0,255), 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(frame, str(det.label), (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)




