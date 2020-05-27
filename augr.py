import logging
import cv2
from tracking.deepSORT.deep_sort import nn_matching
from tracking.deepSORT.deep_sort.tracker import Tracker
from tracking.deepSORT.application_util import preprocessing
from tracking.deepSORT.application_util.visualization import create_unique_color_float, create_unique_color_uchar
from tracking.tracker import run_tracking

from imutils.video import VideoStream
from imutils.video import FPS

from faces.facial_recognition import FaceRecognizer

from depth_estimation import depth_estim

from detection import detection

from mapping.atak_mapper import publish_detection

import numpy as np

import matplotlib.pyplot as plt

logging.basicConfig(filename='log.log', format='[%(asctime)s] - %(message)s', level=logging.INFO)

FACENET_ERROR_BEGIN = 'There were no tensor arguments to this function'

class AUGR:
    """
        An Autonomous Ubiquitous Gathering Relay (AUGR) instance.


        Two main public functions:

        `process_frame(frame)` : takes in an image and outputs the image with annotations overlayed

        `process_video(video)` : takes in a video and yields a stream of annotated frames

        Parameters
        ----------
        confidence_threshold :: float :
            a value between 0 and 1 representing the minimum required confidence for us to accept a prediction
        calc_distance :: bool :
            whether or not to calculate relative distances of objects in ingested frames
        calc_tracking :: bool :
            whether or not to track objects across frames. only relevant for streams of images, not for single-frame calculations.
        grab_faces :: bool :
            whether or not to grab the faces of people in this image
    """
    def __init__(self, confidence_threshold=0.6, calc_distance=True, calc_tracking=True, grab_faces=True, manual_location=True, manual_bearing=True):
        self.confidence_threshold = confidence_threshold
        self.calc_distance = calc_distance
        self.calc_tracking = calc_tracking
        self.grab_faces = grab_faces

        self.manual_location = manual_location
        self.manual_bearing = manual_bearing

        self.det_model = detection.load_model()

        self.people = list()

        if self.calc_tracking:
            self._init_tracking()

        if self.calc_distance:
            self.encoder,self.decoder = depth_estim.load_model()

        if self.grab_faces:
            self.facereg = FaceRecognizer()

        # set location and bearing

        if not self.manual_location:
            self.set_location()
        else:
            self.set_location((0,0)) # default to 0,0

        if not self.manual_bearing:
            self.set_bearing()
        else:
            self.set_bearing(0) # default to 0 degrees

    def _init_tracking(self):
        # Parameters
        self.nn_budget = 100
        self.max_cosine_distance = 0.2
        self.nms_max_overlap = 1.0

        # We use Cosine Distance for nearest neighbors
        self.track_metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        self.tracker = Tracker(self.track_metric)

    def set_location(self, latlong=None):
        """
            Set the latitude and longitude of this AUGR.

            Parameters
            ----------
            latlong :: tuple of float :
                the latitude and longitude coordinates to set.
                if None, read from a GPS sensor. if None and self.manual_location, do nothing
        """

        if latlong is None and self.manual_location:
            pass # this should do nothing

        if latlong is not None:
            lat,lon = latlong
            self.lat = lat
            self.lon = lon
        else:
            # TODO put GPS sensor code here
            raise NotImplementedError()

    def set_bearing(self, bearing=None):
        """
            Set the bearing of this AUGR.

            Parameters
            ----------
            bearing :: float :
                the bearing to set.
                if None, read from a magnet sensor. if None and self.manual_bearing, do nothing
        """

        if bearing is None and self.manual_bearing:
            pass # this should do nothing

        if bearing is not None:
            self.bearing = bearing
        else:
            # TODO put magnet sensor code here
            raise NotImplementedError()

    def load_video_stream(self, video_stream, stream_has_ret):
        """
            Parameters
            ----------
            video_stream :: cv2 video stream :
                the video stream to detect on
            stream_has_ret :: bool :
                whether or not video_stream yields a tuple `ret,frame` or just frame
        """

        self.video_stream = video_stream
        self.stream_has_ret = stream_has_ret

    def run(self, visualize=False, save_output=True):
        """
            Runs AUGR on self.video_stream

            Parameters
            ----------
            visualize :: bool :
                whether or not to display annotated frames to the user, else just text
            save_output :: bool :
                whether or not to save the output of this run. saves to log.txt (and out.mp4 if visualize=True)
        """

        save_as_video = visualize and save_output

        out_size = (800,600)

        if save_as_video: out = cv2.VideoWriter('out.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 15, out_size)

        try:
            self._main_loop(visualize, save_output, out=out if save_as_video else None, out_size=out_size)
        except KeyboardInterrupt as e:
            if save_as_video: out.release()
            cv2.destroyAllWindows()
            self.video_stream.stop()
        finally:
            if save_as_video: out.release()
            cv2.destroyAllWindows()
            self.video_stream.stop()

    def _main_loop(self, visualize, save_output, out=None, out_size=(800,600)):
        for detection_list,frame in detection.get_detections_from_stream(self.video_stream, self.stream_has_ret, net=self.det_model):

            # update out people
            for person in self.people:
                person.update(detection_list)

            # detection list will now only contain frames that were not a match with any existing people

            self.people = [person for person in self.people if person.relevance > 0]
            self.people.extend(detection_list)

            soph_distance = False

            if self.calc_distance:
                if soph_distance:
                    depth = depth_estim.get_depth_frame(self.encoder, self.decoder, frame)

                    for person in self.people:
                        person.set_distance(depth)
                else:
                    for person in self.people:
                        person.set_distance(None, sophisticated=False)

            if self.grab_faces:
                for person in self.people:
                    if person.should_retry():
                        try:
                            person.set_face(frame, self.facereg)

                            # try to find a name
                            person.set_name(frame, self.facereg)
                        except RuntimeError as e: # this is a big hack
                            if str(e)[:len(FACENET_ERROR_BEGIN)] == FACENET_ERROR_BEGIN:
                                logging.warning('No faces detected - expected a face!')
                            else:
                                raise e

            # if self.calc_tracking:
            #     run_tracking(self.tracker, detection_list, frame, nms_max_overlap=self.nms_max_overlap, visualize=visualize)
            # elif visualize:
            #     detection.draw_detections(detection_list, frame, depth=depth)

            for person in self.people:
                person.draw(frame)


            if visualize or save_output:
                frame = cv2.resize(frame, out_size)

            if visualize and save_output:
                out.write(frame)

            if visualize:
                cv2.imshow('frame', frame)
                cv2.waitKey(1)

if __name__ == "__main__":
    vs = VideoStream(src=0).start()

    a = AUGR(calc_distance=False, calc_tracking=False, grab_faces=True)
    a.load_video_stream(vs, False)
    a.run(True, True)

