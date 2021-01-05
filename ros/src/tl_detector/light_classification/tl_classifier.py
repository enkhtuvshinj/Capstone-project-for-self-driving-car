import numpy as np
import os
import sys
import tensorflow as tf
import time
import rospy
import cv2
from collections import defaultdict
from io import StringIO
from PIL import Image
from styx_msgs.msg import TrafficLight


class TLClassifier(object):
    def __init__(self):
        # Load frozen inference graph model
        self.SSD_GRAPH_FILE= './light_classification/frozen_inference_graph.pb'
        self.detection_graph = self.load_graph()

        # Color translation between classification code ID and ros traffic light state
        self.color_translation_table = {1: TrafficLight.GREEN,
                                        2: TrafficLight.RED,
                                        3: TrafficLight.YELLOW,
                                        4: TrafficLight.UNKNOWN}
        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')

        # The classification of the object (integer id).
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
  
        self.max_img_width = 300
        self.max_img_height = 300
        
        
    def get_classification(self, cv_image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        # Pre-processing image
        image_np = self.process_image(cv_image)
        image_np = np.expand_dims(image_np, axis=0)

        # Configuration for ProtocolMessage
        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        
        with tf.Session(graph=self.detection_graph, config=config) as sess:                
            # Actual detection.
            (boxes, scores, classes) = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes], 
                                                feed_dict={self.image_tensor: image_np})
            # Remove unnecessary dimensions
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)

            # Filter boxes with a confidence score less than `confidence_cutoff`
            confidence_cutoff = 0.5
            boxes, scores, classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes)
            
            # Gets predicted traffic light
            traffic_light = TrafficLight()
            if classes.size > 0:
                traffic_light.state = self.color_translation_table[classes[0]]
            else:
                traffic_light.state = TrafficLight.UNKNOWN
            
            # Debugging
            rospy.loginfo("-------------------------------")
            rospy.loginfo(classes)
            rospy.loginfo(scores)
            
        return traffic_light.state
    

    def load_graph(self):
        """ Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.SSD_GRAPH_FILE, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph


    def process_image(self, image):
        """ Preprocesses image before a classification"""
        image = cv2.resize(image, (self.max_img_width, self.max_img_height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
        
        
    def filter_boxes(self, min_score, boxes, scores, classes):
        """ Returns boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)
        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes