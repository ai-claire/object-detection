import numpy as np
import os
import sys
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import collections
from tensorflow import Graph


from keras.models import load_model
import cv2
import numpy as np
# from PIL import Image

import cv2

cap = cv2.VideoCapture(0)
# using a video instead!

if tf.__version__ < '1.4.0':
	raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util


# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90



# In[ ]:
print("Started to load")

detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')
print("Loaded")

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`. Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[ ]:


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
print("Got map")



img_width, img_height = 499, 569

if K.image_data_format() == 'channels_first':
	input_shape = (3, img_width, img_height)
else:
	input_shape = (img_width, img_height, 3)


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

def init_model():
	model.load_weights('second_try.h5', by_name=True)

	model.compile(loss='binary_crossentropy',
		optimizer='rmsprop',
		metrics=['accuracy'])


init_model()

def get_action(image, xmin, ymin, xmax, ymax):
	ymin *= image.shape[0]
	ymax *= image.shape[0]
	xmin *= image.shape[1]
	xmax *= image.shape[1]
	img1 = image[int(ymin):int(ymax), int(xmin):int(xmax)]
	print("Shape: ", img1.shape)
	img1 = cv2.resize(img1, (img_width, img_height))
	img1 = np.reshape(img1, [1, img_width, img_height, 3])
	print("Shape: ", img1.shape)

	

	classes1 = model.predict_classes(img1)
	if classes1[0][0] == 0:
		return 'Picking up'
	return 'Idle'

def visualize_boxes_and_labels_on_image_array(image,
	boxes,
	classes,
	scores,
	category_index,
	max_boxes_to_draw=20,
	min_score_thresh=.5,
	line_thickness=4):
# normalized coordinates: True, linethickness 8

	box_to_display_str_map = collections.defaultdict(list)
	box_to_color_map = collections.defaultdict(str)
	
	
	for i in range(min(max_boxes_to_draw, boxes.shape[0])):
		if scores[i] > min_score_thresh:
			box = tuple(boxes[i].tolist())

			if classes[i] in category_index.keys():
				class_name = category_index[classes[i]]['name']
			else:
				class_name = 'N/A'

			# add only people
			if class_name == 'person':
				ymin, xmin, ymax, xmax = box
				display_str = '{}: {}%: {}'.format(
						class_name,
						int(100*scores[i]),
						get_action(image, xmin, ymin, xmax, ymax)
						)


				box_to_display_str_map[box].append(display_str)
				
				box_to_color_map[box] = vis_util.STANDARD_COLORS[
						classes[i] % len(vis_util.STANDARD_COLORS)]

	# Draw all boxes onto image.
	for box, color in box_to_color_map.items():
		ymin, xmin, ymax, xmax = box
		print(ymin, ymax, xmin, xmax)

		vis_util.draw_bounding_box_on_image_array(
				image,
				ymin,
				xmin,
				ymax,
				xmax,
				color=color,
				thickness=line_thickness,
				display_str_list=box_to_display_str_map[box],
				use_normalized_coordinates=True)

	return image







def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape(
			(im_height, im_width, 3)).astype(np.uint8)


print("opening camera")

with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
		# Definite input and output Tensors for detection_graph
		image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
		# Each box represents a part of the image where a particular object was detected.
		detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
		# Each score represent how level of confidence for each of the objects.
		# Score is shown on the result image, together with the class label.
		detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
		detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
		num_detections = detection_graph.get_tensor_by_name('num_detections:0')
		
		# cap = cv2.VideoCapture('test.avi')
		
		# while True:
		while (cap.isOpened()):
			ret, image_np = cap.read()
			# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
			image_np_expanded = np.expand_dims(image_np, axis=0)
				# Actual detection.
			(boxes, scores, classes, num) = sess.run(
						[detection_boxes, detection_scores, detection_classes, num_detections],
						feed_dict={image_tensor: image_np_expanded})
				# Visualization of the results of a detection.

			visualize_boxes_and_labels_on_image_array(
						image_np,
						np.squeeze(boxes),
						np.squeeze(classes).astype(np.int32),
						np.squeeze(scores),
						category_index,
						line_thickness=8)
			cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))

			if cv2.waitKey(25) & 0xFF == ord('q'):
				cv2.destroyAllWindows()


