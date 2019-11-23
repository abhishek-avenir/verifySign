
import cv2
import numpy as np
import os

from skimage.filters import threshold_local
from skimage import measure

from config import *


def crop_image_to_signature(
		image, output_folder=None, threshold=MIN_BLOB_PIXELS,
		image_shape=(WIDTH, HEIGHT), padding=PADDING):

	image = cv2.resize(image, image_shape)
	 
	# extract the Value component from the HSV color space and
	# apply adaptive thresholding to reveal the characters on the image
	H, S, V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
	V = cv2.normalize(V, None, 0, 255, cv2.NORM_MINMAX)
	T = threshold_local(V, THRESH_LOCAL_SIZE, offset=THRESH_LOCAL_OFFSET,
						method="gaussian")
	thresh = (V < T).astype("uint8") * 255
	# cv2.imshow("Thresholded image", thresh)

	# perform connected components analysis on the thresholded images and
	# initialize the mask to hold only the "large" components
	# we are interested in
	labels = measure.label(
		thresh, neighbors=NEIGHBOURS_FOR_CONNECTED_COMPS, background=0)
	mask = np.zeros(thresh.shape, dtype="uint8")

	# loop over the unique components
	for (i, label) in enumerate(np.unique(labels)):
		# if this is the background label, ignore it
		if label == 0:
			continue
	 
		# otherwise, construct the label mask to display only connected components for
		# the current label
		labelMask = np.zeros(thresh.shape, dtype="uint8")
		labelMask[labels == label] = 255
		numPixels = cv2.countNonZero(labelMask)
	 
		# if the number of pixels in the component is sufficiently large, add it to our
		# mask of "large" blobs
		if numPixels > threshold:
			mask = cv2.add(mask, labelMask)
		# cv2.putText(labelMask, str(numPixels), (w-30, h-20),
		# 			    cv2.FONT_HERSHEY_SIMPLEX, 0.35, 255, 1, 
		# 			    cv2.LINE_AA)
		# cv2.imshow("Label", labelMask)
		# cv2.waitKey(0)
	 
	# show the large components in the image
	# cv2.imshow("Large blobs", mask)

	[rows, cols] = np.where(mask != 0)
	try:
		min_row = max(0, min(rows)-padding)
		max_row = max(rows) + padding
		min_col = max(0, min(cols)-padding)
		max_col = max(cols) + padding
	except:
		print(f"Failed to crop image to signature.")
		return

	cropped_image = image[min_row:max_row+1, min_col:max_col+1]
	border = np.array([[[0, 0, 0]]*5]*cropped_image.shape[0], dtype='uint8')
	to_show = np.hstack(
		(cv2.resize(image, (cropped_image.shape[1], cropped_image.shape[0])),
		 border,
		 cropped_image))
	# cv2.imshow('Bounded Signature', to_show)

	# k = cv2.waitKey(0)
	# cv2.destroyAllWindows()
	# if k == ord('q'):
	# 	return False
	# if k != 27:
	resized = cv2.resize(cropped_image, image_shape)
	gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	norm_image = np.zeros_like(gray)
	norm_image = cv2.normalize(gray,  norm_image, 0, 255, cv2.NORM_MINMAX)
	# if output_folder:
	# 	output_image = os.path.join(output_folder, os.path.basename(image_path))
	# 	print(f"Saving {output_image}.. ")
	# 	cv2.imwrite(output_image, norm_image)
	# else:
	return norm_image
