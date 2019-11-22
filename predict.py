
import cv2
import os
import numpy as np

from argparse import ArgumentParser
from config import *
from connected_comp import crop_image_to_signature
from create_model import get_model
from utils import load_pickle


PREDICT_FOLDER = 'data/client/to_predict/'

TRAIN_PICKLE = os.path.join(SAVE_PATH, TRAIN_DATA_PICKLE)

class ClassifySignature(object):

	def __init__(self, predict_path=PREDICT_FOLDER,
				 train_pickle_path=TRAIN_PICKLE,
				 width=WIDTH, height=HEIGHT):
		self.width = width
		self.height = height
		self.predict_path = predict_path
		self.train_images = load_pickle(train_pickle_path)
		
	def predict_against_originals(self, image, persons, identity=False):

		img = crop_image_to_signature(image)
		input_shape = (self.width, self.height, 1)

		model = get_model(input_shape)
		model.load_weights(os.path.join(SAVE_PATH, 'weights.h5'))

		if identity:
			persons = self.train_images.keys()
		print(f"Verifying against: {list(persons)}")
		probabilites_by_person = {}
		for person in persons:
			originals = self.train_images[person]['originals']
			image_to_predict = cv2.resize(img, (self.width, self.height))
			pairs = [np.zeros((len(originals), self.width, self.height, 1))
					 for i in range(2)]	
			for idx_1 in range(len(originals)):																																																																																																																																													
				pairs[0][idx_1, :, :, :] = \
					originals[idx_1].reshape(self.width, self.height, 1)
				pairs[1][idx_1, :, :, :] = \
					image_to_predict.reshape(self.width, self.height, 1)

			probs = np.array(model.predict(pairs)).flatten()
			probabilites_by_person[person] = probs.mean()
			# print("Probabilites: {}".format(probs))
			# print("Average probability: {}".format(probs.mean()))
			# print("Min probability: {}".format(probs.min()))
			# print("Max probability: {}".format(probs.max()))
			h, w = image_to_predict.shape
			cv2.putText(
				image_to_predict, str(probs.mean().round(2)), (w-30, h-5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,255), 1, cv2.LINE_AA)
			cv2.imshow(f"Against {person}", image_to_predict)
		print("Image: {}".format(image))
		for person, probability in probabilites_by_person.items():
			print(f"Probability against `{person}`: {probability}")
		cv2.waitKey(0)
		cv2.destroyAllWindows()


if __name__ == "__main__":
	parser = ArgumentParser()
	group = parser.add_mutually_exclusive_group()
	group.add_argument('-p', '--persons', nargs='+')
	group.add_argument('-id', '--identify', action='store_true')
	parser.add_argument('-i', '--image', required=True)
	args = parser.parse_args()
	c = ClassifySignature()
	c.predict_against_originals(args.image, args.persons, args.identify)
