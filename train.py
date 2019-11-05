
import h5py
import numpy as np
import pandas as pd
import pickle
import os

import cv2
import time
from collections import defaultdict

from sklearn.utils import shuffle

import numpy.random as rng

from create_model import get_model


TRAIN_FOLDER = "data/client/to_train/"
EVAL_FOLDER = 'data/client/to_eval/'
PREDICT_FOLDER = 'data/client/to_predict/'
SAVE_PATH = 'data/client/model'
TRAIN_DATA_PICKLE = 'train_images.pkl'
EVAL_DATA_PICKLE = 'eval_images.pkl'
TEST_DATA_PICKLE = 'test_images.pkl'

WIDTH = 105
HEIGHT = 105

BATCH_SIZE = 6
N_ITER = 50
EVALUATE_EVERY = 10
EVAL_BATCH_SIZE = 4

THRESHOLD = 0.5

def save_to_pickle(data, file_path):
	with open(file_path, 'wb') as f:
		pickle.dump(data, f)

def load_pickle(file_path):
	with open(file_path, 'rb') as f:
		return pickle.load(f)


class Preprocess(object):

	def __init__(self, width=WIDTH, height=HEIGHT, images_path=TRAIN_FOLDER,
				 save_path=SAVE_PATH, is_train_data=False):
		self.width = width
		self.height = height
		self.save_path = save_path
		self.path = images_path
		self.images_data = {}
		self.is_train_data = is_train_data

	def resize_image(self, image):
		# cv2.imshow("Image", image)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		return cv2.resize(image, (self.width, self.height))

	def process_images(self):
		for person in os.listdir(self.path):
			print("Processing: {}".format(person))
			self.images_data[person] = {}
			self.images_data[person]['originals'] = np.stack(
				[self.resize_image(cv2.imread(
					os.path.join(self.path, person, "orig", file), 0))
				 for file in os.listdir(
				 	os.path.join(self.path, person, "orig"))])
			self.images_data[person]['forgeries'] = np.stack(
				[self.resize_image(cv2.imread(
					os.path.join(self.path, person, "forg", file), 0))
				 for file in os.listdir(
					os.path.join(self.path, person, "forg"))])
		return self.images_data

	def save_to_pickle(self, pickle_file=None):
		if not pickle_file:
			if self.is_train_data:
				pickle_file = TRAIN_DATA_PICKLE
			else:
				pickle_file = EVAL_DATA_PICKLE
		path = os.path.join(self.save_path, pickle_file)
		save_to_pickle(self.images_data, path)
		return path



class GenerateBatch(object):

	def __init__(self, batch_size, data_pickle_path, width=WIDTH, height=HEIGHT):
		if not data_pickle_path:
			self.images_data = load_pickle(os.path.join(SAVE_PATH, TRAIN_DATA_PICKLE))
		else:
			self.images_data = load_pickle(data_pickle_path)
		self.batch_size = batch_size

	def get_batch(self):

	    # initialize 2 empty arrays for the input image batch
	    pairs = [np.zeros((self.batch_size, width, height, 1)) for i in range(2)]
	    
	    # initialize vector for the targets
	    targets = np.zeros((self.batch_size,))
	    
	    # make one half of it '1's, so 2nd half of batch has same class
	    targets[self.batch_size//2:] = 1

	    for i in range(self.batch_size):
	    	person = rng.choice(list(self.images_data.keys()))

    		originals = self.images_data[person]['originals']
	    	n_originals, w, h = originals.shape

	    	forgeries = self.images_data[person]['forgeries']
	    	n_forgeries, w, h = forgeries.shape

	    	idx_1 = rng.randint(0, n_originals)
	    	pairs[0][i, :, :, :] = originals[idx_1].reshape(width, height, 1)

	    	if i >= self.batch_size // 2:
	    		idx_2 = rng.randint(0, n_originals)
	    		pairs[1][i,:,:,:] = originals[idx_2].reshape(width, height, 1)
	    	else:
	    		idx_2 = rng.randint(0, n_forgeries)
	    		pairs[1][i,:,:,:] = forgeries[idx_2].reshape(width, height, 1)
	    
	    return pairs, targets

	# TODO
	# def generate(self):
	#     while True:
	#         pairs, targets = get_batch(self.batch_size)
	#         yield (pairs, targets)

class ClassifySignature(object):

	def __init__(self, predict_path=PREDICT_FOLDER,
				 train_pickle_path="data/client/model/train_images.pkl",
				 width=WIDTH, height=HEIGHT):
		self.width = WIDTH
		self.height = HEIGHT
		self.predict_path = predict_path
		self.train_images = load_pickle(train_pickle_path)
		
	def predict_against_originals(self):

		input_shape = (self.width, self.height, 1)

		model = get_model(input_shape)
		model.load_weights(os.path.join(SAVE_PATH, 'weights.h5'))

		for person in os.listdir(self.predict_path):
			originals = self.train_images[person]['originals']
			for image in os.listdir(os.path.join(self.predict_path, person)):
				img = cv2.imread(os.path.join(self.predict_path, person, image), 0)
				image_to_predict = cv2.resize(img, (self.width, self.height))
				pairs = [np.zeros((len(originals), self.width, self.height, 1))
						 for i in range(2)]	
				for idx_1 in range(len(originals)):																																																																																																																																													
					pairs[0][idx_1, :, :, :] = \
						originals[idx_1].reshape(self.width, self.height, 1)
					pairs[1][idx_1, :, :, :] = image_to_predict.reshape(self.width, self.height, 1)

				probs = np.array(model.predict(pairs)).flatten()
				print("Probabilites: {}".format(probs))
				print("Average probability: {}".format(probs.mean()))
				print("Min probability: {}".format(probs.min()))
				print("Max probability: {}".format(probs.max()))
				h, w = img.shape
				cv2.putText(img, str(probs.mean().round(2)), (w-60, h-20),
						    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,255), 1, 
						    cv2.LINE_AA)
				cv2.imshow(image, img)
				cv2.waitKey(0)
				cv2.destroyAllWindows()


if __name__ == "__main__":
	c = ClassifySignature()
	c.predict_against_originals()

'''

def train_model(train_data_path, batch_size=BATCH_SIZE, n_iter=N_ITER,
				evaluate_every=EVALUATE_EVERY, eval_batch_size=EVAL_BATCH_SIZE,
				width=WIDTH, height=HEIGHT, model_path=SAVE_PATH,
				eval_data_path=None):

	input_shape = (width, height, 1)

	model = get_model(input_shape)

	gd = GenerateBatch(batch_size, train_data_path, width, height)
	if eval_data_path:
		gd_eval = GenerateBatch(eval_batch_size, eval_data_path, width, height)

	t_start = time.time()
	print("Starting training process!")
	print("-------------------------------------")
	for i in range(1, n_iter+1):
		(inputs, targets) = gd.get_batch()
		loss = model.train_on_batch(inputs, targets)
		if i % evaluate_every == 0:
			print("\n ------------- \n")
			print("Time for {0} iterations: {1} mins".format(
				  i, (time.time()-t_start)/60.0))
			print("Train Loss: {0}".format(loss))
			if eval_data_path:
				(eval_inputs, eval_targets) = gd_eval.get_batch()
				probs = model.predict(eval_inputs)
				got_right = 0
				for i in range(len(probs)):
					if not(bool(probs[i][0] < THRESHOLD) ^ (not bool(eval_targets[i]))):
						got_right += 1
				print("Eval accuracy: {}".format(got_right/len(probs)))
			model.save_weights(os.path.join(model_path, 'weights.h5'))


if __name__ == "__main__":

	width = WIDTH
	height = HEIGHT
	
	p = Preprocess(width=width, height=height, images_path=TRAIN_FOLDER,
				   is_train_data=True)
	p.process_images()
	train_pickle_path = p.save_to_pickle()

	p = Preprocess(width=width, height=height, images_path=EVAL_FOLDER,
				   is_train_data=False)
	p.process_images()
	eval_pickle_path = p.save_to_pickle()

	train_model(train_data_path=train_pickle_path, width=width, height=height,
				eval_data_path=eval_pickle_path)

'''