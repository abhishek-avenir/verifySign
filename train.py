
import cv2
import numpy.random as rng
import numpy as np
import os
import pandas as pd
import time

from collections import defaultdict
from sklearn.utils import shuffle

from create_model import get_model
from config import *
from utils import save_to_pickle, load_pickle


class Preprocess(object):

	def __init__(self, width=WIDTH, height=HEIGHT, save_path=SAVE_PATH,
				 is_train_data=False):
		self.width = width
		self.height = height
		self.save_path = save_path
		self.images_data = {}
		self.is_train_data = is_train_data

	def process_images(self):
		if self.is_train_data:
			f = "train"
		else:
			f = "eval"
		for person, path_details in PATHS.items():
			print(f"Processing: {person}")
			self.images_data[person] = {}
			path_to_orig = path_details[f]['orig']
			path_to_forg = path_details[f]['forg']
			self.images_data[person]['originals'] = \
				np.stack([cv2.imread(os.path.join(path_to_orig, file), 0)
						 for file in os.listdir(path_to_orig)])
			self.images_data[person]['forgeries'] = \
				np.stack([cv2.imread(os.path.join(path_to_forg, file), 0)
						 for file in os.listdir(path_to_forg)])
		person = rng.choice(list(self.images_data.keys()))
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
		self.width = width
		self.height = height

	def get_batch(self):

	    # initialize 2 empty arrays for the input image batch
	    pairs = [np.zeros((self.batch_size, self.width, self.height, 1)) for i in range(2)]
	    
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
	    	pairs[0][i, :, :, :] = originals[idx_1].reshape(self.width, self.height, 1)

	    	if i >= self.batch_size // 2:
	    		idx_2 = rng.randint(0, n_originals)
	    		pairs[1][i,:,:,:] = originals[idx_2].reshape(self.width, self.height, 1)
	    	else:
	    		idx_2 = rng.randint(0, n_forgeries)
	    		pairs[1][i,:,:,:] = forgeries[idx_2].reshape(self.width, self.height, 1)
	    
	    return pairs, targets

	# TODO
	# def generate(self):
	#     while True:
	#         pairs, targets = get_batch(self.batch_size)
	#         yield (pairs, targets)


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
					if not(bool(probs[i][0] < CLASSIFIER_THRESHOLD) ^
						   (not bool(eval_targets[i]))):
						got_right += 1
				print("Eval accuracy: {}".format(got_right/len(probs)))
			model.save_weights(os.path.join(model_path, 'weights.h5'))


if __name__ == "__main__":

	p = Preprocess(width=WIDTH, height=HEIGHT, is_train_data=True)
	p.process_images()
	train_pickle_path = p.save_to_pickle()

	p = Preprocess(width=WIDTH, height=HEIGHT, is_train_data=False)
	p.process_images()
	eval_pickle_path = p.save_to_pickle()

	train_model(train_data_path=train_pickle_path, width=WIDTH, height=HEIGHT,
				eval_data_path=eval_pickle_path)
