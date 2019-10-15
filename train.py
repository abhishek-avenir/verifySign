

import sys
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

import cv2
import time
from collections import defaultdict

from sklearn.utils import shuffle

import numpy.random as rng

from create_model import get_model


TRAIN_FOLDER = "data/custom2/to_train/"
EVAL_FOLDER = 'data/custom2/to_eval/'
SAVE_PATH = 'data/custom2/'
TRAIN_DATA_PICKLE = 'train_images.pkl'
EVAL_DATA_PICKLE = 'eval_images.pkl'


WIDTH = 105
HEIGHT = 105


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
				[self.resize_image(cv2.imread(os.path.join(self.path, person, "orig", file), 0))
				 for file in os.listdir(os.path.join(self.path, person, "orig"))])
			self.images_data[person]['forgeries'] = np.stack(
				[self.resize_image(cv2.imread(os.path.join(self.path, person, "forg", file), 0))
				 for file in os.listdir(os.path.join(self.path, person, "forg"))])
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

	    for person, image_data in self.images_data.items():

    		originals = image_data['originals']
	    	n_originals, w, h = originals.shape

	    	forgeries = image_data['forgeries']
	    	n_forgeries, w, h = forgeries.shape

	    	for i in range(self.batch_size):

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


def validate_evalset(eval_batch_size=4):
    """Create pairs of test image, support set for testing N way one-shot learning. """
    n_classes, n_examples, w, h = X.shape

    indices = rng.randint(0, n_examples,size=(N,))
    if language is not None: # if language is specified, select characters for that language
        low, high = categories[language]
        if N > high - low:
            raise ValueError("This language ({}) has less than {} letters".format(language, N))
        categories = rng.choice(range(low,high),size=(N,),replace=False)

    else: # if no language specified just pick a bunch of random letters
        categories = rng.choice(range(n_classes),size=(N,),replace=True)
#     true_category = categories[0]
    true_category = 0
    ex1, ex2 = rng.choice(n_examples,replace=False,size=(2,))
    test_image = np.asarray([X[true_category,ex1,:,:]]*N).reshape(N, w, h,1)
    support_set = X[categories,indices,:,:]
    support_set[0,:,:] = X[true_category,ex2]
    support_set = support_set.reshape(N, w, h,1)
    targets = np.zeros((N,))
    targets[0] = 1
    targets, test_image, support_set = shuffle(targets, test_image, support_set)
    pairs = [test_image,support_set]

    return pairs, targets


def train_model(train_data_path, batch_size=6, n_iter=50, evaluate_every=10,
				width=WIDTH, height=HEIGHT, model_path=SAVE_PATH,
				eval_batch_size=4, eval_data_path=None):

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
			print("Time for {0} iterations: {1} mins".format(i, (time.time()-t_start)/60.0))
			print("Train Loss: {0}".format(loss))
			if eval_data_path:
				(eval_inputs, eval_targets) = gd_eval.get_batch()
				probs = model.predict(eval_inputs)
				got_right = 0
				for i in range(len(probs)):
					if not(bool(probs[i][0] < 0.5) ^ (not bool(eval_targets[i]))):
						got_right += 1
				print("Eval accuracy: {}".format(got_right/len(probs)))
			model.save_weights(os.path.join(model_path, 'weights.{}.h5'.format(i)))


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

	# batch_size = 6
	# n_iter = 50
	# evaluate_every = 10
	train_model(train_data_path=train_pickle_path, width=width, height=height,
				eval_data_path=eval_pickle_path)

