
import cv2
import os
import numpy as np

from create_model import get_model
from utils import load_pickle


PREDICT_FOLDER = 'data/client/to_predict/'
MODEL_PATH = "models/client/"
TRAIN_PICKLE = os.path.join(MODEL_PATH, "train_images.pkl")
WIDTH = 105
HEIGHT = 105

class ClassifySignature(object):

	def __init__(self, predict_path=PREDICT_FOLDER,
				 train_pickle_path=TRAIN_PICKLE,
				 width=WIDTH, height=HEIGHT):
		self.width = WIDTH
		self.height = HEIGHT
		self.predict_path = predict_path
		self.train_images = load_pickle(train_pickle_path)
		
	def predict_against_originals(self):

		input_shape = (self.width, self.height, 1)

		model = get_model(input_shape)
		model.load_weights(os.path.join(MODEL_PATH, 'weights.h5'))

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
				print("Image: {}".format(image))
				print("Probabilites: {}".format(probs))
				print("Average probability: {}".format(probs.mean()))
				print("Min probability: {}".format(probs.min()))
				print("Max probability: {}".format(probs.max()))
				h, w = image_to_predict.shape
				cv2.putText(image_to_predict, str(probs.mean().round(2)), (w-30, h-20),
						    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,255), 1, 
						    cv2.LINE_AA)
				cv2.imshow(image, image_to_predict)
				cv2.waitKey(0)
				cv2.destroyAllWindows()


if __name__ == "__main__":
	c = ClassifySignature()
	c.predict_against_originals()
