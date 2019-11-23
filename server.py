
import cv2
import datetime
import json
import keras
import numpy as np
import os
import tensorflow as tf

from flask import Flask, request, jsonify, Response, render_template
from predict import ClassifySignature
from PIL import Image

app = Flask(__name__)


@app.route("/")
def hello():
  return render_template('index.html')


def load_model():
	global cs
	global graph
	global session
	graph = tf.get_default_graph()
	session = tf.Session()
	with session.as_default():
		cs = ClassifySignature()


@app.route('/classify', methods=['POST'])
def predict():
	data = {'success': False}
	if request.method == 'POST':
		if request.files.get("file"):
			image = np.array(Image.open(request.files['file']))
			with graph.as_default():
				with session.as_default():
					data['predictions'] = cs.predict_against_originals(
						image, persons=None, identify=True)
			data['success'] = True
	return jsonify(data)


if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
	load_model()
	app.run(threaded=False)

'''
curl -X POST -F file=@data/client/to_predict/person_1/forg_IMG_0804.jpeg \
	'http://localhost:5000/classify'
'''
