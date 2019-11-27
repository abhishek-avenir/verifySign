
import cv2
import os

from argparse import ArgumentParser
from copy import copy
from glob import glob

from config import *
from connected_comp_new import crop_image_to_signature


def crop_image_by_cc(input_folder, output_folder):
	for ext in copy(EXTENSIONS):
		EXTENSIONS.append(ext.upper())

	files = []
	for ext in EXTENSIONS:
		files.extend(glob("{}/*.{}".format(input_folder, ext)))

	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	for file in files:
		crop_image_to_signature(file, output_folder, threshold=MIN_BLOB_PIXELS,
								image_shape=(WIDTH, HEIGHT))

def process():
	for input_folder, output_folder in INPUT_TO_OUTPUT.items():
		crop_image_by_cc(input_folder, output_folder)

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument('-i', '--input-folder')
	parser.add_argument('-o', '--output-folder')
	parser.add_argument('-c', '--consider-config', action='store_true')
	args = parser.parse_args()
	if args.consider_config:
		process()
	else:
		crop_image_by_cc(args.input_folder, args.output_folder)
