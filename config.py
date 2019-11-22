
# Preprocessing image
WIDTH = 150
HEIGHT = 75

EXTENSIONS = ['jpg', 'jpeg', 'png']

# Cropping around signature
MIN_BLOB_PIXELS = 50
PADDING = 2
THRESH_LOCAL_SIZE = 29
THRESH_LOCAL_OFFSET = 15
NEIGHBOURS_FOR_CONNECTED_COMPS = 8

INPUT_TO_OUTPUT = {
	"data/client/to_train/person_1/orig":
		"data/client/to_train/person_1/orig_cropped",
	"data/client/to_train/person_1/forg":
		"data/client/to_train/person_1/forg_cropped",
	"data/client/to_eval/person_1/orig":
		"data/client/to_eval/person_1/orig_cropped",
	"data/client/to_eval/person_1/forg":
		"data/client/to_eval/person_1/forg_cropped"}

PATHS = {
	"person_1": {
		'train': {
			'orig': "data/client/to_train/person_1/orig_cropped",
			'forg': "data/client/to_train/person_1/forg_cropped"},
		'eval': {
			'orig': "data/client/to_eval/person_1/orig_cropped",
			'forg': "data/client/to_eval/person_1/forg_cropped"}}}

SAVE_PATH = 'models/client/'

TRAIN_DATA_PICKLE = 'train_images.pkl'
EVAL_DATA_PICKLE = 'eval_images.pkl'
TEST_DATA_PICKLE = 'test_images.pkl'

BATCH_SIZE = 6
N_ITER = 50
EVALUATE_EVERY = 10
EVAL_BATCH_SIZE = 4

CLASSIFIER_THRESHOLD = 0.5
