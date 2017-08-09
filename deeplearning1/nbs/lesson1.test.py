from __future__ import division,print_function

import matplotlib
import os, json
from glob import glob
import numpy as np
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt


from imp import reload
import utils
reload(utils)
#reload(theano)
from utils import plots


path = "../data/dogscats/test/unknown/"
batch_size = 12

from vgg16 import Vgg16
import glob
import pandas as pd
import os

from keras.preprocessing import image


vgg = Vgg16()

def predict(test_folder, vgg):
	path = os.path.join(test_folder, '*.jpg') 
	images = glob.glob(path);

	records = []
	for n, path_img in enumerate(images):
	    if n%15 == 0:
		    print("Processing img #" + str(n))
	    probs = vgg.model.predict(
	        image.img_to_array(image.load_img(path_img, target_size=[224, 224])).reshape(1, 3,
                                                                     224,
                                                                     224))
	    number = os.path.split(path_img)[-1][0:-4]
	    records.append({'id': number, 'label': probs[0][1]})

	df = pd.DataFrame.from_records(records)
	df['id'] = pd.to_numeric(df['id'])
	df = df.sort_values('id')
	df.to_csv('submission.csv', index=False)

predict(path, vgg)


