import pickle
import gzip

import numpy as np


def load_data():
    with gzip.open('./mnist_data/mnist.pkl.gz', 'rb') as f:

        training, validation, test = pickle.load(f, encoding='latin1')

        """
        training - an array of tuples with the first entry being a row vector 
                   of each 784 pixel value and the second its classification
                   as a single alue (0-9)
        validation - same format as training
        test - ....
        """

        training_images = [np.reshape(img, (784)) for img in training[0]]
        training_results = []
        for y in training[1]:
            vec = np.zeros((10))
            vec[y] = 1.0
            training_results.append(vec)
        training = [training_images, training_results]

        validation = [
            [np.reshape(img, (784)) for img in validation[0]],
            validation[1]  
        ]

        test = [
            [np.reshape(img, (784)) for img in test[0]],
            test[1]
        ]

        return {
            'train': training, 
            'validate': validation,
            'test': test
}