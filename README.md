# TF Boilerplate 

I found that it is not to hard for Tensorflow projects to become monoliths if one does not take some consideration when organizing their code. As a result, I decided to search look for some guidelines on how to structure a Tensorflow project and came across this popular template [link](https://github.com/MrGemy95/Tensorflow-Project-Template). Looking through the code, I found a lot to love, however, like any boilerplate I also found some things I did not like. This inspired me to write my own boilerplate, taking insipiration from what I liked in this project, while giving it my own flare.


## Structure

```bash
core/
    exceptions.py
    logger.py
    model.py
    trainer.py
    validators.py
logs/
snapshots/
main.py
```

- **core** - contains the main functionality of the boilerplate.
- **logs** - Where session logs are stored for tensorboard visualization.
- **snapshots** - Where models are saved.


## Core

As seen in the template this is based off of I also decided to use the same architecture consisting of a model, trainer and logger component. 

### Model

The model component is an abstract class containing the following: 

```python
class Model:
    """Base class for a Tensorflow models
    Args:
        config (dict): The configuration.
    """
    def __init__(self, config):
        config_validator(config)
        self.config = config
        self.name = self.config['name']
        self.saver = None
        with tf.variable_scope('global_step'):
            self.global_step = tf.Variable(
                0,
                trainable=False,
                name='global_step'
            )

        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        self.data = tf.data.Dataset.from_tensor_slices(
            (self.x, self.y)
        ).batch(
            config['batch_size']
        ).repeat()
        self.data_iter = self.data.make_initializable_iterator()

    def model_constructor(self):
        """Abstract method used to define the model in the derived class."""
        raise NotImplementedError

    def save(self, sess):
        """Saves a snapshot of the model
        Args:
            sess (tf.Session): The current session.
        """
        if self.saver is None:
            raise SaverNotInitialized((
                'saver not defined, please make sure '
                'self.saver = tf.train.Saver('
                'max_to_keep=config[\'saver_max_to_keep\']'
                ') is in the constructor.'
            ))
        print('Saving model...')
        save_dir = os.path.join(
            self.config['save_dir'],
            self.name
        )
        self.saver.save(sess, save_dir + '/' + self.name, self.global_step)
        print('Save completed.\n')

    def load(self, sess):
        """loads the last snapshot of the model
        Args:
            sess (tf.Session): The current session.
        """
        if self.saver is None:
            raise SaverNotInitialized((
                'saver not defined, please make sure '
                'self.saver = tf.train.Saver('
                'max_to_keep=config[\'saver_max_to_keep\']'
                ') is in the constructor.'
            ))
        save_dir = os.path.join(
            self.config['save_dir'],
            self.name
        )
        latest_checkpoint = tf.train.latest_checkpoint(save_dir)
        if latest_checkpoint:
            print(
                'Loading model checkpoint {} ...\n'.format(latest_checkpoint)
            )
        self.saver.restore(sess, latest_checkpoint)
```


This class functions the same way as the class seen in the template. That is, one will inherit from this class and implement the model_construtor function. The biggest difference here is that I decided to make use of the tf.data.Dataset API. In the constructor you will notice the following lines:


```python
self.x = tf.placeholder(tf.float32)
self.y = tf.placeholder(tf.float32)

self.data = tf.data.Dataset.from_tensor_slices(
    (self.x, self.y)
).batch(
    config['batch_size']
).repeat()
self.data_iter = self.data.make_initializable_iterator()
```

Here I create a placeholder for the feature vectors (x) and labels (y) and create a dataset. From this dataset I then create an initializatle iterator. This will be very benefical because later I will be able to feed in different datasets without having to rewrite parts of my code. 



### Trainer

The trainer base class has the most changes compared to the other project. It still functions the same, where one defines the epoch function, but includes the methods train, test, and predict seen below:

```python
def train(self, features, labels):
    """Iterates over epochs preforming the training_step for each epoch.
    
    Args:
        features (np.array): The feature vectors of the training set.
        labels (np.array): The labels of the training set.
    """
    with tf.name_scope('train'):
        self.sess.run(
            self.model.data_iter.initializer, 
            feed_dict={
                self.model.x: features,
                self.model.y: labels
            }
        )
        for epoch in range(self.model.config['num_epochs']):
            print(
                'Epoch %d/%d' % (epoch+1, self.model.config['num_epochs'])
            )
            self.epoch()

def test(self, features, labels):
    """Determines the loss and accuracy of the trained model on the 
    test set.
    
    Args:
        features (np.array): The feature vectors of the test set.
        labels (np.array): The labels of the test set.
    """
    self.sess.run(
        self.model.data_iter.initializer, 
        feed_dict={
            self.model.x: features,
            self.model.y: labels
        }
    )
    with tf.name_scope('test'):
        _, loss, accuracy = self.sess.run([
                self.model.optimizer,
                self.model.cross_entropy,
                self.model.accuracy
            ]
        )

        print('test cost: ', loss)
        print('test accuracy: ', accuracy)
        current_iteration = self.model.global_step.eval(self.sess)
        self.logger.log(
            'cost',
            loss,
            current_iteration,
            training=False
        )
        self.logger.log(
            'accuracy',
            accuracy,
            current_iteration,
            training=False
        )
        self.model.save(self.sess)

def predict(self, feature):
    """Abstract method to fetch a prediction.
    Args:
        feature (np.array): The feature vector to predict a label of.
    """
raise NotImplementedError
```


In the training method, one passes the training features and labels. Here you can see the power of the dataset API. Here the initalizer method of the data iterator is called and fed the training data. In a similar fashion the dataset API is also used in the test method to feed in the test set. Finally, I added a predict method, which is ment to be implemented by the child class in order to fetch a prediction for a single feature vector. 


### Logger


The logger is fairly straightfoward. It contains only 2 methods log and add_graph. The log method will log a value that is passed. It has optional arguments such as is_image and trainig. If is_image is true it will store the value as an image and if training is false it will log to the test summary. The add graph method will save the graph to the summary. This is useful when one wants to also view the graph in tensorboard.


## Example

Below is an example of how to use the core classes. The example below is just a simple MLP for the MNIST dataset.

```python
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from core.model import Model
from core.trainer import Trainer
from core.logger import Logger


class TestModel(Model):
    def __init__(self, config):
        super(TestModel, self).__init__(config)
        self.model_constructor()
        self.saver = tf.train.Saver(
            max_to_keep=self.config['saver_max_to_keep']
        )
    
    def model_constructor(self):
        """Constructs a simple MLP for the mnist dataset."""
        n_layer_1 = 256 # 1st layer number of neurons
        n_layer_2 = 256 # 2nd layer number of neurons
        n_input = 784 # MNIST data input (img shape: 28*28)
        n_classes = 10 # MNIST total classes (0-9 digits)

        with tf.name_scope('input'):
            self.features, self.labels = self.data_iter.get_next()

        with tf.name_scope('layer_1'):
            weights = tf.Variable(
                tf.random_normal([n_input, n_layer_1]),
                name='weights'
            )
            bias = tf.Variable(
                tf.random_normal([n_layer_1]),
                name='bias'
            )
            layer_1 = tf.sigmoid(
                tf.add(
                    tf.matmul(self.features, weights),
                    bias
                )
            )

        with tf.name_scope('layer_2'):
            weights = tf.Variable(
                tf.random_normal([n_layer_1, n_layer_2]),
                name='weights'
            )
            bias = tf.Variable(
                tf.random_normal([n_layer_2]),
                name='bias'
            )
            layer_2 = tf.sigmoid(
                tf.add(
                    tf.matmul(layer_1, weights),
                    bias
                )
            )

        with tf.name_scope('layer_3'):
            weights = tf.Variable(
                tf.random_normal([n_layer_2, n_classes]),
                name='weights'
            )
            bias = tf.Variable(
                tf.random_normal([n_classes]),
                name='bias'
            )
            output = tf.sigmoid(
                tf.add(
                    tf.matmul(layer_2, weights),
                    bias
                )
            )

        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.labels,
                logits=output
            )
        )

        self.optimizer = tf.train.AdamOptimizer(
            self.config['learning_rate']
        ).minimize(
            self.cross_entropy,
            global_step=self.global_step
        )

        self.pred = tf.nn.softmax(output)
        correct_prediction = tf.equal(
            tf.argmax(self.pred, 1),
            tf.argmax(self.labels, 1)
        )

        self.accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32)
        )

class TestTrainer(Trainer):
    def __init__(self, sess, model, logger):
        super(TestTrainer, self).__init__(sess, model, logger)

    def epoch(self):
        """The logic for a single epoch of training. Loops though the batches 
        and logs the average loss and accuracy.
        """
        batch_loop = tqdm(range(self.model.config['iterations_per_epoch']))
        losses = []
        accuracies = []
        for _ in batch_loop:
            loss, accuracy = self.batch_step()
            losses.append(loss)
            accuracies.append(accuracy)

        print('')
        print('cost: %f' % np.mean(losses))
        print('accuracy: %f \n' % np.mean(accuracies))


        current_iteration = self.model.global_step.eval(self.sess)
        self.logger.log(
            'cost',
            np.mean(losses),
            current_iteration
        )
        self.logger.log(
            'accuracy',
            np.mean(accuracies),
            current_iteration
        )
        self.model.save(self.sess)

    def batch_step(self):
        """Calculates the loss and accuracy per batch."""
        _, loss, accuracy = self.sess.run([
                self.model.optimizer,
                self.model.cross_entropy,
                self.model.accuracy
            ]
        )
        return loss, accuracy

    def predict(self, feature):
        prediction = self.sess.run(
            self.model.pred,
            feed_dict={
                self.model.features: [feature]
            }
        )
        return np.argmax(prediction)

def main():
    # Load mnist dataset
    train, test = tf.keras.datasets.mnist.load_data()

    # Build training set
    mnist_x, mnist_y = train
    vec_mnist_y = []
    for y in mnist_y:
        vec = np.zeros((10))
        vec[y] = 1.0
        vec_mnist_y.append(vec)
    train_x = mnist_x.reshape((60000, 784)).astype(np.float32)
    train_y = np.array(vec_mnist_y).astype(np.float32)

    # Build test set
    mnist_x, mnist_y = test
    vec_mnist_y = []
    for y in mnist_y:
        vec = np.zeros((10))
        vec[y] = 1.0
        vec_mnist_y.append(vec)
    test_x = mnist_x.reshape((mnist_x.shape[0], 784)).astype(np.float32)
    test_y = np.array(vec_mnist_y).astype(np.float32)

    config = {
        'name': 'TestModel',
        'batch_size': 100,
        'num_epochs': 5,
        'iterations_per_epoch': int(len(train_x)/100),
        'learning_rate': 0.001,
        'summary_dir': 'logs',
        'save_dir': 'snapshots',
        'saver_max_to_keep': 5
    }

    # Instantiate the session, logger, model and trainer
    sess = tf.Session()
    logger = Logger(sess, config)
    model = TestModel(config)
    trainer = TestTrainer(sess, model, logger)

    # Visualize the graph in Tensorboard
    logger.add_graph(sess.graph)

    # Load the model if pre-trained
    model.load(sess)

    # Train
    trainer.train(train_x, train_y)

    # Test
    trainer.test(test_x, test_y)
    
    # Get a prediction for a single feature vector
    prediction = trainer.predict(test_x[0])
    print('prediction: ', prediction)
    print('actual: ', np.argmax(test_y[0]))

if __name__ == '__main__':
    main()
```