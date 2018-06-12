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