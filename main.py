import tensorflow as tf
import numpy as np
from tqdm import tqdm
from core.model import Model
from core.train import Train
from core.logger import Logger
# Get MNIST dataset
from mnist_loader import load_data


class TestModel(Model):
    def __init__(self, config, data):
        super(TestModel, self).__init__(config, data)
        self.model_constructor()
        self.saver_init()
    
    def saver_init(self):
        self.saver = tf.train.Saver(max_to_keep=self.config['max_to_keep'])

    def model_constructor(self):
        n_layer_1 = 256 # 1st layer number of neurons
        n_layer_2 = 256 # 2nd layer number of neurons
        n_input = 784 # MNIST data input (img shape: 28*28)
        n_classes = 10 # MNIST total classes (0-9 digits)

        with tf.name_scope('input'):
            batch_x, batch_y = self.data.get_next()
            self.x = tf.cast(batch_x, tf.float32)
            self.y = tf.cast(batch_y, tf.float32)
        # Build layers 
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
                    tf.matmul(self.x, weights),
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

        with tf.name_scope('loss'):
            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=self.y,
                    logits=output
                )
            )
            self.optimizer = tf.train.AdamOptimizer(
                self.config['learning_rate']
            ).minimize(
                self.cross_entropy,
                global_step=self.global_step
            )
            pred = tf.nn.softmax(output)
            correct_prediction = tf.equal(
                tf.argmax(pred, 1),
                tf.argmax(self.y, 1)
            )
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32)
            )

class TestTrain(Train):
    def __init__(self, sess, model, data, config, logger):
        super(TestTrain, self).__init__(sess, model, data, config, logger)

    def training_step(self):
        batch_loop = tqdm(range(self.config['iterations_per_batch']))
        losses = []
        accuracies = []
        for _ in batch_loop:
            loss, accuracy = self.batch_step()
            losses.append(loss)
            accuracies.append(accuracy)

        print('cost: %f' % np.mean(losses))
        print('accuracy: %f \n' % np.mean(accuracies))


        current_iteration = self.model.global_step.eval(self.sess)
        items_to_log = {
            'cost': np.mean(losses),
            'accuracy': np.mean(accuracies)
        }
        self.logger.log(current_iteration, items_to_log)
        self.model.save(self.sess)

    def batch_step(self):

        _, loss, accuracy = self.sess.run([
                self.model.optimizer,
                self.model.cross_entropy,
                self.model.accuracy
            ]
        )

        return loss, accuracy

def main():
    # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    train, test = tf.keras.datasets.mnist.load_data()
    mnist_x, mnist_y = train
    vec_mnist_y = []
    for y in mnist_y:
        vec = np.zeros((10))
        vec[y] = 1.0
        vec_mnist_y.append(vec)

    mnist_x = mnist_x.reshape((60000, 784))
    mnist_y = np.array(vec_mnist_y)

    """
    mnist = load_data()
    mnist_x = np.array(mnist['train'][0])
    mnist_y = np.array(mnist['train'][1])
    print(mnist_x.shape)
    """
    data = tf.data.Dataset.from_tensor_slices((mnist_x, mnist_y))
    data = data.batch(100).repeat()
    data_it = data.make_one_shot_iterator()

    config = {
        'name': 'TestModel',
        'batch_size': 100,
        'iterations_per_batch': int(len(mnist_x)/100),
        'learning_rate': 0.001,
        'max_epoch': 5,
        'summary_dir': 'logs',
        'save_dir': 'snapshots',
        'max_to_keep': 5
    }

    sess = tf.Session()
    logger = Logger(sess, config)
    model = TestModel(config, data_it)
    logger.add_graph(sess.graph)
    trainer = TestTrain(sess, model, data_it, config, logger)
    model.load(sess)
    trainer.train()


if __name__ == '__main__':
    main()