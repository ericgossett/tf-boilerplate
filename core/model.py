import tensorflow as tf
import os

class SaverNotInitialized(Exception):
    pass

class Model:
    def __init__(self, config):
        self.config = config
        self.saver = None
        with tf.variable_scope('global_step'):
            self.global_step = tf.Variable(
                0,
                trainable=False,
                name='global_step'
            )

    def model_constructor(self):
        raise NotImplementedError

    def saver_init(self):
        # self.saver = tf.train.Saver(max_to_keep=config['max_to_keep'])
        raise NotImplementedError

    def save(self, sess):
        if self.saver is None:
            raise SaverNotInitialized('saver not defined, please make sure saver_init() is called in constructor.')
        print('Saving model...')
        name = self.__class__.__name__
        save_dir = os.path.join(
            self.config['save_dir'],
            name
        )
        self.saver.save(sess, save_dir + '/' + name, self.global_step)
        print('Save completed.')

    def load(self, sess):
        if self.saver is None:
            raise SaverNotInitialized('saver not defined, please make sure saver_init() is called in constructor.')
        name = self.__class__.__name__
        save_dir = os.path.join(
            self.config['save_dir'],
            name
        )
        latest_checkpoint = tf.train.latest_checkpoint(save_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
