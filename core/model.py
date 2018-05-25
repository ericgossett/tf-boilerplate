import tensorflow as tf
import os

from .validators import config_validator
from .exceptions import SaverNotInitialized


class Model:
    """Base class for a Tensorflow models

    Args:
        config (dict): The configuration.
    """
    def __init__(self, config, data):
        config_validator(config)
        self.config = config
        self.data = data
        self.name = self.config['name']
        self.saver = None
        with tf.variable_scope('global_step'):
            self.global_step = tf.Variable(
                0,
                trainable=False,
                name='global_step'
            )

    def model_constructor(self):
        """Abstract method used to define the model in the derived class."""
        raise NotImplementedError

    def saver_init(self):
        """Abstract method used to initalize the saver in the derived class.

        In the dervied class addd the following to the body:

        >>> self.saver = tf.train.Saver(max_to_keep=config['max_to_keep'])

        Then call this method in the derived classes constructor.
        """
        # self.saver = tf.train.Saver(max_to_keep=config['max_to_keep'])
        raise NotImplementedError

    def save(self, sess):
        """Saves a snapshot of the model

        Args:
            sess (tf.Session): The current session.
        """
        if self.saver is None:
            raise SaverNotInitialized((
                'saver not defined, please make sure '
                'saver_init() is called in the constructor.'
            ))
        print('Saving model...')
        save_dir = os.path.join(
            self.config['save_dir'],
            self.name
        )
        self.saver.save(sess, save_dir + '/' + self.name, self.global_step)
        print('Save completed.')

    def load(self, sess):
        """loads the last snapshot of the model

        Args:
            sess (tf.Session): The current session.
        """
        if self.saver is None:
            raise SaverNotInitialized((
                'saver not defined, please make sure '
                'saver_init() is called in the constructor.'
            ))
        save_dir = os.path.join(
            self.config['save_dir'],
            self.name
        )
        latest_checkpoint = tf.train.latest_checkpoint(save_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
