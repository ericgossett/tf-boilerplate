import os
import tensorflow as tf

class Logger:
    def __init__(self, sess, config):
        self.summary_dir = config['summary_dir']
        self.sess = sess
        self.placeholders = {}
        self.ops = {}
        self.train_summary_writer = tf.summary.FileWriter(
            os.path.join(self.summary_dir, 'training')
        )
        self.test_summary_writer = tf.summary.FileWriter(
            os.path.join(self.summary_dir, 'testing')
        )

    def log(self, step, items_to_log, summary_type='train'):
        summary_writer = (
            self.train_summary_writer if summary_type == 'train' else 
            self.test_summary_writer
        )
        summary_list = []
        for key, val in items_to_log.items():
            if key not in self.placeholders:
                with tf.name_scope(key):
                    if len(val.shape) <= 1:
                        self.placeholders[key] = tf.placeholder(
                            'float32',
                            val.shape,
                            name=key
                        )
                        self.ops[key] = tf.summary.scalar(
                            'value',
                            self.placeholders[key]
                        )
                        self.ops[key+'-hist'] = tf.summary.histogram(
                            'histogram',
                            self.placeholders[key]
                        )
                    else:
                        self.placeholders[key] = tf.placeholder(
                            'float32',
                            [None] + list(val.shape[1:]),
                            name=key
                        )
                        self.ops[key] = tf.summary.image(
                            key,
                            self.placeholders[key]
                        ) 
            summary_list.extend((
                self.sess.run(
                    self.ops[key], {self.placeholders[key]: val}
                ),
                self.sess.run(
                    self.ops[key+'-hist'], {self.placeholders[key]: val}
                )
            ))
        for summary in summary_list:
            summary_writer.add_summary(summary, step)
        summary_writer.flush()

    def add_graph(self, graph, summary_type='train'):
        summary_writer = (
            self.train_summary_writer if summary_type == "train" else 
            self.test_summary_writer
        )

        summary_writer.add_graph(graph)