import tensorflow as tf


class BalancedAccuracy(tf.keras.metrics.Metric):
    """
    Computes Balanced Accuracy: (Sensitivity + Specificity) / 2
    """

    def __init__(self, name='balanced_accuracy', threshold=0.3, **kwargs):
        super(BalancedAccuracy, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.tn = self.add_weight(name='tn', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        self.tp.assign_add(tf.reduce_sum(y_true * y_pred))
        self.tn.assign_add(tf.reduce_sum((1 - y_true) * (1 - y_pred)))
        self.fp.assign_add(tf.reduce_sum((1 - y_true) * y_pred))
        self.fn.assign_add(tf.reduce_sum(y_true * (1 - y_pred)))

    def result(self):
        sensitivity = tf.math.divide_no_nan(self.tp, self.tp + self.fn)
        specificity = tf.math.divide_no_nan(self.tn, self.tn + self.fp)
        return (sensitivity + specificity) / 2.0

    def reset_state(self):
        self.tp.assign(0.0)
        self.tn.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)

# ===================================================================
# 2. Load the model with the custom metric class
# ===================================================================


# yes we can !
