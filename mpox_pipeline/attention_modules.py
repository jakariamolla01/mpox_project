"""
Attention modules: ECA, SEBlock, CBAM
Implemented as Keras layers, following the paper's equations (Eqs. 1-9).
"""
import tensorflow as tf
from tensorflow.keras import layers


class ECALayer(layers.Layer):
    """Efficient Channel Attention (Wang et al., 2020). Eq. 1-3 in the paper."""
    def __init__(self, k_size=3, **kwargs):
        super().__init__(**kwargs)
        self.k_size = k_size
        self.gap = layers.GlobalAveragePooling2D(keepdims=True)
        self.conv1d = layers.Conv1D(1, kernel_size=k_size, padding="same", use_bias=False)
        self.sigmoid = layers.Activation("sigmoid")

    def call(self, x):
        z = self.gap(x)                       # (B,1,1,C)
        z = tf.squeeze(z, axis=1)              # (B,1,C)
        z = self.conv1d(z)                     # (B,1,C)
        z = self.sigmoid(z)
        z = tf.expand_dims(z, axis=1)          # (B,1,1,C)
        return x * z


class SEBlock(layers.Layer):
    """Squeeze-and-Excitation (Hu et al., 2018). Eq. 4-6 in the paper."""
    def __init__(self, reduction=8, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        c = input_shape[-1]
        self.gap = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(c // self.reduction, activation="relu")
        self.fc2 = layers.Dense(c, activation="sigmoid")

    def call(self, x):
        s = self.gap(x)
        s = self.fc1(s)
        s = self.fc2(s)
        s = tf.reshape(s, [-1, 1, 1, tf.shape(x)[-1]])
        return x * s


class CBAM(layers.Layer):
    """Convolutional Block Attention Module (Woo et al., 2018). Eq. 7-9 in the paper."""
    def __init__(self, reduction=8, spatial_kernel=7, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
        self.spatial_kernel = spatial_kernel

    def build(self, input_shape):
        c = input_shape[-1]
        self.shared_fc1 = layers.Dense(c // self.reduction, activation="relu")
        self.shared_fc2 = layers.Dense(c, activation="sigmoid")
        self.spatial_conv = layers.Conv2D(1, self.spatial_kernel, padding="same",
                                           activation="sigmoid", use_bias=False)

    def call(self, x):
        # Channel attention
        avg_pool = layers.GlobalAveragePooling2D()(x)
        max_pool = layers.GlobalMaxPooling2D()(x)
        avg_out = self.shared_fc2(self.shared_fc1(avg_pool))
        max_out = self.shared_fc2(self.shared_fc1(max_pool))
        channel_att = tf.nn.sigmoid(avg_out + max_out)
        channel_att = tf.reshape(channel_att, [-1, 1, 1, tf.shape(x)[-1]])
        x = x * channel_att

        # Spatial attention
        avg_pool_s = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool_s = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool_s, max_pool_s], axis=-1)
        spatial_att = self.spatial_conv(concat)
        return x * spatial_att


ATTENTION_REGISTRY = {
    "eca": ECALayer,
    "se": SEBlock,
    "cbam": CBAM,
    "none": None,
}
