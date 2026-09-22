"""
Builds base / attention-augmented models exactly per Figure 4 and Figure 8 of the paper.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model, applications
from attention_modules import ATTENTION_REGISTRY

# name -> (keras.applications class, preprocess_input fn, default input size)
BACKBONES = {
    "MobileNetV2":       (applications.MobileNetV2, applications.mobilenet_v2.preprocess_input, 224),
    "InceptionV3":       (applications.InceptionV3, applications.inception_v3.preprocess_input, 299),
    "InceptionResNetV2": (applications.InceptionResNetV2, applications.inception_resnet_v2.preprocess_input, 299),
    "ResNet50V2":        (applications.ResNet50V2, applications.resnet_v2.preprocess_input, 224),
    "ResNet101V2":       (applications.ResNet101V2, applications.resnet_v2.preprocess_input, 224),
    "DenseNet121":       (applications.DenseNet121, applications.densenet.preprocess_input, 224),
    "DenseNet201":       (applications.DenseNet201, applications.densenet.preprocess_input, 224),
    "EfficientNetB0":    (applications.EfficientNetB0, applications.efficientnet.preprocess_input, 224),
    "EfficientNetB2":    (applications.EfficientNetB2, applications.efficientnet.preprocess_input, 260),
    "EfficientNetV2S":   (applications.EfficientNetV2S, applications.efficientnet_v2.preprocess_input, 384),
}


def _dense_head(x, num_classes):
    """Three dense blocks + BN + Dropout(0.3), per Figure 4/8."""
    for units in (1024, 256, 128):
        x = layers.Dense(units, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
    # dtype='float32' on the final layer keeps softmax numerically stable under mixed precision
    return layers.Dense(num_classes, activation="softmax", dtype="float32")(x)


def build_model(backbone_name, num_classes, attention_type="none",
                 attention_depth="single", fine_tune=False, input_size=None):
    """
    attention_type: 'none' | 'eca' | 'se' | 'cbam'
    attention_depth: 'single' | 'double'  (double inserts Conv2D(256) between two attention blocks)
    fine_tune: if False, backbone is frozen (transfer learning as in the paper's baseline runs)
    """
    backbone_cls, preprocess_fn, default_size = BACKBONES[backbone_name]
    size = input_size or default_size
    base = backbone_cls(include_top=False, weights="imagenet", input_shape=(size, size, 3))
    base.trainable = fine_tune

    x = base.output  # feature map (H, W, C)

    att_cls = ATTENTION_REGISTRY.get(attention_type)
    if att_cls is not None:
        x = att_cls()(x)
        if attention_depth == "double":
            x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
            x = att_cls()(x)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = _dense_head(x, num_classes)

    model = Model(inputs=base.input, outputs=outputs,
                   name=f"{backbone_name}_{attention_type}_{attention_depth}")
    return model, preprocess_fn, size


if __name__ == "__main__":
    m, pfn, sz = build_model("ResNet50V2", num_classes=6, attention_type="cbam", attention_depth="single")
    m.summary()
