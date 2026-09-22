"""
Grad-CAM (fully implemented) + usage notes for SHAP and LIME (paper's 3 XAI methods).
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:  # (B, H, W, C)
            return layer.name
    raise ValueError("No conv layer found.")


def grad_cam(model, img_array, class_index=None, last_conv_layer_name=None):
    """
    img_array: preprocessed (1, H, W, 3) array, already normalized as the model expects.
    Returns a (H, W) heatmap normalized to [0, 1].
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    grad_model = tf.keras.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        class_channel = predictions[:, class_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(img, heatmap, alpha=0.4, save_path=None):
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], (img.shape[0], img.shape[1])).numpy()
    heatmap_resized = np.uint8(255 * heatmap_resized.squeeze())
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_resized]
    superimposed = jet_heatmap * alpha + img / 255.0
    superimposed = np.clip(superimposed, 0, 1)
    if save_path:
        plt.imsave(save_path, superimposed)
    return superimposed


# --- SHAP (paper: pixel/superpixel attribution, quantitative) ---
# pip install shap
#   import shap
#   masker = shap.maskers.Image("inpaint_telea", img_array[0].shape)
#   explainer = shap.Explainer(model, masker, output_names=classes)
#   shap_values = explainer(img_array, max_evals=500, batch_size=32)
#   shap.image_plot(shap_values)

# --- LIME (paper: local superpixel-based explanation) ---
# pip install lime
#   from lime import lime_image
#   explainer = lime_image.LimeImageExplainer()
#   explanation = explainer.explain_instance(img_array[0], model.predict, top_labels=1,
#                                             hide_color=0, num_samples=1000)
#   temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
#                                                positive_only=True, num_features=5)
