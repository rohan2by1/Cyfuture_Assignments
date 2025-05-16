import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr

# Helper function to load and preprocess images
def load_img(path_to_img, max_dim=512):
    img = Image.open(path_to_img)
    long = max(img.size)
    scale = max_dim / long
    img = img.resize(
    (round(img.size[0]*scale), round(img.size[1]*scale)),
    Image.Resampling.LANCZOS
)
    img = np.array(img)
    img = img[tf.newaxis, :]
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        tensor = tensor[0]
    return Image.fromarray(tensor)

# VGG19 layer names for style/content
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# Build the model
def get_model():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs
    return tf.keras.models.Model(vgg.input, model_outputs)

# Gram matrix
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

# Extract features
def get_feature_representations(model, content_path, style_path):
    content_image = load_img(content_path)
    style_image = load_img(style_path)
    style_outputs = model(style_image)
    content_outputs = model(content_image)
    style_features = [style_layer for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_layer for content_layer in content_outputs[num_style_layers:]]
    return style_features, content_features

# Compute loss
def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights
    model_outputs = model(init_image)
    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    style_score = 0
    content_score = 0

    # Style loss
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        comb_style_gram = gram_matrix(comb_style)
        style_score += weight_per_style_layer * tf.reduce_mean((comb_style_gram - target_style) ** 2)

    # Content loss
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * tf.reduce_mean((comb_content - target_content)**2)

    style_score *= style_weight
    content_score *= content_weight

    loss = style_score + content_score
    return loss, style_score, content_score

# Optimization step
@tf.function()
def compute_grads(cfg):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)
    return tape.gradient(all_loss[0], cfg['init_image']), all_loss

# Main style transfer function
def run_style_transfer(content_path, style_path, num_iterations=200, content_weight=1e4, style_weight=1e-2):
    model = get_model()
    for layer in model.layers:
        layer.trainable = False

    # Get the style and content feature representations (from our specified intermediate layers)
    style_features, content_features = get_feature_representations(model, content_path, style_path)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

    # Set initial image
    init_image = load_img(content_path)
    init_image = tf.Variable(init_image, dtype=tf.float32)

    # Create config
    opt = tf.optimizers.Adam(learning_rate=0.02)
    loss_weights = (style_weight, content_weight)
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features
    }

    best_loss, best_img = float('inf'), None

    for i in range(num_iterations):
        grads, all_loss = compute_grads(cfg)
        loss, style_score, content_score = all_loss
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, 0.0, 1.0)
        init_image.assign(clipped)
        if loss < best_loss:
            best_loss = loss
            best_img = init_image.numpy()
        if i % 50 == 0:
            print(f"Iteration {i}, Loss: {loss.numpy()}, style: {style_score.numpy()}, content: {content_score.numpy()}")

    return tensor_to_image(best_img)

# Gradio UI
def neural_style_transfer_ui(content_img, style_img, steps, style_weight, content_weight):
    result = run_style_transfer(content_img, style_img, int(steps), float(content_weight), float(style_weight))
    return result

demo = gr.Interface(
    fn=neural_style_transfer_ui,
    inputs=[
        gr.Image(type="filepath", label="Content Image"),
        gr.Image(type="filepath", label="Style Image"),
        gr.Number(value=100, label="Steps (iterations)"),
        gr.Number(value=1e-2, label="Style weight"),
        gr.Number(value=1e4, label="Content weight"),
    ],
    outputs=gr.Image(type="pil", label="Stylized Image"),
    title="Neural Style Transfer Demo",
    description="Upload content and style images, adjust parameters, and generate a stylized image!"
)

if __name__ == "__main__":
    demo.launch()