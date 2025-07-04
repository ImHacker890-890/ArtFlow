import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np


IMG_SIZE = 128
BATCH_SIZE = 16
TEXT_EMBEDDING_DIM = 512
TIMESTEPS = 1000
EPOCHS = 50


images, texts = load_coco_data()


import tensorflow_hub as hub
text_encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


def build_unet():
    
    img_input = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    t_input = layers.Input((1,))
    text_input = layers.Input((TEXT_EMBEDDING_DIM,))
    
    
    t_emb = layers.Dense(IMG_SIZE * IMG_SIZE)(t_input)
    t_emb = layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(t_emb)
    
    
    text_emb = layers.Dense(IMG_SIZE * IMG_SIZE)(text_input)
    text_emb = layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(text_emb)
    
    
    x = layers.Concatenate()([img_input, t_emb, text_emb])
    
    # Downsample
    x = layers.Conv2D(64, 3, padding="same", activation="swish")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="swish")(x)
    x = layers.MaxPooling2D()(x)
    
    # Upsample
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="swish")(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="swish")(x)
    
    
    output = layers.Conv2D(3, 3, padding="same", activation="tanh")(x)
    
    return Model([img_input, t_input, text_input], output)

model = build_unet()
model.summary()


def forward_diffusion(x0, t):
    noise = tf.random.normal(shape=x0.shape)
    alpha = 1.0 - (t / TIMESTEPS)
    return tf.sqrt(alpha) * x0 + tf.sqrt(1 - alpha) * noise, noise


optimizer = tf.keras.optimizers.Adam(1e-4)
loss_fn = tf.keras.losses.MeanSquaredError()


text_embeddings = text_encoder(texts).numpy()

dataset = tf.data.Dataset.from_tensor_slices((images, text_embeddings))
dataset = dataset.shuffle(1000).batch(BATCH_SIZE)

for epoch in range(EPOCHS):
    for batch_imgs, batch_text in dataset:
        t = tf.random.uniform((BATCH_SIZE, 1), 0, 1)
        
        with tf.GradientTape() as tape:
            x_noisy, noise = forward_diffusion(batch_imgs, t)
            pred_noise = model([x_noisy, t, batch_text], training=True)
            loss = loss_fn(noise, pred_noise)
        
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    print(f"Epoch {epoch + 1}, Loss: {loss.numpy()}")

model.save("diffusion_model.h5")
