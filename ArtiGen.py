import os
import numpy as np
import imageio
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose, concatenate, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import cv2

# Define constants
IMAGE_SIZE = (256, 256, 3)
LATENT_DIM = 100
NUM_CLASSES = 2  # Number of classes (e.g., artists)
EPOCHS = 1000
BATCH_SIZE = 16
SAMPLE_INTERVAL = 100

# Define the generator model
def build_generator():
    input_layer = Input(shape=(LATENT_DIM,))
    input_label = Input(shape=(1,), dtype='int32')
    
    # Embedding layer for label
    label_embedding = Embedding(NUM_CLASSES, LATENT_DIM)(input_label)
    label_embedding = Flatten()(label_embedding)
    
    # Concatenate the noise and label
    combined = concatenate([input_layer, label_embedding])
    x = Dense(64 * 64 * 64, activation="relu")(combined)
    x = Reshape((64, 64, 64))(x)
    x = Conv2DTranspose(64, kernel_size=3, strides=2, padding="same", activation="relu")(x)
    x = Conv2DTranspose(32, kernel_size=3, strides=2, padding="same", activation="relu")(x)
    x = Conv2DTranspose(3, kernel_size=3, strides=1, padding="same", activation="sigmoid")(x)
    model = Model([input_layer, input_label], x)
    return model

# Define the discriminator model
def build_discriminator():
    input_layer = Input(shape=IMAGE_SIZE)
    input_label = Input(shape=(1,), dtype='int32')
    
    # Embedding layer for label
    label_embedding = Embedding(NUM_CLASSES, np.prod(IMAGE_SIZE))(input_label)
    label_embedding = Reshape(IMAGE_SIZE)(label_embedding)
    
    # Concatenate the image and label
    combined = concatenate([input_layer, label_embedding])
    
    x = Conv2D(32, kernel_size=3, strides=2, padding="same", activation="relu")(combined)
    x = Conv2D(64, kernel_size=3, strides=2, padding="same", activation="relu")(x)
    x = Conv2D(128, kernel_size=3, strides=2, padding="same", activation="relu")(x)
    x = Flatten()(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model([input_layer, input_label], x)
    return model
    
# Load images
def load_images(image1_path, image2_path):
    try:
        image1 = cv2.imread(image1_path)
        if image1 is None:
            print(f"Failed to load image1 from {image1_path}")
            return None, None
        image1 = cv2.resize(image1, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
        image1 = image1 / 255.0

        image2 = cv2.imread(image2_path)
        if image2 is None:
            print(f"Failed to load image2 from {image2_path}")
            return None, None
        image2 = cv2.resize(image2, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
        image2 = image2 / 255.0

        return image1, image2
    except Exception as e:
        print("Error loading images:", e)
        return None, None

# Combine images using the generator
def combine_images(generator, latent_dim, image1, image2, label):
    latent_noise = np.random.normal(0, 1, (1, latent_dim))
    label = np.array([[label]])
    generated_image = generator.predict([latent_noise, label])[0]

    combined_image = generated_image * image1 + (1 - generated_image) * image2
    combined_image = (combined_image * 255).astype(np.uint8)
    return combined_image

# Train the GAN
def train_gan(generator, discriminator, combined_model, images, labels):
    combined_model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    for epoch in range(EPOCHS):
        idx = np.random.randint(0, images.shape[0], BATCH_SIZE)
        real_images = images[idx]
        real_labels = labels[idx]
        latent_noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
        generated_images = generator.predict([latent_noise, real_labels])

        # Train the discriminator
        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch([real_images, real_labels], np.ones((BATCH_SIZE, 1)))
        d_loss_fake = discriminator.train_on_batch([generated_images, real_labels], np.zeros((BATCH_SIZE, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train the generator
        discriminator.trainable = False
        g_loss = combined_model.train_on_batch([latent_noise, real_labels], np.ones((BATCH_SIZE, 1)))

        print(f"Epoch {epoch}, D Loss: {d_loss}, G Loss: {g_loss}")
        if(epoch==20):
            break

def main(image1_path, image2_path, output_path):
    try:
        # Load images
        image1, image2 = load_images(image1_path, image2_path)

        # Generate labels for the images (e.g., artist labels)
        label1 = 0  # Label for image1
        label2 = 1  # Label for image2

        # Build and compile the discriminator
        discriminator = build_discriminator()
        discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

        # Build and compile the generator
        generator = build_generator()

        # Combine discriminator and generator to form the GAN
        discriminator.trainable = False
        gan_input = Input(shape=(LATENT_DIM,))
        gan_label = Input(shape=[1,], dtype='int32')
        combined_output = discriminator([generator([gan_input, gan_label]), gan_label])
        combined_model = Model([gan_input, gan_label], combined_output)
        combined_model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

        # Train GAN
        images = np.array([image1, image2])
        labels = np.array([[label1], [label2]])
        train_gan(generator, discriminator, combined_model, images, labels)

        # Generate combined image
        combined_image = combine_images(generator, LATENT_DIM, image1, image2, label1)

        # Save combined image
        cv2.imwrite(output_path, combined_image)

        return True, "Combined image generated successfully."

    except Exception as e:
        return False, f"An error occurred: {str(e)}"

if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Combine two images using GAN")
    parser.add_argument("image1_path", type=str, help="Path to the first image")
    parser.add_argument("image2_path", type=str, help="Path to the second image")
    parser.add_argument("output_path", type=str, help="Path to save the combined image")
    args = parser.parse_args()

    # Call main function with provided paths
    success, message = main(args.image1_path, args.image2_path, args.output_path)
    if success:
        print(message)
    else:
        print("Error:", message)
