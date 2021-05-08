import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt
import numpy as np


class Dataset:
    def __init__(self, dataset_name='omniglot'):
        ds_train, self.ds_test = tfds.load(dataset_name, split=["train", "test"], as_supervised=True, shuffle_files=False)

        self.data = {}

        def extraction(image, label):
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.rgb_to_grayscale(image)
            image = tf.image.resize(image, [32, 32])
            return image, label

        for image, label in ds_train.map(extraction):
            image = image.numpy()
            label = str(label.numpy())
            if label not in self.data:
                self.data[label] = []
            self.data[label].append(image)
        self.labels = list(self.data.keys())

    def init_test_dataset(self):
        def extraction(image, label):
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.rgb_to_grayscale(image)
            image = tf.image.resize(image, [32, 32])
            return image, label
        
        for image, label in self.ds_test.map(extraction):
            image = image.numpy()
            label = str(label.numpy())
            if label not in self.test_data:
                self.test_data[label] = []
            self.test_data[label].append(image)
        self.test_labels = list(self.test_data.keys())
        
    def shuffle_labels(self):
        np.random.shuffle(self.labels)

    def get_mini_dataset(self, label_idx, batch_size=20):
        #random_label = self.labels[random.randint(0, len(self.labels)-1)]
        label = self.labels[label_idx]
        indices = np.random.choice(len(self.data[label]), batch_size, replace=False)
        dataset = np.array(self.data[label])[indices].astype(np.float32)
        return dataset
        
    def get_test_dataset(self, label_idx, batch_size=20):
        #random_label = self.test_labels[random.randint(0, len(self.test_labels)-1)]
        label = self.labels[label_idx]
        indices = np.random.choice(len(self.test_data[label]), batch_size, replace=False)
        dataset = np.array(self.test_data[label])[indices].astype(np.float32)

        return dataset

"""
    Define GAN model
"""

def define_discriminator(in_shape=(32, 32, 1)):
    model = keras.Sequential([
        layers.Conv2D(64, (3, 3), strides=(2,2), padding='same', input_shape=in_shape),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.4),
        layers.Conv2D(64, (3, 3), strides=(2,2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def define_generator(input_dim):
    n_nodes = 128 * 8 * 8 # foundation for 8x8 image
    model = keras.Sequential([
        layers.Dense(n_nodes, input_dim=input_dim),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((8,8,128)),
        # Upsample to 16x16
        layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        # Upsample to 32x32
        layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(1, (7,7), activation='sigmoid', padding='same')
        
    ])
    
    return model

def define_gan(g_model, d_model):
    d_model.trainable = False
    model = keras.Sequential([
        g_model, d_model
    ])
    opt = keras.optimizers.Adam(lr=0.0001, beta_1 = 0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

"""
    Define data generation functions
"""

def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

def generate_real_samples(dataset, label_idx, n_samples=20):
    images = dataset.get_mini_dataset(label_idx, n_samples)  # in a numpy array format
    labels = np.ones((len(images), 1))
    data = tf.data.Dataset.from_tensor_slices((images, labels))
    data = data.shuffle(100).batch(n_samples) # data.shape = (n_samples, 28, 28, 1)
    return data

def generate_fake_samples(g_model, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X_gan = g_model(x_input) # generated fake image shape: (n_samples, 28, 28, 1)
    y = np.zeros((n_samples, 1))
    data = tf.data.Dataset.from_tensor_slices((X_gan, y))
    data = data.shuffle(100).batch(n_samples)    
    
    return data

def generate_train_samples(latent_dim, n_samples):
    X_gan = generate_latent_points(latent_dim, n_samples)
    y_gan = np.ones((n_samples, 1))

    data = tf.data.Dataset.from_tensor_slices((X_gan, y_gan))
    data = data.shuffle(100).batch(n_samples)    
    
    return data

"""
    Define testing functions
"""

def save_plot(examples, epoch, n=4):
    for i in range(n):
        plt.subplot(1, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i, :, :, 0], cmap='gray_r')
    filename='img/generated_plot_e%03d.png'%(epoch+1)
    plt.savefig(filename)
    plt.close()

def compare_plot(real_examples, fake_examples, epoch, n_samples):
    for i in range(2):
        for j in range(n_samples):
            plt.subplot(2, n_samples, 1 + (i*n_samples) + j)
            plt.axis('off')
            if i == 0:
                plt.imshow(real_examples[j, :, :, 0], cmap='gray_r')
            elif i == 1:
                plt.imshow(fake_examples[j, :, :, 0], cmap='gray_r')
    filename='img/compare_plot_e%03d.png'%(epoch+1)
    plt.savefig(filename)
    plt.close()
    
def summarize_performance(epoch, g_model, d_model, gan_model, dataset, latent_dim, n_samples):
    # Algorithm 2. FIGR Generation
    dataset.shuffle_labels()
    real_data = generate_real_samples(dataset, 0, n_samples)
    fake_data = generate_fake_samples(g_model, latent_dim, n_samples)
    
    all_data = real_data.concatenate(fake_data).shuffle(100)

    d_model.fit(all_data, epochs=1, verbose=0)

    
    train_data = generate_train_samples(latent_dim, n_samples)
    gan_model.fit(train_data, epochs=5, verbose=0)
    
    fake_data = generate_fake_samples(g_model, latent_dim, n_samples)
   
    _, acc_real = d_model.evaluate(real_data, verbose=0)
    _, acc_fake = d_model.evaluate(fake_data, verbose=0)
    
    print(">Round {}\n\tDiscriminator accuracy real: {:.2f}% fake: {:.2f}%".format(epoch, acc_real*100, acc_fake*100))
    
    x_input = generate_latent_points(latent_dim, n_samples)
    X_fake = g_model(x_input)
    X_real = dataset.get_mini_dataset(0, n_samples)

    compare_plot(X_real, X_fake, epoch, n_samples)
    filename_g='model/generator_model_%03d.h5' % (epoch+1)
    filename_d='model/discriminator_model_%03d.h5' % (epoch+1)
    g_model.save(filename_g)
    d_model.save(filename_d)

"""
    Define training functions
"""

def weight_difference(old_weights, weights):
    new_weights = list()
    for i in range(len(weights)):
        new_weights.append((old_weights[i] - weights[i]))
    return new_weights

def inner_loop(g_model, d_model, gan_model, dataset, n_samples, inner_loops, latent_dim, label_idx):
    # Sample a task from the training dataset 
    real_data = generate_real_samples(dataset, label_idx, n_samples)
    # Generate latent vectors -> generate fake images
    fake_data = generate_fake_samples(g_model, latent_dim, n_samples)
            
    all_data = real_data.concatenate(fake_data).shuffle(100)
    
    d_model.fit(all_data, epochs=inner_loops, verbose=0)
                    
    # Train the generator
    train_data = generate_train_samples(latent_dim, n_samples)
    gan_model.fit(train_data, epochs=inner_loops, verbose=0)

def reptile_train(g_model, d_model, gan_model, 
                  dataset, latent_dim, n_samples,
                  inner_loops = 10, n_epochs=200, meta_step_size=0.00001):
        
    # Algorithm 1. FIGR training
    
    opt_g = keras.optimizers.Adam(learning_rate=meta_step_size)
    opt_d = keras.optimizers.Adam(learning_rate=meta_step_size)
    
    # How many epochs the model should train for
    for i in range(n_epochs):
        # On how many labels/classes the model should be optimized on per epoch
        for j in range(501):
            # Make a copy of phi_generator and phi_discriminator
            phi_g = g_model.get_weights()
            phi_d = d_model.get_weights()

            # Train the GAN model & discriminator
            inner_loop(g_model, d_model, gan_model, dataset, n_samples, inner_loops, latent_dim, j)
        
            # Get the newly trained model weights
            g_weights = g_model.get_weights()
            g_model.set_weights(g_weights)

            d_weights = d_model.get_weights()
            d_model.set_weights(d_weights)

            # Calculate weight difference between optimized weights and original weights
            g_grads = weight_difference(g_weights, phi_g)
            d_grads = weight_difference(d_weights, phi_d)

            opt_g.apply_gradients(zip(g_grads, g_model.trainable_weights))
            opt_d.apply_gradients(zip(d_grads, d_model.trainable_weights))

        dataset.shuffle_labels()

        if (i + 1) % 10 == 0:
            summarize_performance(i+1, g_model, d_model, gan_model, dataset, latent_dim, n_samples)

def main():
    # Tensorflow GPU settings
    #gpu_num = 0
    #gpus = tf.config.experimental.list_physical_devices('GPU')
    #if gpus:
    #    tf.config.experimental.set_visible_devices(gpus[gpu_num], 'GPU')
    #    tf.config.experimental.set_memory_growth(gpus[gpu_num], True)

    dataset = Dataset()

    latent_dim=100
    n_samples=5
    epochs=200


    d_model = define_discriminator()
    g_model = define_generator(latent_dim)
    gan_model = define_gan(g_model, d_model)

    reptile_train(g_model, d_model, gan_model, dataset, latent_dim, n_samples, n_epochs=epochs)

if __name__ == '__main__':
  main()
