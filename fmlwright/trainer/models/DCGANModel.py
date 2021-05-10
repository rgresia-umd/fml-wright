import logging

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from fmlwright.core.utils import create_z_random
from fmlwright.trainer.models import BaseModel
from fmlwright.trainer.neural_networks.networks import (
    create_discriminator,
    create_generator,
)

log = logging.getLogger(__name__)


class DCGAN(BaseModel):
    """DCGAN Model"""

    def __init__(self, conf):
        """Initialize the DCGAN.
        Args:
            conf (dict): loaded configuration file.
        """
        super().__init__(conf)

        conf_generator = conf["nn_structure"]["generator"]
        self.G = load_model("./results/floorplan/living_room/DCGAN/20210509-181947/models/generator.h5")
            #create_generator(conf_generator, self.input_shape)
        self.noise_dim = conf["nn_structure"]["generator"]["noise_dim"]

        conf_discriminator = conf["nn_structure"]["discriminator"]
        #create_discriminator(conf_discriminator, self.input_shape)
        self.D = load_model("./results/floorplan/living_room/DCGAN/20210509-181947/models/discriminator.h5")


        self.disc_loss_function = (
            BinaryCrossentropy(from_logits=True)
        )
        self.gen_loss_function = BinaryCrossentropy(from_logits=True)

        if self.ttur:
            self.D_optimizer = Adam(learning_rate=self.d_lr)
            self.G_optimizer = Adam(learning_rate=self.g_lr)
        else:
            self.D_optimizer = Adam(learning_rate=self.lr)
            self.G_optimizer = Adam(learning_rate=self.lr)

        self.disc_optimizers = [self.D_optimizer]
        self.generator_optimizers = [self.G_optimizer]

    def calc_G_loss(self, fake_output):
        """Calculate the G loss.
        Args:
            fake_output (tf.tensor): discriminator's evaluation of fake images
        Returns:
            Tensor with a specific part of just the G loss.
        """
        return self.gen_loss_function(tf.ones_like(fake_output), fake_output)

    def calc_D_loss(self,  real_output, fake_output):
        """Calculate the D loss.
        Args:
            fake_output (tf.tensor): discriminator's evaluation of fake images
            real_output (tf.tensor): discriminator's evaluation of real images
        Returns:
            tensors with the D loss parts.
        """
        D_real_loss = self.disc_loss_function(tf.ones_like(real_output), real_output)

        D_fake_loss = self.disc_loss_function(tf.zeros_like(fake_output), fake_output)
        
        D_total_loss = D_fake_loss + D_real_loss
        return D_total_loss, D_fake_loss, D_real_loss
    
    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.batch_size, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
          generated_images = self.G(noise, training=True)

          real_output = self.D(images[1], training=True)
          fake_output = self.D(generated_images, training=True)

          G_loss = self.calc_G_loss(fake_output)
          D_total_loss, D_fake_loss, D_real_loss = self.calc_D_loss(real_output, fake_output)

        G_grads = gen_tape.gradient(G_loss, self.G.trainable_variables)
        D_grads = disc_tape.gradient(D_total_loss, self.D.trainable_variables)

        self.G_optimizer.apply_gradients(zip(G_grads, self.G.trainable_variables))
        self.D_optimizer.apply_gradients(zip(D_grads, self.D.trainable_variables))
        
        return G_loss, D_total_loss, D_fake_loss, D_real_loss

    @tf.function
    def batch_train(self, batch, current_step, disc_std):
        """Batch train the model.
        Args:
            batch (tensor): tensor with batches of real input and target.
            current_step (tensor): tensor with int value depicting current step.
            disc_std (tensor): tensor with float value depicting current std for disc noise.
        """
        G_loss, D_total_loss, D_fake_loss, D_real_loss = self.train_step(batch)

        with self.summary_writer.as_default():
            tf.summary.scalar("model/total_D_loss", D_total_loss, step=current_step)
            tf.summary.scalar("model/total_G_loss", G_loss, step=current_step)
            tf.summary.scalar("D/D_fake_loss", D_fake_loss, step=current_step)
            tf.summary.scalar("D/D_real_loss", D_real_loss, step=current_step)
            
            tf.summary.scalar(
                "model_info/G_lr", self.G_optimizer.learning_rate, step=current_step
            )
            tf.summary.scalar(
                "model_info/D_lr", self.D_optimizer.learning_rate, step=current_step
            )

    def load_models(self, models_directory, version=None):
        """Load all models.
        Args:
            models_directory (Path): Root directory of models.
            version (int): version number.
        """
        version = "_" + str(version) if version else ""
        self.D = load_model(models_directory / f"discriminator{version}.h5")
        self.G = load_model(models_directory / f"generator{version}.h5")

    def save_models(self, models_directory, version=None):
        """Save the model weights.
        Args:
            models_directory (Path): Root directory of models.
            version (int): version number. For these models it's the current step number.
        """
        models_directory.mkdir(parents=True, exist_ok=True)
        version = "_" + str(version) if version else ""

        self.D.save(models_directory / f"discriminator{version}.h5")
        self.G.save(models_directory / f"generator{version}.h5")

    def create_example(self, example, filename):
        """Creates and stores four examples based on a random input image.
        Args:
            example (tf batch): tensorflow batch.
            filename (str): File name.
        """
        for input_image, output_image in example:
            predictions = {}
            random_noise = tf.random.normal([1, self.noise_dim])
            for i in np.arange(4):
                predicted_image = self.G.predict(random_noise)
                predicted_image = (predicted_image[0] * 0.5) + 0.5
                predictions[i] = predicted_image

            fig, axes = plt.subplots(figsize=(15, 3 * 6), nrows=2, ncols=3,)

            input_results = [
                (random_noise.numpy() * 0.5) + 0.5,
                (output_image[0] * 0.5) + 0.5,
            ] + list(predictions.values())

            titles = ["Input Image", "Ground Truth"] + [
                f"Prediction_{pred_num}" for pred_num in list(predictions.keys())
            ]

            for title, img, ax in zip(titles, input_results, axes.flatten()):
                plt.subplot(ax)
                plt.title(title, fontweight="bold")
                plt.imshow(img)
                plt.axis("off")

            for ax in axes.flatten():
                plt.subplot(ax)
                plt.axis("off")

            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
