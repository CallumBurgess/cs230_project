import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
from sklearn.model_selection import train_test_split
import os 
import sys
import numpy as np
from PIL import Image
tf.enable_eager_execution()

IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 2
N_CHANNELS = 1
N_CLASSES = 11
seed_train_validation = 1
shuffle_value = True
validation_split = 0.2

genre1, genre2 = "pop", "classical"
directory1 = "./spectrograms/" + genre1 
directory2 = "./spectrograms/" + genre2


genre1_paths = [os.path.join(directory1, filename) for filename in os.listdir(directory1)]
genre2_paths = [os.path.join(directory2, filename) for filename in os.listdir(directory2)]


dataset1 = tf.data.Dataset.from_tensor_slices(genre1_paths)
dataset2 = tf.data.Dataset.from_tensor_slices(genre2_paths)


def normalize(image_path):
  image = tf.io.read_file(image_path)
  image = tf.image.decode_image(image, channels=1)
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return tf.reshape(tf.image.grayscale_to_rgb(image), (IMG_HEIGHT, IMG_WIDTH, 3))

dataset1 = dataset1.cache().map(normalize).cache().shuffle(
    100).batch(2)
split = 4*len(genre1_paths)//5
train1, test1 = dataset1.take(split),dataset1.skip(split)

dataset2 = dataset2.cache().map(normalize).cache().shuffle(
    100).batch(2)
split = 4*len(genre2_paths)//5
train2, test2 = dataset2.take(split),dataset2.skip(split)

### Most of this code structure and logic is based on the cyclegan tf tutorial located at the following colab:  https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/cyclegan.ipynb#scrollTo=2M7LmLtGEMQJ


OUTPUT_CHANNELS = 3

generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)



loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real, generated):
  real_loss = loss_obj(tf.ones_like(real), real)
  generated_loss = loss_obj(tf.zeros_like(generated), generated)
  total_disc_loss = real_loss + generated_loss
  return total_disc_loss * 0.5

def generator_loss(generated):
  return loss_obj(tf.ones_like(generated), generated)

def calc_cycle_loss(real_image, cycled_image):
  loss = tf.reduce_mean(tf.abs(real_image - cycled_image))
  return 8 * loss

generator_g_optimizer = tf.keras.optimizers.Adam(1e-5, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(1e-5, beta_1=0.5)
discriminator_x_optimizer = tf.keras.optimizers.Adam(1e-5, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(1e-5, beta_1=0.5)


def train_step(real_x, real_y):
  with tf.GradientTape(persistent=True) as tape:
  	#generators 
    fake_y = generator_g(real_x, training=True)
    cycled_x = generator_f(fake_y, training=True)
    fake_x = generator_f(real_y, training=True)
    cycled_y = generator_g(fake_x, training=True)
    #discriminators
    disc_real_x = discriminator_x(real_x, training=True)
    disc_real_y = discriminator_y(real_y, training=True)
    disc_fake_x = discriminator_x(fake_x, training=True)
    disc_fake_y = discriminator_y(fake_y, training=True)
    #calciulate losses
    gen_g_loss = generator_loss(disc_fake_y)
    gen_f_loss = generator_loss(disc_fake_x)
    total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)
    total_gen_g_loss = gen_g_loss + total_cycle_loss 
    total_gen_f_loss = gen_f_loss + total_cycle_loss 

    disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
    disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)
  
  # Calculate the gradients for generator and discriminator
  generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                        generator_g.trainable_variables)
  generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                        generator_f.trainable_variables)
  
  discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                            discriminator_x.trainable_variables)
  discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                            discriminator_y.trainable_variables)
  
  # Apply the gradients to the optimizer
  generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                            generator_g.trainable_variables))

  generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                            generator_f.trainable_variables))
  
  discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                discriminator_x.trainable_variables))
  
  discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))

  return total_gen_g_loss, total_gen_f_loss, total_cycle_loss, disc_x_loss, disc_y_loss

EPOCHS = 200

test_image = genre1_paths[-3]
print(test_image)
def gen_image(epoch):
  input_image = normalize(test_image)
  test_input = input_image
  test_input = tf.expand_dims(test_input, axis=0)
  prediction = generator_g(test_input)
  img = (prediction[0]*0.5 + 0.5)*256
  img = tf.cast(img, tf.uint8)
  i = img.numpy()
  (Image.fromarray(i)).save("./cycle_generated/ " + str(epoch) + "generated.jpeg")

def train():
	for epoch in range(EPOCHS):
	  n = 0
	  total_gen_g_loss, total_gen_f_loss, total_cycle_loss, disc_x_loss, disc_y_loss= 0.0,0.0,0.0,0.0,0.0
	  for image_x, image_y in tf.data.Dataset.zip((train1, train2)):

	    total_gen_g_loss, total_gen_f_loss, total_cycle_loss, disc_x_loss, disc_y_loss = train_step(image_x, image_y)
	    if n % 10 == 0:
	    	print(n)
	    n += 1

	  print("Losses:", total_gen_g_loss, total_gen_f_loss, total_cycle_loss, disc_x_loss, disc_y_loss)
	  gen_image(epoch)
	  generator_g.save_weights("./gan_model/" + genre2 + "generator")
	generator_f.save_weights("./gan_model/" + genre1 + "generator")
	discriminator_x.save_weights("./gan_model/" + genre1 + "discriminator")
	discriminator_y.save_weights("./gan_model/" + genre2 + "discriminator")

train()

print("done")














