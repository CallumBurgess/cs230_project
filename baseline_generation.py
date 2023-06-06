import tensorflow as tf
from PIL import Image
import os 
import random
import sys
from IPython.display import Audio


def calc_loss(img, genre, content, model):
  # convert image into same form as training
  img = (img + 1)*128
  img_batch = tf.expand_dims(img, axis=0)
  # get activations
  [content_activations, genre_activations] = model(img_batch)
  content_dist = tf.reduce_mean(tf.square(tf.subtract(content_activations, content)))
  genre_dist = tf.reduce_mean(tf.square(tf.subtract(genre_activations, genre)))
  # return scaled mse loss
  return content_dist + genre_dist

class DeepDreamAudio(tf.Module):
  def __init__(self, model):
    self.model = model

  @tf.function(
      input_signature=(
        tf.TensorSpec(shape=[None,None], dtype=tf.float32), #img
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32), # genre activation
        tf.TensorSpec(shape=[None,None,None, None], dtype=tf.float32), #style activation
        tf.TensorSpec(shape=[], dtype=tf.int32), #steps
        tf.TensorSpec(shape=[], dtype=tf.float32),)#step size
  )
  def __call__(self, img, genre, content, steps, step_size):
      loss = tf.constant(0.0)
      # run backprop to train image over specified loss
      for n in tf.range(1,steps):
        with tf.GradientTape() as tape:
          tape.watch(img)
          loss = calc_loss(img, genre, content, self.model)
          #print(tape.watched_variables())

        gradients = tape.gradient(loss, img)
        gradients /= tf.math.reduce_std(gradients) + 1e-8 
        #scale lr over time
        lr = step_size/tf.sqrt(tf.cast(n, tf.float32))
        img = img - gradients * tf.cast(lr,tf.float32)

      return loss, img


deepdreamaudio = DeepDreamAudio(spec_model)

# set up training of image 
def run_deep_dream_simple(img, genre, content, steps=100, step_size=0.005):
  img = tf.keras.applications.inception_v3.preprocess_input(img)
  img = tf.convert_to_tensor(img)
  step_size = tf.convert_to_tensor(step_size)
  steps_remaining = steps
  step = 0
  while steps_remaining:
    if steps_remaining>100:
      run_steps = tf.constant(100)
    else:
      run_steps = tf.constant(steps_remaining)
    steps_remaining -= run_steps
    step += run_steps

    loss, img = deepdreamaudio(img, tf.constant(genre), tf.constant(content), run_steps, tf.constant(step_size))
    print ("Step {}, loss {}".format(step, loss))
  return img



model = tf.keras.models.load_model('./classification_model')
layer_names = []
for l in model.layers:
  layer_names.append(l.name)
print(layer_names)

content_layer = layer_names[2]
genre_layer = layer_names[-2]
layers = [model.get_layer(content_layer).output, model.get_layer(genre_layer).output]
spec_model = tf.keras.Model(inputs=model.input, outputs=layers)

source_image_path, genre = sys.argv[0], sys.argv[1]
folder_path = "/spectrograms/" + str(genre)
genre_image_path = random.choice(os.listdir(folder_path))

image_s = Image.open(source_image_path)
image = np.array(image_s)  
content = spec_model(tf.expand_dims(image_s, axis=0))[0]
content = tf.convert_to_tensor(content, dtype=tf.float32)

print("before content image_s: ")
before_image = Image.fromarray(image_s, mode="L")
before_image.show()

image_g = Image.open(genre_image_path)
image = np.array(image_g)  
content = spec_model(tf.expand_dims(image_g, axis=0))[1]
content = tf.convert_to_tensor(content, dtype=tf.float32)

print("before content image_g: ")
before_image = Image.fromarray(image_g, mode="L")
before_image.show()

print("random init: ")
rand_init = np.random.randint(0, 255, (IMG_HEIGHT, IMG_WIDTH))
rand_init = np.asarray(rand_init,dtype=np.uint8)
rand_image = Image.fromarray(rand_init, mode="L")
rand_image.show()

print("image after: ")
converted = run_deep_dream_simple(rand_init, genre, content,steps=3000)
converted = ((converted.numpy() + 1)*128).astype('int')
converted = np.asarray(converted,dtype=np.uint8)
converted_image = Image.fromarray(converted, mode="L")
converted_image.save("generated_spectro.jpg")
converted_image.show()

difference = np.mean(converted - image_s)
print("difference: ", difference)








