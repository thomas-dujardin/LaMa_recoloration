
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, default='lama/bubble_1024.png')
parser.add_argument('--mask_path', type=str, default='lama/bubble_1024_mask.png')
args = parser.parse_args()
image_path = args.image_path
mask_path = args.mask_path


# Import libraries
import tensorflow.compat.v1 as tf
import numpy as np
import time
from functools import reduce
from skimage import io as skio
import skimage
import skimage.transform
from tqdm import tqdm 
import utils
from skimage import io as skio
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm 
import cv2
import numpy as np
from transformers import ViTFeatureExtractor, ViTModel

# Load the pre-trained ViT model and feature extractor
model = ViTModel.from_pretrained('google/vit-base-patch16-224')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# Define the function to patch the image
def get_masked_patches(mask, tile_size, color = 255):
    """
    Given a binary mask and the size of the patches, returns a list of tuples 
    containing the four coordinates of all patches in the masked zone.
    
    Args:
    - mask: A binary mask as a 2D NumPy array.
    - tile_size: The size of the patches (width and height) in pixels.
    
    Returns:
    - A list of tuples containing the four coordinates (top-left, top-right, 
      bottom-right, bottom-left) of all patches in the masked zone.
    """
    # Find the masked pixels
    masked_pixels = np.where(mask == color)
    
    # Create a list to store the patch coordinates
    patch_coords = []
    
    # Iterate over all the masked pixels
    for i in range(len(masked_pixels[0])):
        row, col = masked_pixels[0][i], masked_pixels[1][i]
        
        # Check if the current pixel is the top-left corner of a patch
        if (row % tile_size == 0) and (col % tile_size == 0):
            # Check if the patch is completely inside the mask
            if (row + tile_size < mask.shape[0]) and (col + tile_size < mask.shape[1]):
                # Append the four coordinates of the patch to the list
                patch_coords.append(((row, col), (row, col + tile_size), 
                                     (row + tile_size, col + tile_size), 
                                     (row + tile_size, col)))
    
    return patch_coords



## Part 1: Load gatys model
#Useful functions for gatys texture synthsis
TEXTURE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
LEARNING_RATE = .02
NORM_TERM = 2.
TEXTURE_WEIGHT = 3.
NORM_WEIGHT = 0.1
OUT_PATH = 'sortie_gatys.png'


def run_gatys(texture_array, content_array, EPOCHS=1500):
  """Run gatys texture synthesis"""
  sess=tf.Session()

  texture, image_shape = texture_array, texture_array.shape
  image_shape = [1] + image_shape
  texture = texture.reshape(image_shape).astype(np.float32)
  with tf.name_scope('vgg_texture'):
      texture_model = utils.Vgg19()
      texture_model.build(texture, image_shape[1:])

  grams={}
  # calcul des matrices de gram de l'image d'origine
  for l in TEXTURE_LAYERS:

      #tableau=sess.run(getattr(texture_model,l)) #gettattr renvoie texture_model.l
      tableau=(getattr(texture_model,l)).numpy()
      tableau=tableau.reshape(tableau.shape[1:])
      shape=tableau.shape

      tableau=tableau.reshape((shape[0]*shape[1],-1))

      grams[l]=np.matmul (tableau.T,tableau)/ (shape[0]*shape[1])

      

  sess.close()

  #Actual optimisation process to produce an image that mimicks the target texture
  with tf.Session() as sess:
      sample_size=[1,512,512,3]
      #content_image_init = tf.truncated_normal(sample_size, mean=.5, stddev=.1)
      #content_image = tf.Variable(content_image_init, dtype=tf.float32)

      # load image and resize
      img = content_array / 255.0
      img_resized = tf.image.resize(img, [512, 512])

      # reshape to match content_image tensor shape
      img_reshaped = tf.reshape(img_resized, [1, 512, 512, 3])
      content_image = tf.Variable(img_reshaped, dtype=tf.float32)


    
      x_model = utils.Vgg19()
      x_model.build(content_image, sample_size[1:])

      # Loss functions
      with tf.name_scope('loss'):
          # Texture
          if TEXTURE_WEIGHT is 0:
              texture_loss = tf.constant(0.)
          else:
              unweighted_texture_loss = utils.get_texture_loss(x_model, grams)#texture_model)
              texture_loss = unweighted_texture_loss * TEXTURE_WEIGHT

          # Norm regularization
          if NORM_WEIGHT is 0:
              norm_loss = tf.constant(0.)
          else:
              norm_loss = (utils.get_l2_norm_loss(content_image) ** NORM_TERM) * NORM_WEIGHT

          # Total loss
          total_loss = texture_loss + norm_loss  
      # Update image
      with tf.name_scope('update_image'):
          optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
          grads = optimizer.compute_gradients(total_loss, [content_image])
          clipped_grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads]
          update_image = optimizer.apply_gradients(clipped_grads)

      # Train
    
      sess.run(tf.global_variables_initializer())
      start_time = time.time()
      for i in tqdm(range(EPOCHS)):
          _, loss = sess.run([update_image, total_loss])
      print('etape numero',i,' parmi ',EPOCHS,'loss', loss)

      # FIN
      elapsed = time.time() - start_time
      print("Training complete. The session took %.2f seconds to complete." % elapsed)
      print("Rendering final image and closing TensorFlow session..")

      sess.close()

      return content_image
  


# Part 2: Inpainting with gatys 

# Load the input image and its corresponding binary mask
image = cv2.imread(image_path)
mask = cv2.imread(mask_path, 0)  # load as grayscale

# Define the patch size
tile_size = 100
masked_coords = get_masked_patches(mask, tile_size)
unmasked_coords = get_masked_patches(mask, tile_size, color=0)
masked_patches = [image[m_coord[0][0]:m_coord[2][0], m_coord[0][1]:m_coord[2][1]] for m_coord in masked_coords]
unmasked_patches = [image[u_coord[0][0]:u_coord[2][0], u_coord[0][1]:u_coord[2][1]] for u_coord in unmasked_coords]

# Extract features for the masked patches
masked_features = []
for patch in masked_patches:
    inputs = feature_extractor(patch, return_tensors='pt')
    features = model(**inputs).last_hidden_state.squeeze().detach().numpy()
    masked_features.append(features)



# Extract features for the unmasked patches
unmasked_features = []
for patch in unmasked_patches:
    inputs = feature_extractor(patch, return_tensors='pt')
    features = model(**inputs).last_hidden_state.squeeze().detach().numpy()
    unmasked_features.append(features)

# Replace the masked patches with the best-matching unmasked patches based on feature distances
for m_feat, m_coord in tqdm(zip(masked_features, masked_coords)):
    best_metric = float('inf')
    best_patch = None
    
    for u_feat, u_patch in zip(unmasked_features, unmasked_patches):
        metric = np.linalg.norm(m_feat - u_feat)
        if metric < best_metric:
            best_metric = metric
            best_patch = u_patch
    
    # Replace the masked patch with the best unmasked patch in the input image
    image[m_coord[0][0]:m_coord[2][0], m_coord[0][1]:m_coord[2][1]] = run_gatys(best_patch, image[m_coord[0][0]:m_coord[2][0], m_coord[0][1]:m_coord[2][1]], EPOCHS=1500)

# save array image as png
skio.imsave('output.png', image)