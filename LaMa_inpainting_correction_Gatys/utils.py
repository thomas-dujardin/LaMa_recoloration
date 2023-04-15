# Import libraries
import tensorflow.compat.v1 as tf
import numpy as np
from functools import reduce
from skimage import io as skio
import skimage
import skimage.transform


# Calcul de la matrice de Gram d'un bloc convolutif
def convert_to_gram(filter_maps):
    # Get the dimensions of the filter maps to reshape them into two dimenions
    dimension = filter_maps.get_shape().as_list()
    reshaped_maps = tf.reshape(filter_maps, [dimension[1] * dimension[2], dimension[3]])

    # on normalise par la taille spatiale pour obtenir une valeur comparable entre images 
    return tf.matmul(reshaped_maps, reshaped_maps, transpose_a=True)/((dimension[1] * dimension[2]))


# Compute the L2-norm divided by squared number of dimensions
def get_l2_norm_loss(diffs):
    shape = diffs.get_shape().as_list()
    size = reduce(lambda x, y: x * y, shape) ** 2
    sum_of_squared_diffs = tf.reduce_sum(tf.square(diffs))
    return sum_of_squared_diffs / size


# Calcul de la loss texture etant donne le resaeau x et les matrices de gram s
def get_texture_loss(x, s):
    with tf.name_scope('get_style_loss'):
        texture_layer_losses = [get_texture_loss_for_layer(x, s, l) for l in TEXTURE_LAYERS]
        texture_weights = tf.constant([1. / len(texture_layer_losses)] * len(texture_layer_losses), tf.float32)
        weighted_layer_losses = tf.multiply(texture_weights, tf.convert_to_tensor(texture_layer_losses))
        return tf.reduce_sum(weighted_layer_losses)


# La loss texture pour une couche particuliere l
def get_texture_loss_for_layer(x, s, l):
    with tf.name_scope('get_style_loss_for_layer'):

        x_layer_maps = getattr(x, l)

        x_layer_gram = convert_to_gram(x_layer_maps)
        t_layer_gram = s[l] 

        shape = x_layer_maps.get_shape().as_list()
        size = shape[-1]**2 
        gram_loss = get_l2_norm_loss(x_layer_gram - t_layer_gram)
        return gram_loss / size



def load_image(path):
  
    img = skio.imread(path) / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()

    # Crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    shape = list(img.shape)

    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (shape[0], shape[1]))
    return resized_img, shape

def render_img(session, x, save=False, out_path=None):
    shape = x.get_shape().as_list()
    img = np.clip(session.run(x), 0, 1.0)
    print(img.shape)
    skio.imsave(out_path,img[0])


#VGG19 class definition
VGG_MEAN = [103.939, 116.779, 123.68]
data = None
weights_file='./vgg19.npy'
TEXTURE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

class Vgg19:
    def __init__(self, vgg19_npy_path=weights_file):
        global data


        if data is None:
            data = np.load(vgg19_npy_path, encoding='latin1',allow_pickle=True)
            self.data_dict = data.item()
            print("VGG19 weights loaded")

        else:
            self.data_dict = data.item()

    def build(self, rgb, shape):
        rgb_scaled = rgb * 255.0
        num_channels = shape[2]
        channel_shape = shape
        channel_shape[2] = 1

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)

        assert red.get_shape().as_list()[1:] == channel_shape
        assert green.get_shape().as_list()[1:] == channel_shape
        assert blue.get_shape().as_list()[1:] == channel_shape

        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])

        shape[2] = num_channels
        assert bgr.get_shape().as_list()[1:] == shape

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.avg_pool(self.conv1_2, 'pool1',self.filtre_moyenneur(64))

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.avg_pool(self.conv2_2, 'pool2',self.filtre_moyenneur(128))

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4")
        self.pool3 = self.avg_pool(self.conv3_4, 'pool3',self.filtre_moyenneur(256))

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4")
        self.pool4 = self.avg_pool(self.conv4_4, 'pool4',self.filtre_moyenneur(512))

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")

        self.data_dict = None

    def filtre_moyenneur(self,nbf):
        filtre=np.zeros((2,2,nbf,nbf),np.float32)
        for k in range(nbf):
            filtre[:,:,k,k]=0.25
        return filtre
    def avg_pool(self, bottom, name,filtre_inutile):
        return tf.nn.avg_pool(bottom,
            ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
#    def avg_pool(self, bottom, name,filtre):
#        return tf.nn.conv2d(bottom,tf.constant(filtre),strides=[1,2,2,1],padding='SAME',name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom,
            ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")

#END HYPERPARAMETERS

# Calcul de la matrice de Gram d'un bloc convolutif
def convert_to_gram(filter_maps):
    # Get the dimensions of the filter maps to reshape them into two dimenions
    dimension = filter_maps.get_shape().as_list()
    reshaped_maps = tf.reshape(filter_maps, [dimension[1] * dimension[2], dimension[3]])

    # on normalise par la taille spatiale pour obtenir une valeur comparable entre images 
    return tf.matmul(reshaped_maps, reshaped_maps, transpose_a=True)/((dimension[1] * dimension[2]))


# Compute the L2-norm divided by squared number of dimensions
def get_l2_norm_loss(diffs):
    shape = diffs.get_shape().as_list()
    size = reduce(lambda x, y: x * y, shape) ** 2
    sum_of_squared_diffs = tf.reduce_sum(tf.square(diffs))
    return sum_of_squared_diffs / size


# Calcul de la loss texture etant donne le resaeau x et les matrices de gram s
def get_texture_loss(x, s):
    with tf.name_scope('get_style_loss'):
        texture_layer_losses = [get_texture_loss_for_layer(x, s, l) for l in TEXTURE_LAYERS]
        texture_weights = tf.constant([1. / len(texture_layer_losses)] * len(texture_layer_losses), tf.float32)
        weighted_layer_losses = tf.multiply(texture_weights, tf.convert_to_tensor(texture_layer_losses))
        return tf.reduce_sum(weighted_layer_losses)


# La loss texture pour une couche particuliere l
def get_texture_loss_for_layer(x, s, l):
    with tf.name_scope('get_style_loss_for_layer'):

        x_layer_maps = getattr(x, l)

        x_layer_gram = convert_to_gram(x_layer_maps)
        t_layer_gram = s[l] 

        shape = x_layer_maps.get_shape().as_list()
        size = shape[-1]**2 
        gram_loss = get_l2_norm_loss(x_layer_gram - t_layer_gram)
        return gram_loss / size

