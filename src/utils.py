import numpy as np

import tensorflow as tf
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Activation, BatchNormalization, LeakyReLU, Dropout, Layer
from keras.models import Model
from keras import backend as K

import matplotlib.pyplot as plt

# A layer that creates an activity regularization loss
class SamplingLayer(Layer):
		def __init__(self, loss_factor = 1):
				super(SamplingLayer, self).__init__()
				self.loss_factor = loss_factor

		def call(self, inputs):
				mean_mu, log_var = inputs

				# The KL-Loss we are adding here, gets in the total_loss of the whole VAE model.
				kl_loss = -0.5 * (1 + log_var - tf.square(mean_mu) - tf.exp(log_var))
				self.add_loss(self.loss_factor * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)))

				epsilon = K.random_normal(shape=K.shape(mean_mu), mean=0., stddev=1.)

				return mean_mu + K.exp(log_var/2)*epsilon


###3
# ENCODER
def build_vae_encoder(input_dim, output_dim, conv_filters, conv_kernel_size, 
									conv_strides, use_batch_norm = False, use_dropout = False,
									loss_factor = 1):
	
	# Clear tensorflow session to reset layer index numbers to 0 for LeakyRelu, 
	# BatchNormalization and Dropout.
	# Otherwise, the names of above mentioned layers in the model 
	# would be inconsistent
	global K
	K.clear_session()
	
	# Number of Conv layers
	n_layers = len(conv_filters)

	# Define model input
	encoder_input = Input(shape = input_dim, name = 'encoder_input')
	x = encoder_input

	# Add convolutional layers
	for i in range(n_layers):
			x = Conv2D(filters = conv_filters[i], 
									kernel_size = conv_kernel_size[i],
									strides = conv_strides[i], 
									padding = 'same',
									name = 'encoder_conv_' + str(i)
									)(x)
			if use_batch_norm:
				x = BathcNormalization()(x)
	
			x = LeakyReLU()(x)

			if use_dropout:
				x = Dropout(rate=0.25)(x)

	# Required for reshaping latent vector while building Decoder
	shape_before_flattening = K.int_shape(x)[1:] 
	
	x = Flatten()(x)
	
	mean_mu = Dense(output_dim, name = 'mu')(x)
	log_var = Dense(output_dim, name = 'log_var')(x)


	### Legacy
	### This has been replaced by the custom Layer SamplingLayer
	# # Defining a function for sampling
	# def sampling(args):
	#	 mean_mu, log_var = args
	#	 epsilon = K.random_normal(shape=K.shape(mean_mu), mean=0., stddev=1.) 
	#	 return mean_mu + K.exp(log_var/2)*epsilon	 
	
	# # Using a Keras Lambda Layer to include the sampling function as a layer 
	# # in the model
	# encoder_output = Lambda(sampling, name='encoder_output')([mean_mu, log_var])

	encoder_output = SamplingLayer(loss_factor=loss_factor)([mean_mu, log_var])

	return encoder_input, encoder_output, mean_mu, log_var, shape_before_flattening, Model(encoder_input, encoder_output, name="VAE_Encoder")


###4
# Decoder
def build_decoder(input_dim, shape_before_flattening, conv_filters, conv_kernel_size, 
									conv_strides):

	# Number of Conv layers
	n_layers = len(conv_filters)

	# Define model input
	decoder_input = Input(shape = (input_dim,) , name = 'decoder_input')

	# To get an exact mirror image of the encoder
	x = Dense(np.prod(shape_before_flattening))(decoder_input)
	x = Reshape(shape_before_flattening)(x)

	# Add convolutional layers
	for i in range(n_layers):
			x = Conv2DTranspose(filters = conv_filters[i], 
									kernel_size = conv_kernel_size[i],
									strides = conv_strides[i], 
									padding = 'same',
									name = 'decoder_conv_' + str(i)
									)(x)
			
			# Adding a sigmoid layer at the end to restrict the outputs 
			# between 0 and 1
			if i < n_layers - 1:
				x = LeakyReLU()(x)
			else:
				x = Activation('sigmoid')(x)

	# Define model output
	decoder_output = x

	return decoder_input, decoder_output, Model(decoder_input, decoder_output, name="VAE_Decoder")


###6
def r_loss(y_true, y_pred):
		return K.mean(K.square(y_true - y_pred), axis = [1,2,3])

def total_loss(y_true, y_pred):
		return r_loss(y_true, y_pred)


###7
def plot_compare_vae(vae_model, data_flow, images=None):
	
	if images is None:
		example_batch = next(data_flow)
		example_batch = example_batch[0]
		images = example_batch[:10]

	n_to_show = images.shape[0]
	reconst_images = vae_model.predict(images)

	fig = plt.figure(figsize=(15, 3))
	fig.subplots_adjust(hspace=0.4, wspace=0.4)

	for i in range(n_to_show):
			img = images[i].squeeze()
			sub = fig.add_subplot(2, n_to_show, i+1)
			sub.axis('off')				
			sub.imshow(img)

	for i in range(n_to_show):
			img = reconst_images[i].squeeze()
			sub = fig.add_subplot(2, n_to_show, i+n_to_show+1)
			sub.axis('off')
			sub.imshow(img)	


###8
def vae_generate_images(vae_decoder, z_dim, n_to_show=10):
	reconst_images = vae_decoder.predict(np.random.normal(0,1,size=(n_to_show,z_dim)))

	fig = plt.figure(figsize=(15, 3))
	fig.subplots_adjust(hspace=0.4, wspace=0.4)

	for i in range(n_to_show):
				img = reconst_images[i].squeeze()
				sub = fig.add_subplot(2, n_to_show, i+1)
				sub.axis('off')				
				sub.imshow(img)


###Legacy
# def kl_loss(y_true, y_pred):
#		 global mean_mu
#		 global log_var
#		 kl_loss =	-0.5 * K.sum(1 + log_var - K.square(mean_mu) - K.exp(log_var), axis = 1)
#		 # print(kl_loss)
#		 return kl_loss

# def kl_loss_new(y_true, y_pred):
#		 global mean_mu
#		 global log_var
#		 kl_loss = -0.5 * (1 + log_var - tf.square(mean_mu) - tf.exp(log_var))
#		 kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
#		 return kl_loss
