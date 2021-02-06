import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam

from utils import build_vae_encoder, build_decoder, vae_generate_images, plot_compare_vae, r_loss, total_loss


if __name__ == '__main__' :

	##1
	WEIGHTS_FOLDER = './weights/'
	DATA_FOLDER = './data/img_align_celeba/'


	##2
	filenames = np.array(glob(os.path.join(DATA_FOLDER, '*/*.jpg')))
	NUM_IMAGES = len(filenames)
	print("Total number of images : " + str(NUM_IMAGES))
	# prints : Total number of images : 202599


	LEARNING_RATE = 0.0005
	N_EPOCHS = 10
	LOSS_FACTOR = 10

	INPUT_DIM = (128,128,3) # Image dimension
	BATCH_SIZE = int(512) ## Original is 512
	Z_DIM = 512 # Dimension of the latent vector (z)

	data_flow = ImageDataGenerator(rescale=1./255).flow_from_directory(DATA_FOLDER, 
																																		 target_size = INPUT_DIM[:2],
																																		 batch_size = BATCH_SIZE,
																																		 shuffle = True,
																																		 class_mode = 'input',
																																		 subset = 'training'
																																		 )

	###3
	# ENCODER
	vae_encoder_input, vae_encoder_output,	mean_mu, log_var, vae_shape_before_flattening, vae_encoder	= build_vae_encoder(input_dim = INPUT_DIM,
																			output_dim = Z_DIM, 
																			conv_filters = [32, 64, 64, 64],
																			conv_kernel_size = [3,3,3,3],
																			conv_strides = [2,2,2,2],
																			loss_factor = LOSS_FACTOR)
	print(vae_encoder.summary())


	###4
	# Decoder
	vae_decoder_input, vae_decoder_output, vae_decoder = build_decoder(input_dim = Z_DIM,
																					shape_before_flattening = vae_shape_before_flattening,
																					conv_filters = [64,64,32,3],
																					conv_kernel_size = [3,3,3,3],
																					conv_strides = [2,2,2,2]
																					)
	print(vae_decoder.summary())


	###5
	# The input to the model will be the image fed to the encoder.
	vae_input = vae_encoder_input

	# Output will be the output of the decoder. The term - decoder(encoder_output) 
	# combines the model by passing the encoder output to the input of the decoder.
	vae_output = vae_decoder(vae_encoder_output)

	# Input to the combined model will be the input to the encoder.
	# Output of the combined model will be the output of the decoder.
	# vae_model = Model(vae_input, vae_output)
	vae_model = Model(vae_input, vae_decoder(vae_encoder(vae_input)), name="Variational_Auto_Encoder")

	# Compiling the model
	adam_optimizer = Adam(lr = LEARNING_RATE)
	vae_model.compile(optimizer=adam_optimizer, loss = total_loss, metrics = [r_loss])

	# Loading the model
	vae_model.load_weights(os.path.join(WEIGHTS_FOLDER, 'VAE/weights.h5'))


	###7
	example_batch = next(data_flow)
	example_batch = example_batch[0]
	example_images = example_batch[:10]
	plot_compare_vae(vae_model, data_flow, images = example_images)	


	###8
	vae_generate_images(vae_decoder, Z_DIM, n_to_show=10)	 


	###9
  z_test = vae_encoder.predict(example_batch[:200])

	x = np.linspace(-3, 3, 300)

	fig = plt.figure(figsize=(20, 20))
	fig.subplots_adjust(hspace=0.6, wspace=0.4)

	for i in range(50):
			ax = fig.add_subplot(5, 10, i+1)
			ax.hist(z_test[:,i], density=True, bins = 20)
			ax.axis('off')
			ax.text(0.5, -0.35, str(i), fontsize=10, ha='center', transform=ax.transAxes)
			ax.plot(x,norm.pdf(x))

	plt.show()