import os
from glob import glob

import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam

from utils import build_vae_encoder, build_decoder, r_loss, total_loss


if __name__ == '__main__' :

	### Hardcoded parameters
	LEARNING_RATE = 0.0005
	N_EPOCHS = 10
	LOSS_FACTOR = 0.0001

	INPUT_DIM = (128,128,3) # Image dimension
	BATCH_SIZE = int(256) ## Original is 512
	Z_DIM = 512 # Dimension of the latent vector (z)



	##1
	WEIGHTS_FOLDER = './weights/'
	DATA_FOLDER = './data/img_align_celeba/'


	##2
	filenames = np.array(glob(os.path.join(DATA_FOLDER, '*/*.jpg')))
	NUM_IMAGES = len(filenames)
	print("Total number of images : " + str(NUM_IMAGES))
	# prints : Total number of images : 202599

	data_flow = ImageDataGenerator(rescale=1./255).flow_from_directory(DATA_FOLDER, 
																	   target_size = INPUT_DIM[:2],
																	   batch_size = BATCH_SIZE,
																	   shuffle = True,
																	   class_mode = 'input',
																	   subset = 'training'
																	   )


	###3
	# ENCODER
	vae_encoder_input, vae_encoder_output,  mean_mu, log_var, vae_shape_before_flattening, vae_encoder  = build_vae_encoder(input_dim = INPUT_DIM,
										output_dim = Z_DIM, 
										conv_filters = [32, 64, 64, 64],
										conv_kernel_size = [3,3,3,3],
										conv_strides = [2,2,2,2],
										use_dropout = True,
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
	 
	# Compiling the model with the optimizer Adam
	adam_optimizer = Adam(lr = LEARNING_RATE)
	tf.config.experimental_run_functions_eagerly(True)
	vae_model.compile(optimizer=adam_optimizer, loss = total_loss, metrics = [r_loss])#, experimental_run_tf_function=False)

	# Not sure what this does ?
	checkpoint_vae = ModelCheckpoint(os.path.join(WEIGHTS_FOLDER, 'VAE/weights.h5'), save_weights_only = True, verbose=1)

	# Fitting the model (auto-saves the weights at every epoch)
	vae_model.fit(data_flow,
							shuffle=True,
							epochs = N_EPOCHS,
							initial_epoch = 0,
							steps_per_epoch=NUM_IMAGES / BATCH_SIZE,
							callbacks=[checkpoint_vae])
