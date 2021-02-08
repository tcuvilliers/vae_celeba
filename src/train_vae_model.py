import os
from glob import glob

import numpy as np

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from utils import build_vae_encoder, build_decoder, r_loss, total_loss


if __name__ == '__main__' :

	param_list = [[0.0005,0.00044,512],\
				[0.0005,0.00044,1024],\
				[0.0005,0.00044,768],\
				[0.0001,0.00044,768],\
				[0.0001,0.001,768],\
				[0.0005,0.001,768],\
				[0.0005,0.000044,768],\
				[0.00005,0.00044,768],\
				]

	h5_list = ["VAE/weights_5e-4_44e-5_512Z_Ep10.h5",\
			"VAE/weights_5e-4_44e-5_1024Z_Ep10.h5",\
			"VAE/weights_5e-4_44e-5_768Z_Ep10.h5",\
			"VAE/weights_1e-4_44e-5_768Z_Ep10.h5",\
			"VAE/weights_1e-4_1e-3_768Z_Ep10.h5",\
			"VAE/weights_5e-4_1e-3_768Z_Ep10.h5",\
			"VAE/weights_5e-4_44e-6_768Z_Ep10.h5",\
			"VAE/weights_5e-5_44e-5_768Z_Ep10.h5",\
			]


	for i in range(len(param_list)):

		### Hardcoded parameters
		N_EPOCHS = 10
		INPUT_DIM = (224,224,3) # Image dimension
		BATCH_SIZE = int(128) ## Original is 512


		LOSS_FACTOR = param_list[i][0]
		LEARNING_RATE = param_list[i][1]
		Z_DIM = param_list[i][2] # Dimension of the latent vector (z)

		H5_FILE = h5_list[i]



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

		# SaveFile
		checkpoint_vae = ModelCheckpoint(os.path.join(WEIGHTS_FOLDER, H5_FILE), save_weights_only = True, verbose=1)

		# Fitting the model (auto-saves the weights at every epoch)
		vae_model.fit(data_flow,
								shuffle=True,
								epochs = N_EPOCHS,
								initial_epoch = 0,
								steps_per_epoch=NUM_IMAGES / BATCH_SIZE,
								callbacks=[checkpoint_vae])
