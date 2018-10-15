import numpy as np
import scipy.misc

MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 3))


def load_image(path, image_height, image_width):
	image_raw = scipy.misc.imread(path)
	# Resize the image for convnet input and add an extra dimension
	image_raw = scipy.misc.imresize(image_raw, (image_height, image_width))
	# Substract the image with mean value
	image = image_raw - MEAN_VALUES
	return image_raw, image


def recover_image(image):
	image_raw = image + MEAN_VALUES
	image_raw = np.clip(image_raw, 0, 255).astype('uint8')
	return image_raw


def save_image(path, image):
	# Output should add back the mean.
	image = recover_image(image)
	scipy.misc.imsave(path, image)


def generate_noise_image(content_image, noise_ratio, image_height, image_width, color_channels):
	"""
	Returns a noise image intermixed with the content image at a certain ratio.
	"""
	# Create a noise image which will be mixed with the content image
	noise_image = np.random.uniform(
		-20, 20,
		(image_height, image_width, color_channels)).astype('float32')
	# Take a weighted average of the values
	gen_image = noise_image * noise_ratio + content_image * (1.0 - noise_ratio)
	return gen_image
