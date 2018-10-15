import os
import sys
import numpy as np
import tensorflow as tf
from model import content_loss_func, style_loss_func
from basenet import load_vgg_model
from utils import load_image, generate_noise_image, save_image

flags = tf.app.flags
flags.DEFINE_integer('image_height', 480, 'image height')
flags.DEFINE_integer('image_width', 640, 'image width')
flags.DEFINE_integer('color_channels', 3, 'channel number of color')
flags.DEFINE_integer('iteration', 500, 'number of iteration')
flags.DEFINE_string('vgg_model_path', 'imagenet-vgg-verydeep-19.mat', 'vgg model path')
flags.DEFINE_string('content_image', 'images/trojan_shrine.jpg', 'content image path')
flags.DEFINE_string('style_image', 'images/muse.jpg', 'style image path')
flags.DEFINE_string('output', 'output', 'output_path')
FLAGS = flags.FLAGS


def train_one_step(sess, model, image_height, image_width):
	# Construct content_loss using content_image
	_, content_image = load_image(FLAGS.content_image, image_height, image_width)
	content_image_list = np.reshape(content_image, ((1,) + content_image.shape))
	sess.run(model['input'].assign(content_image_list))
	content_loss = content_loss_func(sess, model)

	# Construct style_loss using style_image
	_, style_image = load_image(FLAGS.style_image, image_height, image_width)
	style_image_list = np.reshape(style_image, ((1,) + style_image.shape))
	sess.run(model['input'].assign(style_image_list))
	style_loss = style_loss_func(sess, model)

	# Constant to put more emphasis on content loss
	ALPHA = 0.0025
	# Constant to put more emphasis on style loss
	BETA = 1
	# Total loss
	total_loss = ALPHA * content_loss + BETA * style_loss

	# We minimize the total_loss
	optimizer = tf.train.AdamOptimizer(2.0)
	train_step = optimizer.minimize(total_loss)
	return train_step, total_loss


def train():
	config = tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	_, content_image = load_image(
		FLAGS.content_image,
		FLAGS.image_height,
		FLAGS.image_width
	)

	gen_image = generate_noise_image(
		content_image,
		0.75,
		FLAGS.image_height,
		FLAGS.image_width,
		FLAGS.color_channels
	)
	input_image_list = np.reshape(gen_image, ((1,) + gen_image.shape))

	model = load_vgg_model(
		FLAGS.vgg_model_path,
		FLAGS.image_height,
		FLAGS.image_width,
		FLAGS.color_channels
	)
	sess.run(model['input'].assign(input_image_list))
	train_step, total_loss = train_one_step(sess, model, FLAGS.image_height, FLAGS.image_width)
	sess.run(tf.global_variables_initializer())
	for iter in range(FLAGS.iteration):
		sess.run(train_step)
		if iter % 50 == 0:
			mixed_image = sess.run(model['input'])
			print('Iteration {}: loss = {:e}'.format(iter, sess.run(total_loss)))

			if not os.path.exists(FLAGS.output):
				os.mkdir(FLAGS.output)

			filename = 'output/{}.png'.format(iter)
			save_image(filename, mixed_image[0])


def main(_):
	train()


if __name__ == '__main__':
	tf.app.run()
