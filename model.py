import tensorflow as tf

CONTENT_LAYER = 'conv4_2'
STYLE_LAYERS = [
	('conv1_1', 0.5),
	('conv2_1', 0.5),
	('conv3_1', 0.5),
	('conv4_1', 0.5),
	('conv5_1', 0.5),
]


def content_loss_func(sess, model):
	"""
	Content loss function as defined in the paper.
	"""

	def _content_loss(current_feat, content_feat):
		"""
		Inputs:
		- current_feat: features of the current image, Tensor with shape [1, height, width, channels]
		- content_feat: features of the content image, Tensor with shape [1, height, width, channels]

		Returns:
		- scalar content loss
		"""
		loss = 0.5 * tf.reduce_sum(tf.pow(content_feat - current_feat, 2))
		return loss

	return _content_loss(sess.run(model[CONTENT_LAYER]), model[CONTENT_LAYER])


def style_loss_func(sess, model):
	"""
	Style loss function as defined in the paper.
	"""

	def _gram_matrix(feat):
		"""
		Compute the Gram matrix from features.

		Inputs:
		- feat: Tensor of shape (1, H, W, C) giving features for a single image.

		Returns:
		- gram: Tensor of shape (C, C) giving the (optionally normalized) Gram matrices for the input image.
		"""
		N = int(feat.shape[3])
		M = int(feat.shape[1] * feat.shape[2])
		Ft = tf.reshape(feat, (M, N))
		gram = tf.matmul(tf.transpose(Ft), Ft)
		return gram

	def _style_loss(current_feat, style_feat):
		"""
		Inputs:
		- current_feat: features of the current image, Tensor with shape [1, height, width, channels]
		- style_feat: features of the style image, Tensor with shape [1, height, width, channels]

		Returns:
		- scalar style loss
		"""
		assert (current_feat.shape == style_feat.shape)
		N = current_feat.shape[3]
		M = current_feat.shape[1] * current_feat.shape[2]
		c_gram = _gram_matrix(current_feat)
		s_gram = _gram_matrix(style_feat)
		loss = (1 / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(tf.pow(c_gram - s_gram, 2))
		return loss

	E = [_style_loss(sess.run(model[layer_name]), model[layer_name]) for layer_name, _ in STYLE_LAYERS]
	W = [w for _, w in STYLE_LAYERS]
	loss = sum([W[l] * E[l] for l in range(len(STYLE_LAYERS))])
	return loss
