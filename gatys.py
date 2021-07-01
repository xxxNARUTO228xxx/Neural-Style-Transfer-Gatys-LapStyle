import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)

    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x


class Gatys:
    def gram_matrix(self, input_tensor):
        channels = int(input_tensor.shape[-1])
        a = tf.reshape(input_tensor, [-1, channels])
        n = tf.shape(a)[0]
        gram = tf.matmul(a, a, transpose_a=True)
        return gram / tf.cast(n, tf.float32)


    def compute_loss(self, init_image):
        style_weight, content_weight = self.loss_weights

        model_outputs = self.model(init_image)

        style_output_features = model_outputs[:self.num_style_layers]
        content_output_features = model_outputs[self.num_style_layers:]

        # compute style loss for each style layer
        style_score = 0
        weight_per_style_layer = 1.0 / float(self.num_style_layers)
        for target_style, comb_style in zip(self.gram_style_features, style_output_features):
            for mtx in comb_style:
                gram_style = self.gram_matrix(mtx)
                loss_value = tf.reduce_sum(tf.square(gram_style - target_style))
                style_score += weight_per_style_layer * loss_value/tf.cast((4*tf.square(gram_style.get_shape()[0]+gram_style.get_shape()[1])), tf.float32)
            style_score /= (len(comb_style)*len(comb_style))

        # compute content loss
        content_score = tf.reduce_sum(tf.square(content_output_features[0] - self.content_features))/2

        # total value of loss function
        loss = style_weight * style_score + content_weight * content_score
        return loss, style_score, content_score


    def transfer(self, img, img_style, num_iterations=100, content_weight=1, style_weight=1):
        # Preprocessing
        x_img = keras.applications.vgg16.preprocess_input(np.expand_dims(img, axis=0))
        x_style = keras.applications.vgg16.preprocess_input(np.expand_dims(img_style, axis=0))

        vgg = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')
        vgg.trainable = False

        # content layer
        content_layer = 'block5_conv3'

        # style layers
        style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1',
                        'block4_conv1'
                       ]

        self.num_style_layers = len(style_layers)

        # list of outputs
        style_outputs = [vgg.get_layer(name).output for name in style_layers]
        content_output = [vgg.get_layer(content_layer).output]
        model_outputs = style_outputs + content_output

        # define our non-trainable model (we change our image, not model)
        self.model = keras.models.Model(vgg.input, model_outputs)
        for layer in self.model.layers:
            layer.trainable = False

        # get style and content features
        style_outputs = self.model(x_style)
        content_output = self.model(x_img)
        self.style_features = [style_layer[0] for style_layer in style_outputs[:self.num_style_layers]]
        self.content_features = content_output[self.num_style_layers:][0]

        # compute gram matrices for each style layer
        self.gram_style_features = [self.gram_matrix(style_feature) for style_feature in self.style_features]

        init_image = np.copy(x_img)
        init_image = tf.Variable(init_image, dtype=tf.float32)

        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=2, beta1=0.99, epsilon=1e-1)
        best_loss, best_img = float('inf'), None
        self.loss_weights = (style_weight, content_weight)

        for i in range(num_iterations):
            with tf.GradientTape() as tape:
               all_loss = self.compute_loss(init_image)

            loss, style_score, content_score = all_loss
            grads = tape.gradient(loss, init_image)
            opt.apply_gradients([(grads, init_image)])

            if loss < best_loss:
                # Update best loss and best image from total loss.
                best_loss = loss
                best_img = deprocess_img(init_image.numpy())

            print('Iteration: {}'.format(i), loss)

        print(best_loss)
        return best_img
