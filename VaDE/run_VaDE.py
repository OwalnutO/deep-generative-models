"""
run_VaDE.py

Reference:
Z. Jiang, Y. Zheng, H. Tan, B. Tang, and H. Zhou, "Variational deep embedding: A generative approach to clustering", In International Joint Conference on Artificial Intelligence, 2017.

LZhu, 2018-Jun
"""

import sys
import math
import numpy as np
import scipy.io as sio
import os
os.environ['KERAS_BACKEND'] = 'tensorflow' # set up tensorflow backend for keras
from os import path
import matplotlib
if os.name != 'nt' and os.environ.get('DISPLAY','') == '':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.interactive(False)
import h5py
from sklearn import mixture
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from keras import losses
from keras import initializers
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape, Activation, Lambda, Layer
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.convolutional import MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import plot_model
from keras.optimizers import *
from keras import backend as K
K.set_image_data_format('channels_last')
from sklearn.utils.linear_assignment_ import linear_assignment



# remove duplicate data
def remove_duplicates(dep, data):
    _, idxUnq = np.unique(dep, return_index=True)  # sorted array
    dep = dep[idxUnq]
    if isinstance(data, list):
        data = [d for d in data]
    else:
        data = [data]
    for i in range(len(data)):
        data[i] = data[i][idxUnq]
    return dep, data


# calculate negative log-likelihood of Gaussian distribution
def neg_log_ll(n_row, n_col, n_chn, alpha=1):
    def neg_log_ll_loss(y_true, y_pred):
        loss = losses.binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
        loss *= n_row * n_col * n_chn
        loss *= alpha
        return loss
    return neg_log_ll_loss


# sampling function on the latent variable (z) domain
def sample_z(args):
    z_mean, z_log_var = args
    batch_size = K.shape(z_mean)[0]
    nz = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch_size, nz), mean=0.0, stddev=1.0)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# normalize data to [a, b]
def normalize_data(data, a=0, b=1, method='sample'):
    if method is 'sample':  # normalize for each sample
        min_data = data.min(axis=tuple(np.arange(1, data.ndim)), keepdims=True)
        max_data = data.max(axis=tuple(np.arange(1, data.ndim)), keepdims=True)
    elif method is 'all':  # normalize across all samples
        min_data = data.min()
        max_data = data.max()
    else:
        raise NotImplementedError
    data_norm = ((b - a) * (data - min_data) / (max_data - min_data)) + a
    return data_norm, min_data, max_data


# denormalize data
def denormalize_data(data, min_data, max_data, a=0, b=1):
    data_denorm = ((data - a) * (max_data - min_data) / (b - a)) + min_data
    return data_denorm


def cluster_acc(y_true, y_pred):
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, w


# customized VaDE KL divergence loss layer
class KLDivergenceLossLayer(Layer):
    def __init__(self, n_centroids, **kwargs):
        self.n_centroids = n_centroids
        super(KLDivergenceLossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.latent_dim = input_shape[0][1]
        # prior probability of each GMM component
        self.pi = self.add_weight(name='pi', shape=(self.n_centroids,), initializer=initializers.Constant(1.0/self.n_centroids), trainable=True)
        # mean of each GMM component
        self.mu = self.add_weight(name='mu', shape=(self.latent_dim, self.n_centroids), initializer=initializers.RandomUniform(-0.05, 0.05), trainable=True)
        # variance of each GMM component
        self.sigma = self.add_weight(name='sigma', shape=(self.latent_dim, self.n_centroids), initializer=initializers.RandomUniform(0.95, 1.05), trainable=True)
        super(KLDivergenceLossLayer, self).build(input_shape)

    def call(self, args):
        z_mean, z_log_var = args

        # reparametrization
        batch_size = K.shape(z_mean)[0]
        nz = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch_size, nz), mean=0.0, stddev=1.0)
        z = z_mean + K.exp(0.5 * z_log_var) * epsilon

        # clip the values
        # self.pi = K.clip(self.pi, K.epsilon(), 1)
        # self.sigma = K.clip(self.sigma, K.epsilon(), 100)

        z_nc = K.permute_dimensions(K.repeat(z, self.n_centroids), [0, 2, 1])
        z_mean_nc = K.permute_dimensions(K.repeat(z_mean, self.n_centroids), [0, 2, 1])
        z_log_var_nc = K.permute_dimensions(K.repeat(z_log_var, self.n_centroids), [0, 2, 1])

        p_c_z = K.exp(K.sum((K.log(self.pi) - 0.5 * K.log(2 * math.pi * self.sigma) - K.square(z_nc - self.mu) / (2 * self.sigma)), axis=1)) + K.epsilon()
        q_c_given_x = p_c_z / K.sum(p_c_z, axis=-1, keepdims=True)

        # ELBO components except the multivariate Gaussian distribution term
        log_p_z_given_c = -0.5 * K.sum(q_c_given_x * (self.latent_dim * K.log(math.pi * 2) + K.sum(K.log(self.sigma), axis=0) + K.sum(K.exp(z_log_var_nc) / self.sigma, axis=1) + K.sum(K.square(z_mean_nc - self.mu) / self.sigma, axis=1)), axis=1)
        log_p_c = K.sum(K.log(self.pi) * q_c_given_x, axis=1)
        log_q_z_given_x = -0.5 * (self.latent_dim * K.log(math.pi * 2) + K.sum(z_log_var + 1, axis=1))
        log_q_c_given_x = K.sum(K.log(q_c_given_x) * q_c_given_x, axis=1)
        neg_elbo_loss_without_nll = K.mean(-log_p_z_given_c - log_p_c + log_q_z_given_x + log_q_c_given_x)
        self.add_loss(neg_elbo_loss_without_nll, args)

        return [z, z_mean, z_log_var, q_c_given_x]

    def compute_output_shape(self, input_shape):
        z_mean_shape, z_log_var_shape = input_shape
        z_shape = z_mean_shape
        q_c_given_x_shape = (None, self.n_centroids)
        return [z_shape, z_mean_shape, z_log_var_shape, q_c_given_x_shape]


# build VaDE network
def build_vade(n_centroids, n_row, n_col, n_chn, output_size, alpha=1, lr=1e-3, leaky_relu_alpha=0.2):
    opt = Adam(lr=lr)
    # encoder
    vade_input = Input(shape=(n_row, n_col, n_chn))
    x = Conv2D(n_filters[0], (3, 3), strides=2, padding='same')(vade_input)
    # x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    x = Conv2D(n_filters[1], (3, 3), strides=2, padding='same')(x)
    # x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    shape_before_flatten = x._keras_shape[1:]
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    z_mean = Dense(output_size, activation='linear', name='z_mean')(x)  # mean of z
    z_log_var = Dense(output_size, activation='linear', name='z_log_var')(x)  # log variance of z
    encoder = Model(vade_input, z_mean)
    z, z_mean, z_log_var, q_c_given_x = KLDivergenceLossLayer(n_centroids=n_centroids, name='kl_divergence_loss')([z_mean, z_log_var])  # reparametrization and custom layer to add the negative ELBO loss

    # classifier
    classifier = Model(vade_input, q_c_given_x)

    # define decoder/generator layers
    decoder_hidden = Dense(1024, activation='relu')
    decoder_expand = Dense(np.prod(shape_before_flatten), activation='relu')
    decoder_reshape = Reshape(shape_before_flatten)
    decoder_deconv_1 = Conv2DTranspose(n_filters[1], (3, 3), strides=2, padding='same')
    decoder_bn_1 = BatchNormalization(axis=-1)
    decoder_actv_1 = Activation('relu')
    decoder_deconv_2 = Conv2DTranspose(n_filters[0], (3, 3), strides=2, padding='same')
    decoder_bn_2 = BatchNormalization(axis=-1)
    decoder_actv_2 = Activation('relu')
    decoder_deconv_3 = Conv2DTranspose(n_chn, (3, 3), strides=1, padding='same', activation='sigmoid')  # output in [0, 1]

    # decoder
    x = decoder_hidden(z)
    x = decoder_expand(x)
    x = decoder_reshape(x)
    x = decoder_deconv_1(x)
    x = decoder_bn_1(x)
    x = decoder_actv_1(x)
    x = decoder_deconv_2(x)
    x = decoder_bn_2(x)
    x = decoder_actv_2(x)
    decoded_output = decoder_deconv_3(x)

    # VaDE model
    vade = Model(vade_input, decoded_output)
    vade.compile(optimizer=opt, loss=neg_log_ll(n_row, n_col, n_chn, alpha=alpha))
    vade.summary()

    # generator
    gen_input = Input(shape=(output_size,))
    x = decoder_hidden(gen_input)
    x = decoder_expand(x)
    x = decoder_reshape(x)
    x = decoder_deconv_1(x)
    x = decoder_bn_1(x)
    x = decoder_actv_1(x)
    x = decoder_deconv_2(x)
    x = decoder_bn_2(x)
    x = decoder_actv_2(x)
    gen_output = decoder_deconv_3(x)
    generator = Model(gen_input, gen_output)

    return vade, encoder, classifier, generator



if __name__ == '__main__':

    # sanity check on mnist
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_train_min, x_train_max = normalize_data(x_train, a=0, b=1, method='sample')
    x_test, x_test_min, x_test_max = normalize_data(x_test, a=0, b=1, method='sample')
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    n_row, n_col, n_ch = x_train.shape[1:]
    n_filters = (128, 256)
    n_class = 10
    nz = 8
    batch_size = 128
    n_epochs = 20
    is_pretrain = True
    n_pretrain_epochs = 5

    vade, encoder, classifier, generator = build_vade(n_class, n_row, n_col, n_ch, nz, alpha=4, lr=1e-3)

    # output folder
    output_path = './output/'
    if not path.exists(output_path):
        os.makedirs(output_path)
    save_filename_prefix = '%sMNIST_vade_zdim%d_class%d' % (output_path, nz, n_class)

    # model visualization
    vade_model_fig_file = '%s_model.pdf' % save_filename_prefix
    plot_model(vade, to_file=vade_model_fig_file, show_shapes=True)

    # train VaDE
    vade_weight_file = '%s_weights.hdf' % save_filename_prefix
    if path.isfile(vade_weight_file):  # load VaDE model weights if weight file exists
        print('loading VaDE model weights')
        vade.load_weights(vade_weight_file)
    else:  # train VaDE and save weights
        print('training VaDE model')
        # pretrain the underlying AutoEncoder model
        if is_pretrain:
            print('pretrain the underlying AutoEncoder model')
            encoder_input = Input(shape=(n_row, n_col, n_ch))
            encoder_output = encoder(encoder_input)
            decoder_output = generator(encoder_output)
            ae = Model(encoder_input, decoder_output)
            ae.compile(optimizer='adam', loss='binary_crossentropy')
            ae.fit(x_train, x_train, batch_size=batch_size, epochs=n_pretrain_epochs, verbose=1,
                   validation_data=(x_test, x_test))
            x_train_enc_pretrain = encoder.predict(x_train)
            # estimate the GMM parameters from the encoded latent variables of the data
            gmm_model = mixture.GaussianMixture(n_components=n_class, covariance_type='diag')
            gmm_model.fit(x_train_enc_pretrain)
            # initialize the GMM parameters
            vade.get_layer('kl_divergence_loss').set_weights([gmm_model.weights_, gmm_model.means_.T, gmm_model.covariances_.T])
        vade.fit(x_train, x_train, batch_size=batch_size, epochs=n_epochs, verbose=1, validation_data=(x_test, x_test))
        vade.save_weights(vade_weight_file)

    # encoding and classification
    x_test_rec = vade.predict(x_test)
    x_test_enc = encoder.predict(x_test)
    y_test_class_softmax = classifier.predict(x_test)
    y_test_class_argmax = np.argmax(y_test_class_softmax, axis=1)

    x_train_enc = encoder.predict(x_train)
    y_train_class_softmax = classifier.predict(x_train)
    y_train_class_argmax = np.argmax(y_train_class_softmax, axis=1)

    # clustering
    x_test_enc_clu = KMeans(n_clusters=n_class).fit(x_test_enc)
    y_test_clu = x_test_enc_clu.labels_

    x_train_enc_clu = KMeans(n_clusters=n_class).fit(x_train_enc)
    y_train_clu = x_train_enc_clu.labels_

    # classification and clustering accuracy
    acc_class_argmax, w_class_argmax = cluster_acc(y_test_class_argmax, y_test)
    acc_clu, w_clu = cluster_acc(y_test_clu, y_test)
    print('accuracy of argmax of classifier results: %.4f' % acc_class_argmax)
    print(w_class_argmax.astype(int))
    print('accuracy of clustering of latent variables results: %.4f' % acc_clu)
    print(w_clu.astype(int))

    # generate new data and save the plot
    gmm_weights, gmm_means, gmm_covariances = vade.get_layer('kl_divergence_loss').get_weights()
    n_image_cols = 10
    plt.figure(figsize=(15, 15))
    save_filename_image_gen = '%s_gen_epoch%d.pdf' % (save_filename_prefix, n_epochs)
    for i in range(n_class):
        z_samples = np.random.multivariate_normal(gmm_means[:, i], np.diag(gmm_covariances[:, i]), size=n_image_cols)
        image_gen = generator.predict(z_samples)
        for j in range(n_image_cols):
            plt.subplot(n_class, n_image_cols, i * n_image_cols + j + 1)
            if n_ch == 1:
                plt.imshow(image_gen[j, :, :, 0], cmap='gray')
            elif n_ch == 3:
                plt.imshow(image_gen[j])
            plt.axis('off')
    plt.savefig(save_filename_image_gen)
    plt.clf()
    plt.close('all')

    # visualize the encoded data
    n_data = 5000
    idx_data = np.random.permutation(x_train_enc.shape[0])[:n_data]
    print('visualization on training')

    vade_encoder_output_fig_file = '%s_tSNE_encoder_output.pdf' % save_filename_prefix
    x_train_enc_tsne = TSNE(n_components=2, init='pca').fit_transform(x_train_enc[idx_data])
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(x_train_enc_tsne[:, 0], x_train_enc_tsne[:, 1], cmap=plt.cm.brg, c=y_train_clu[idx_data])
    plt.colorbar()
    # plt.show()
    fig.savefig(vade_encoder_output_fig_file)
    fig.clf()


    pass