"""
run_VAE.py

LZhu, 2017-Jun
"""

import sys
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
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from keras import losses
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape, Lambda, Layer
from keras.layers.convolutional import Conv1D, Conv2D, Conv2DTranspose
from keras.layers.convolutional import MaxPooling1D, MaxPooling2D, UpSampling1D, UpSampling2D
from keras.utils import plot_model
from keras.optimizers import *
from keras import backend as K
K.set_image_data_format('channels_last')
from scipy.optimize import linear_sum_assignment



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
    from sklearn.utils.linear_assignment_ import linear_assignment
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, w


# customized VAE KL divergence loss layer
class KLDivergenceLossLayer(Layer):
    def __init__(self, **kwargs):
        super(KLDivergenceLossLayer, self).__init__(**kwargs)
    def call(self, args):
        z_mean, z_log_var = args
        kl_loss = -0.5 * K.sum(1. + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=1)  # KL divergence loss
        self.add_loss(K.mean(kl_loss), args)
        return args


# build VAE network
def build_vae(n_row, n_col, n_chn, output_size, alpha=1, lr=1e-3):
    opt = Adam(lr=lr)
    # encoder
    vae_input = Input(shape=(n_row, n_col, n_chn))
    x = Conv2D(n_filters[0], (3, 3), padding='same', activation='relu')(vae_input)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(n_filters[1], (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    shapeBeforeFlatten = x._keras_shape[1:]
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    z_mean = Dense(output_size, activation='linear', name='z_mean')(x)  # mean of z
    z_log_var = Dense(output_size, activation='linear', name='z_log_var')(x)  # log variance of z
    z_mean, z_log_var = KLDivergenceLossLayer(name='kl_divergence_loss')([z_mean, z_log_var])  # add KL loss
    z = Lambda(sample_z, output_shape=(output_size,), name='z')([z_mean, z_log_var])  # reparametrization
    encoder = Model(vae_input, z_mean)

    # decoder
    decoder_hidden = Dense(1024, activation='relu')
    decoder_expand = Dense(np.prod(shapeBeforeFlatten), activation='relu')
    decoder_reshape = Reshape(shapeBeforeFlatten)
    decoder_conv_1 = Conv2D(n_filters[1], (3, 3), padding='same', activation='relu')
    decoder_upsample_1 = UpSampling2D((2, 2))
    decoder_conv_2 = Conv2D(n_filters[0], (3, 3), padding='same', activation='relu')
    decoder_upsample_2 = UpSampling2D((2, 2))
    decoder_conv_3 = Conv2D(n_chn, (3, 3), padding='same', activation='sigmoid')  # output in [0, 1]
    x = decoder_hidden(z)
    x = decoder_expand(x)
    x = decoder_reshape(x)
    x = decoder_conv_1(x)
    x = decoder_upsample_1(x)
    x = decoder_conv_2(x)
    x = decoder_upsample_2(x)
    decoded_output = decoder_conv_3(x)

    # VAE model
    vae = Model(vae_input, decoded_output)
    vae.compile(optimizer=opt, loss=neg_log_ll(n_row, n_col, n_chn, alpha=alpha))  # add negative log-likelihood of Gaussian distribution
    vae.summary()

    # generator
    gen_input = Input(shape=(output_size,))
    x = decoder_hidden(gen_input)
    x = decoder_expand(x)
    x = decoder_reshape(x)
    x = decoder_conv_1(x)
    x = decoder_upsample_1(x)
    x = decoder_conv_2(x)
    x = decoder_upsample_2(x)
    gen_output = decoder_conv_3(x)
    generator = Model(gen_input, gen_output)

    return vae, encoder, generator



if __name__ == '__main__':

    # sanity check on MNIST
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_train_min, x_train_max = normalize_data(x_train, a=0, b=1, method='sample')
    x_test, x_test_min, x_test_max = normalize_data(x_test, a=0, b=1, method='sample')
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    n_row, n_col = 28, 28
    n_filters = (128, 256)
    nc = 10
    nz = 8
    batch_size = 128
    n_epochs = 20

    # build the VAE and its encoder and generator
    vae, encoder, generator = build_vae(n_row, n_col, 1, nz, alpha=4, lr=1e-3)

    # model visualization
    output_path = './output/'
    if not path.exists(output_path):
        os.makedirs(output_path)
    vae_model_fig_file = '%sMNIST_vae_model.pdf' % output_path
    plot_model(vae, to_file=vae_model_fig_file, show_shapes=True)

    # train VAE
    vae_weights_file = '%sMNIST_vae_weights.hdf' % output_path
    if path.isfile(vae_weights_file):  # load VAE model weights if weight file exists
        print('loading VAE model weights')
        vae.load_weights(vae_weights_file)
    else:  # train VAE and save weights
        print('training VAE model')
        vae.fit(x_train, x_train, batch_size=batch_size, epochs=n_epochs, verbose=1, validation_data=(x_test, x_test))
        vae.save_weights(vae_weights_file)
    x_test_rec = vae.predict(x_test)
    x_test_enc = encoder.predict(x_test)
    x_train_enc = encoder.predict(x_train)

    x_test_enc_clu = KMeans(n_clusters=nc).fit(x_test_enc)
    y_test_kmeans = x_test_enc_clu.labels_

    x_train_enc_clu = KMeans(n_clusters=nc).fit(x_train_enc)
    y_train_kmeans = x_train_enc_clu.labels_

    acc_kmeans, w_means = cluster_acc(y_test_kmeans, y_test)
    print('Accuracy of kmeans of latent variables results: %.4f' % acc_kmeans)

    # visualize the encoded data
    n_data = 5000
    idx_data = np.random.permutation(x_train_enc.shape[0])[:n_data]
    print('visualization on training')

    vae_encoder_output_fig_file = '%sMNIST_vae_encoder_train_tsne_output.pdf' % output_path
    x_train_enc_tsne = TSNE(n_components=2, init='pca').fit_transform(x_train_enc[idx_data])
    fig = plt.figure(1, figsize=(10, 10))
    plt.scatter(x_train_enc_tsne[:, 0], x_train_enc_tsne[:, 1], cmap=plt.cm.brg, c=y_train_kmeans[idx_data])
    plt.colorbar()
    # plt.show()
    fig.savefig(vae_encoder_output_fig_file)
    fig.clf()

    pass