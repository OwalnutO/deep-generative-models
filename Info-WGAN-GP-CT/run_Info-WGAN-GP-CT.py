"""
run_Info-WGAN-GP-CT.py

Reference:
X. Chen, Y. Duan, R. Houthooft, J. Schulman, I. Sutskever, and P. Abbeel.: Infogan: interpretable representation learning by information maximizing generative adversarial nets. In NIPS, 2016.
I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. C. Courville.: Improved training of Wasserstein GAN. In NIPS, 2017.
X. Wei, Z. Liu, L. Wang, and B. Gong.: Improving the Improved Training of Wasserstein GANs. International Conference on Learning Representations (ICLR), 2018.

LZhu, 2018-Jul
"""

# import general packages
import sys
import numpy as np
from math import gcd
def lcm(a, b): return abs(a * b) // gcd(a, b) if a and b else 0
import scipy.io as sio
import os
os.environ['KERAS_BACKEND'] = 'tensorflow' # set up tensorflow backend for keras
from os import path
import matplotlib
if os.name != 'nt' and os.environ.get('DISPLAY','') == '':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
import h5py
import random
from tqdm import tqdm
from IPython import display
from functools import partial
from itertools import compress

# import keras packages
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Reshape, Flatten, Activation, Lambda, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.convolutional import MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
from keras.utils import generic_utils
from keras.utils import plot_model  # for model visualization, need to install Graphviz (.msi for Windows), pydot (pip install), graphviz (pip install) and set PATH for Graphviz
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


# get intersect of data
def get_intersect(dep1, data1, dep2, data2):
    idx1in2 = np.in1d(dep1, dep2)  # boolean indices of dep1 whose elements are in dep2
    idx1in2 = np.arange(len(dep1))[idx1in2]  # numerical indices of dep1 whose elements are in dep2
    idx2in1 = np.array([np.argwhere(dep2 == dep1[i]) for i in idx1in2]).flatten()  # numerical indices of dep2 whose elements are in dep1
    dep1 = dep1[idx1in2]
    if isinstance(data1, list):
        data1 = [d for d in data1]
    else:
        data1 = [data1]
    for i in range(len(data1)):
        data1[i] = data1[i][idx1in2]
    dep2 = dep2[idx2in1]
    if isinstance(data2, list):
        data2 = [d for d in data2]
    else:
        data2 = [data2]
    for i in range(len(data2)):
        data2[i] = data2[i][idx2in1]
    return dep1, data1, dep2, data2


# interpolate azimuthal data
def interp_az(data, multiple):
    n_dep, n_az = data.shape
    xp = np.arange(0, 360, 360/n_az)
    x = np.arange(0, 360, 360/n_az/multiple)
    data_out = np.array([np.interp(x, xp, row, period=360) for row in data])
    return data_out


# find layers by name
def find_layers_by_name(model, name):
    layers = model.layers
    bool_mask = [name in l.name for l in layers]
    layers = list(compress(layers, bool_mask))
    return layers


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


# plot loss function
def plot_loss(losses, save=False, save_filename=None):
    plt.figure(figsize=(10, 10))
    g_loss = np.array(losses['g'])
    d_loss = np.array(losses['d'])
    grad_penalty = np.array(losses['grad'])
    ct_penalty = np.array(losses['ct'])
    info_penalty_supervised = np.array(losses['info_supervised'])
    info_penalty_unsupervised = np.array(losses['info_unsupervised'])
    plt.plot(g_loss, label='G loss')
    plt.plot(d_loss, label='D loss')
    plt.plot(grad_penalty, label='grad penalty')
    plt.plot(ct_penalty, label='ct penalty')
    plt.plot(info_penalty_supervised, label='info_supervised')
    plt.plot(info_penalty_unsupervised, label='info_unsupervised')
    plt.legend()
    if save:
        if save_filename is not None:
            plt.savefig(save_filename)
    else:
        plt.show()
    plt.clf()
    plt.close('all')


# generate random noise, categorical and continuous latent variables
def get_noise(dim_noise, dim_cat, dim_cont, batch_size=32):
    noise = np.random.normal(0, 1, size=(batch_size, dim_noise))
    label = np.random.randint(0, dim_cat, size=(batch_size, 1))
    label = np_utils.to_categorical(label, num_classes=dim_cat)
    cont = np.random.uniform(0, 2, size=(batch_size, dim_cont))
    return noise, label, cont


# plot generated images
def plot_gen(generator, dim, figsize=(10, 10), channel=0, save=False, save_filename=None, method='cat', label_val=0, cont_val=0, **kwargs):
    dim_noise = generator.layers[0].input_shape[1]
    dim_cat = generator.layers[1].input_shape[1]
    dim_cont = generator.layers[2].input_shape[1]
    plt.figure(figsize=figsize)
    n_image_row, n_image_col = dim
    if method is 'cat':
        for i in range(dim_cat):
            noise, _, _ = get_noise(dim_noise, dim_cat, dim_cont, batch_size=n_image_col)
            label = np.repeat(i, n_image_col).reshape(-1, 1)
            label = np_utils.to_categorical(label, num_classes=dim_cat)
            cont = np.repeat(cont_val * np.ones((1, dim_cont)), n_image_col, axis=0)
            image_gen = generator.predict([noise, label, cont])
            for j in range(n_image_col):
                plt.subplot(dim_cat, n_image_col, i*n_image_col+j+1)
                if isinstance(channel, int):
                    plt.imshow(image_gen[j, :, :, channel], **kwargs)
                elif channel is 'color':
                    plt.imshow(image_gen[j], **kwargs)
                plt.axis('off')
    elif method is 'cont':
        cont_range_row = np.linspace(0, 2, num=n_image_row)
        if dim_cont >= 2:
            cont_range_col = np.linspace(0, 2, num=n_image_col)
        for i in range(n_image_row):
            noise, _, _ = get_noise(dim_noise, dim_cat, dim_cont, batch_size=n_image_col)
            label = np.repeat(label_val, n_image_col).reshape(-1, 1)
            label = np_utils.to_categorical(label, num_classes=dim_cat)
            if dim_cont >= 2:  # vary the first two dimensions
                cont = np.concatenate([np.array([cont_range_row[i], cont_range_col[j]]).reshape(1, -1) for j in range(n_image_col)])
                cont = np.concatenate([cont, np.zeros((n_image_col, dim_cont-2))], axis=1)
            else:
                cont = np.repeat(cont_range_row[i], n_image_col).reshape(-1, 1)
            image_gen = generator.predict([noise, label, cont])
            for j in range(n_image_col):
                plt.subplot(n_image_row, n_image_col, i*n_image_col+j+1)
                if isinstance(channel, int):
                    plt.imshow(image_gen[j, :, :, channel], **kwargs)
                elif channel is 'color':
                    plt.imshow(image_gen[j], **kwargs)
                plt.axis('off')
    else:
        raise NotImplementedError
    if save:
        if save_filename is not None:
            plt.savefig(save_filename)
    else:
        plt.tight_layout()
        plt.show()
    plt.clf()
    plt.close('all')


def build_generator(n_class, n_cont, n_rows, n_cols, n_out_ch=1, n_first_conv_ch=256, dim_noise=100, leaky_relu_alpha=0):
    g_in_noise = Input(shape=(dim_noise,))
    g_in_cat = Input(shape=(n_class,))
    g_in_cont = Input(shape=(n_cont,))
    g_in = concatenate([g_in_noise, g_in_cat, g_in_cont])
    x = Dense((n_rows//4) * (n_cols//4) * n_first_conv_ch)(g_in)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    x = Reshape((n_rows // 4, n_cols // 4, n_first_conv_ch))(x)
    x = Conv2DTranspose(n_first_conv_ch//2, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    x = Conv2DTranspose(n_first_conv_ch//4, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    g_out = Conv2DTranspose(n_out_ch, (4, 4), strides=1, padding='same', activation='sigmoid')(x)  # output in [0, 1]
    generator = Model([g_in_noise, g_in_cat, g_in_cont], g_out, name='generator')
    print('Summary of Generator (for InfoWGAN-GP)')
    generator.summary()
    return generator


def build_disc_aux(n_class, n_cont, n_rows, n_cols, n_in_ch=1, n_last_conv_ch=256, leaky_relu_alpha=0.2, dropout_rate=0.25):
    d_in = Input(shape=(n_rows, n_cols, n_in_ch))
    x = Conv2D(n_last_conv_ch//4, (4, 4), strides=1, padding='same')(d_in)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    x = Dropout(rate=dropout_rate)(x)
    x = Conv2D(n_last_conv_ch//2, (4, 4), strides=2, padding='same')(x)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    x = Dropout(rate=dropout_rate)(x)
    x = Conv2D(n_last_conv_ch, (4, 4), strides=2, padding='same')(x)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    x = Dropout(rate=dropout_rate)(x)
    d_out_base = Flatten()(x)

    # discriminator value output (for Wasserstein loss)
    x = Dense(1024)(d_out_base)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    d_out_val = Dense(1)(x)  # no activation function

    # categorical output
    x = Dense(1024)(d_out_base)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    c_out_softmax = Dense(n_class, activation='softmax')(x)

    # continuous output
    x = Dense(1024)(d_out_base)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    f_out_mean = Dense(n_cont)(x)
    f_out_logstd = Dense(n_cont)(x)

    # discriminator model
    discriminator = Model(d_in, d_out_val, name='discriminator')
    print('Summary of Discriminator (for InfoWGAN-GP)')
    discriminator.summary()

    # classifier model
    classifier = Model(d_in, c_out_softmax, name='classifier')
    print('Summary of Classifier (for InfoWGAN-GP)')
    classifier.summary()

    # feature extractor model
    feature_extractor = Model(d_in, [f_out_mean, f_out_logstd], name='feature_extractor')
    print('Summary of Feature Extractor (for InfoWGAN-GP)')
    feature_extractor.summary()

    return discriminator, classifier, feature_extractor


def train_infowgangpct(image_set, label_set, generator, generator_opt, discriminator, discriminator_opt, classifier, feature_extractor, losses,
                       label_rate=1,
                       label_mode='cat',
                       lambda_gp=10,
                       lambda_ct=2,
                       ct_ubound=0,
                       lambda_info_sup=1,
                       lambda_info_unsup=0.5,
                       batch_size=32,
                       n_epochs=100,
                       train_dgratio=5,
                       save_every=10,
                       save_mode='multi_channel',
                       save_filename_prefix=None):
    # input for the real data
    real_in = Input(shape=image_set.shape[1:])
    # input for the labels
    label_in = Input(shape=label_set.shape[1:])
    # inputs for the generator
    gen_in_noise = Input(shape=generator.layers[0].input_shape[1:])
    gen_in_cat = Input(shape=generator.layers[1].input_shape[1:])
    gen_in_cont = Input(shape=generator.layers[2].input_shape[1:])
    # generated fake data
    gen_out = generator([gen_in_noise, gen_in_cat, gen_in_cont])
    # discriminator outputs
    disc_out_val_real = discriminator(real_in)
    disc_out_val_fake = discriminator(gen_out)
    # classifier outputs
    class_out_softmax_real = classifier(real_in)
    class_out_softmax_fake = classifier(gen_out)
    # feature extractor outputs
    feature_out_mean_real, feature_out_logstd_real = feature_extractor(real_in)
    feature_out_mean_fake, feature_out_logstd_fake = feature_extractor(gen_out)

    # gradient penalty
    eps_in = K.placeholder(shape=(None, 1, 1, 1))
    interp = eps_in * real_in + (1 - eps_in) * gen_out
    grad_interp = discriminator(interp)
    grad = K.gradients(grad_interp, [interp])[0]
    grad_norm = K.sqrt(K.sum(K.square(grad), axis=np.arange(1, len(grad.shape))))
    grad_penalty = K.mean(K.square(grad_norm - 1))

    # consistency term penalty
    # discriminator output for the 1st virtual data point close to the read data point
    disc_out_val_real_1 = discriminator.layers[0](real_in)
    for d_layer in discriminator.layers[1:]:
        if 'dropout' in d_layer.name:  # do not use the backend wrapper K.dropout as it always sets up a fixed random seed
            disc_out_val_real_1 = K.tf.nn.dropout(disc_out_val_real_1 * 1.,
                                                  keep_prob=1-d_layer.get_config()['rate'],
                                                  noise_shape=d_layer.get_config()['noise_shape'],
                                                  seed=None)
        else:
            disc_out_val_real_1 = d_layer(disc_out_val_real_1)
    # discriminator output for the 2nd virtual data point close to the read data point
    disc_out_val_real_2 = discriminator.layers[0](real_in)
    for d_layer in discriminator.layers[1:]:
        if 'dropout' in d_layer.name:  # do not use the backend wrapper K.dropout as it always sets up a fixed random seed
            disc_out_val_real_2 = K.tf.nn.dropout(disc_out_val_real_2 * 1.,
                                                  keep_prob=1-d_layer.get_config()['rate'],
                                                  noise_shape=d_layer.get_config()['noise_shape'],
                                                  seed=None)
        else:
            disc_out_val_real_2 = d_layer(disc_out_val_real_2)
    consistency_term = K.square(disc_out_val_real_1 - disc_out_val_real_2)
    ct_penalty = K.mean(K.maximum(consistency_term - ct_ubound, 0))

    # information maximization penalty
    if label_mode is 'cat':
        info_penalty_real = K.mean(-K.sum(K.log(class_out_softmax_real + K.epsilon()) * label_in, axis=1))
    elif label_mode is 'cont':
        norm_real = (label_in - feature_out_mean_real) / (K.exp(feature_out_logstd_real) + K.epsilon())
        info_penalty_real = K.mean(K.sum(feature_out_logstd_real + 0.5 * K.square(norm_real), axis=1))
    info_penalty_fake_cat = K.mean(-K.sum(K.log(class_out_softmax_fake + K.epsilon()) * gen_in_cat, axis=1))
    norm_fake_cont = (gen_in_cont - feature_out_mean_fake) / (K.exp(feature_out_logstd_fake) + K.epsilon())
    info_penalty_fake_cont = K.mean(K.sum(feature_out_logstd_fake + 0.5 * K.square(norm_fake_cont), axis=1))

    # optimization
    d_loss_real = K.mean(disc_out_val_real)
    d_loss_fake = K.mean(disc_out_val_fake)

    g_loss = -d_loss_fake
    if label_mode is 'cat':
        g_loss += (lambda_info_sup * info_penalty_fake_cat + lambda_info_unsup * info_penalty_fake_cont)
        g_trainable_weights = generator.trainable_weights + classifier.trainable_weights[-4:]
    elif label_mode is 'cont':
        g_loss += (lambda_info_sup * info_penalty_fake_cont + lambda_info_unsup * info_penalty_fake_cat)
        g_trainable_weights = generator.trainable_weights + feature_extractor.trainable_weights[-6:]
    g_train_updates = generator_opt.get_updates(g_trainable_weights, [], g_loss)
    g_train = K.function([gen_in_noise, gen_in_cat, gen_in_cont],
                         [d_loss_fake, info_penalty_fake_cat, info_penalty_fake_cont],
                         g_train_updates)

    d_loss = d_loss_fake - d_loss_real\
             + lambda_gp * grad_penalty\
             + lambda_ct * ct_penalty
    if label_mode is 'cat':
        d_loss += (lambda_info_sup * info_penalty_real + lambda_info_unsup * info_penalty_fake_cont)
        d_trainable_weights = discriminator.trainable_weights + feature_extractor.trainable_weights[-6:]
    elif label_mode is 'cont':
        d_loss += (lambda_info_sup * info_penalty_real + lambda_info_unsup * info_penalty_fake_cat)
        d_trainable_weights = discriminator.trainable_weights + classifier.trainable_weights[-4:]
    d_train_updates = discriminator_opt.get_updates(d_trainable_weights, [], d_loss)
    d_train = K.function([real_in, label_in, gen_in_noise, gen_in_cat, gen_in_cont, eps_in],
                         [d_loss_real, d_loss_fake, grad_penalty, ct_penalty, info_penalty_real, info_penalty_fake_cat, info_penalty_fake_cont],
                         d_train_updates)

    # training
    n_train = image_set.shape[0]
    dim_noise = generator.layers[0].input_shape[1]
    dim_cat = generator.layers[1].input_shape[1]
    dim_cont = generator.layers[2].input_shape[1]
    n_ch = discriminator.layers[0].input_shape[-1]
    for ie in range(n_epochs):
        print('epoch: %d' % (ie + 1))
        idx_randperm = np.random.permutation(n_train)
        n_batches = n_train // batch_size
        progbar = generic_utils.Progbar(n_batches*batch_size)
        for ib in range(n_batches):
            # real batch
            idx_batch = idx_randperm[range(ib * batch_size, ib * batch_size + batch_size)]
            image_real_batch = image_set[idx_batch]
            # semi-supervision
            toss = np.random.binomial(1, label_rate)
            if toss > 0:
                label_real_batch = label_set[idx_batch]
            else:
                _, label_real_batch, _ = get_noise(dim_noise, dim_cat, dim_cont, batch_size=batch_size)
            # fake batch
            noise_disc, label_disc, cont_disc = get_noise(dim_noise, dim_cat, dim_cont, batch_size=batch_size)
            # train discriminator, classifier and feature extractor
            eps = np.random.uniform(size=(batch_size, 1, 1, 1))
            d_loss_real_train_val,\
            d_loss_fake_train_val,\
            grad_penalty_train_val,\
            ct_penalty_train_val,\
            info_penalty_real_train_val,\
            info_penalty_fake_cat_train_val,\
            info_penalty_fake_cont_train_val = d_train([image_real_batch, label_real_batch, noise_disc, label_disc, cont_disc, eps])
            losses['d'].append(d_loss_fake_train_val - d_loss_real_train_val)
            losses['grad'].append(grad_penalty_train_val)
            losses['ct'].append(ct_penalty_train_val)
            losses['info_supervised'].append(info_penalty_real_train_val)
            if label_mode is 'cat':
                losses['info_unsupervised'].append(info_penalty_fake_cont_train_val)
            elif label_mode is 'cont':
                losses['info_unsupervised'].append(info_penalty_fake_cat_train_val)
            # train generator, classifier and feature extractor
            if ((ib + 1) % train_dgratio == 0):
                noise_gen, label_gen, cont_gen = get_noise(dim_noise, dim_cat, dim_cont, batch_size=batch_size)
                d_loss_fake_train_val,\
                info_penalty_fake_cat_train_val,\
                info_penalty_fake_cont_train_val = g_train([noise_gen, label_gen, cont_gen])
                losses['g'].extend([-d_loss_fake_train_val] * train_dgratio)
                if label_mode is 'cat':
                    losses['info_supervised'][-1] += info_penalty_fake_cat_train_val
                    losses['info_unsupervised'][-1] += info_penalty_fake_cont_train_val
                elif label_mode is 'cont':
                    losses['info_supervised'][-1] += info_penalty_fake_cont_train_val
                    losses['info_unsupervised'][-1] += info_penalty_fake_cat_train_val
                # update progress bar
                progbar.add(batch_size * train_dgratio, values=[('G loss', -d_loss_fake_train_val),
                                                                ('D loss', d_loss_fake_train_val - d_loss_real_train_val),
                                                                ('grad penalty', grad_penalty_train_val),
                                                                ('ct penalty', ct_penalty_train_val),
                                                                ('info (sup) penalty', losses['info_supervised'][-1]),
                                                                ('info (unsup) penalty', losses['info_unsupervised'][-1])])

        # plot interim results
        if ((ie + 1) % save_every == 0) or (ie == n_epochs - 1):
            # display generated images channel by channel
            if save_mode is 'multi_channel':
                for ic in range(n_ch):
                    save_filename_image_gen = '%s_cat_gen_ch%d_epoch%d.pdf' % (save_filename_prefix, ic, ie + 1)
                    plot_gen(generator, (dim_cat, 10), (15, 15), ic, True, save_filename_image_gen, method='cat', cont_val=1, cmap='gray')
                    save_filename_image_gen = '%s_cont_gen_ch%d_epoch%d.pdf' % (save_filename_prefix, ic, ie + 1)
                    plot_gen(generator, (10, 10), (15, 15), ic, True, save_filename_image_gen, method='cont', label_val=1, cmap='gray')
            elif save_mode is 'color':
                save_filename_image_gen = '%s_cat_gen_epoch%d.pdf' % (save_filename_prefix, ie + 1)
                plot_gen(generator, (dim_cat, 10), (15, 15), 'color', True, save_filename_image_gen, method='cat', cont_val=1)
                save_filename_image_gen = '%s_cont_gen_epoch%d.pdf' % (save_filename_prefix, ie + 1)
                plot_gen(generator, (10, 10), (15, 15), 'color', True, save_filename_image_gen, method='cont', label_val=1)

    # plot loss
    save_filename_loss = '%s_loss_epoch%d.pdf' % (save_filename_prefix, n_epochs)
    plot_loss(losses, True, save_filename_loss)



if __name__ == '__main__':

    losses = {'g': [], 'd': [], 'grad': [], 'ct': [], 'info_supervised': [], 'info_unsupervised': []}

    # sanity check on mnist
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_train_min, x_train_max = normalize_data(x_train, a=0, b=1, method='sample')
    x_test, x_test_min, x_test_max = normalize_data(x_test, a=0, b=1, method='sample')
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    n_row, n_col, n_ch = x_train.shape[1:]

    # training hyperparameters
    batch_size = 128
    n_epochs = 100
    n_save_every = 10
    g_lr = 1e-3
    g_beta1 = 0.5
    g_beta2 = 0.9
    d_lr = 2e-4
    d_beta1 = 0.5
    d_beta2 = 0.9
    lambda_gp = 10
    lambda_ct = 2
    train_dgratio = 5
    dim_noise = 50
    n_class = 10
    n_cont = 2
    label_rate = 1
    lambda_info_sup = 10
    lambda_info_unsup = 1

    # output folder
    output_path = './output/'
    if not path.exists(output_path):
        os.makedirs(output_path)
    save_mode = 'multi_channel'
    save_filename_prefix = '%sMNIST_infowgangpct_noise%d_cat%d' % (output_path, dim_noise, n_class)

    generator = build_generator(n_class, n_cont, n_row, n_col, n_out_ch=n_ch, n_first_conv_ch=256, dim_noise=dim_noise)
    generator_opt = Adam(g_lr, g_beta1, g_beta2)
    plot_model(generator, to_file='%s_generator_model.pdf' % save_filename_prefix, show_shapes=True)

    discriminator, classifier, feature_extractor = build_disc_aux(n_class, n_cont, n_row, n_col, n_in_ch=n_ch, n_last_conv_ch=256)
    discriminator_opt = Adam(d_lr, d_beta1, d_beta2)
    plot_model(discriminator, to_file='%s_discriminator_model.pdf' % save_filename_prefix, show_shapes=True)
    plot_model(classifier, to_file='%s_classifier_model.pdf' % save_filename_prefix, show_shapes=True)
    plot_model(feature_extractor, to_file='%s_feature_extractor_model.pdf' % save_filename_prefix, show_shapes=True)

    generator_weights_file = '%s_generator_weights_epoch%d.hdf' % (save_filename_prefix, n_epochs)
    discriminator_weights_file = '%s_discriminator_weights_epoch%d.hdf' % (save_filename_prefix, n_epochs)
    classifier_weights_file = '%s_classifier_weights_epoch%d.hdf' % (save_filename_prefix, n_epochs)
    feature_extractor_weights_file = '%s_feature_extractor_weights_epoch%d.hdf' % (save_filename_prefix, n_epochs)
    if path.isfile(generator_weights_file) \
            and path.isfile(discriminator_weights_file)\
            and path.isfile(classifier_weights_file)\
            and path.isfile(feature_extractor_weights_file):
        # load InfoWGAN-GP weights that already existed
        print('loading InfoWGAN-GP model weights')
        generator.load_weights(generator_weights_file)
        discriminator.load_weights(discriminator_weights_file)
        classifier.load_weights(classifier_weights_file)
        feature_extractor.load_weights(feature_extractor_weights_file)
    else:
        # train InfoWGAN-GP
        print('training InfoWGAN-GP model')
        y_train = np_utils.to_categorical(y_train, num_classes=n_class)
        train_infowgangpct(x_train, y_train, generator, generator_opt, discriminator, discriminator_opt, classifier, feature_extractor, losses,
                           label_rate=label_rate,
                           lambda_gp=lambda_gp,
                           lambda_ct=lambda_ct,
                           lambda_info_sup=lambda_info_sup,
                           lambda_info_unsup=lambda_info_unsup,
                           batch_size=batch_size,
                           n_epochs=n_epochs,
                           train_dgratio=train_dgratio,
                           save_every=n_save_every,
                           save_mode=save_mode,
                           save_filename_prefix=save_filename_prefix)
        generator.save_weights(generator_weights_file)
        discriminator.save_weights(discriminator_weights_file)
        classifier.save_weights(classifier_weights_file)
        feature_extractor.save_weights(feature_extractor_weights_file)

    # accuracy test
    y_test_softmax = classifier.predict(x_test)
    y_test_pred = np.argmax(y_test_softmax, axis=1)
    acc_pred, w_cluster = cluster_acc(y_test, y_test_pred)
    print('Classifier accuracy: %.4f' % acc_pred)
    print(w_cluster.astype(int))

    # generate new data and save the plot
    if save_mode is 'multi_channel':
        for ic in range(n_ch):
            save_filename_image_gen = '%s_cat_gen_ch%d_epoch%d.pdf' % (save_filename_prefix, ic, n_epochs)
            plot_gen(generator, (n_class, 10), (15, 15), ic, True, save_filename_image_gen, method='cat', cont_val=1, cmap='gray')
            save_filename_image_gen = '%s_cont_gen_ch%d_epoch%d.pdf' % (save_filename_prefix, ic, n_epochs)
            plot_gen(generator, (10, 10), (15, 15), ic, True, save_filename_image_gen, method='cont', label_val=1, cmap='gray')
    elif save_mode is 'color':
        save_filename_image_gen = '%s_cat_gen_epoch%d.pdf' % (save_filename_prefix, n_epochs)
        plot_gen(generator, (n_class, 10), (15, 15), 'color', True, save_filename_image_gen, method='cat', cont_val=1)
        save_filename_image_gen = '%s_cont_gen_epoch%d.pdf' % (save_filename_prefix, n_epochs)
        plot_gen(generator, (10, 10), (15, 15), 'color', True, save_filename_image_gen, method='cont', label_val=1)

    pass