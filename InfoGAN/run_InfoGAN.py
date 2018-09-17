"""
run_InfoGAN.py

Reference:
X. Chen, Y. Duan, R. Houthooft, J. Schulman, I. Sutskever, and P. Abbeel. Infogan:  interpretable representation learning by information maximizing generative adversarial nets. In NIPS, 2016.

LZhu, 2018-May
"""

# import general packages
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

# import keras packages
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Reshape, Flatten, Activation, Lambda, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.convolutional import MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import add, concatenate
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


# set network weights trainability
def set_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


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
    c_loss = np.array(losses['c'])
    plt.plot(g_loss, label='G loss')
    plt.plot(d_loss, label='D loss')
    plt.plot(c_loss, label='C loss')
    plt.legend()
    if save:
        if save_filename is not None:
            plt.savefig(save_filename)
    else:
        plt.show()
    plt.clf()
    plt.close('all')


# generate random noise and categorical latent variables
def get_noise(dim_noise, dim_cat, batch_size=32):
    noise = np.random.uniform(-1, 1, size=(batch_size, dim_noise))
    label = np.random.randint(0, dim_cat, size=(batch_size, 1))
    label = np_utils.to_categorical(label, num_classes=dim_cat)
    return noise, label


# plot generated images
def plot_gen(generator, dim, figsize=(10, 10), channel=0, save=False, save_filename=None, **kwargs):
    dim_noise = generator.layers[0].input_shape[1]
    dim_cat = generator.layers[1].input_shape[1]
    n_image_col = dim[1]
    plt.figure(figsize=figsize)
    for i in range(dim_cat):
        noise, _ = get_noise(dim_noise, dim_cat, batch_size=n_image_col)
        label = np.repeat(i, n_image_col).reshape(-1, 1)
        label = np_utils.to_categorical(label, num_classes=dim_cat)
        image_gen = generator.predict([noise, label])
        for j in range(n_image_col):
            plt.subplot(dim_cat, n_image_col, i*n_image_col+j+1)
            if isinstance(channel, int):
                plt.imshow(image_gen[j, :, :, channel], **kwargs)
            elif channel is 'color':
                plt.imshow(image_gen[j], **kwargs)
            plt.axis('off')
    if save:
        if save_filename is not None:
            plt.savefig(save_filename)
    else:
        plt.tight_layout()
        plt.show()
    plt.clf()
    plt.close('all')


def build_generator(n_class, n_rows, n_cols, n_out_ch=1, n_first_conv_ch=256, dim_noise=100):
    g_in_noise = Input(shape=(dim_noise,))
    g_in_cat = Input(shape=(n_class,))
    g_in = concatenate([g_in_noise, g_in_cat])
    x = Dense((n_rows//4) * (n_cols//4) * n_first_conv_ch)(g_in)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = Reshape((n_rows // 4, n_cols // 4, n_first_conv_ch))(x)
    x = Conv2DTranspose(n_first_conv_ch//2, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(n_first_conv_ch//4, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    g_out = Conv2DTranspose(n_out_ch, (4, 4), strides=1, padding='same', activation='sigmoid')(x)  # output in [0, 1]
    generator = Model([g_in_noise, g_in_cat], g_out, name='generator')
    print('Summary of Generator (for InfoGAN)')
    generator.summary()
    return generator


def build_disc_class(n_class, n_rows, n_cols, n_in_ch=1, n_last_conv_ch=256, leaky_relu_alpha=0.2, lr=1e-4, beta_1=0.5, beta_2=0.9):
    d_opt = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2)
    d_in = Input(shape=(n_rows, n_cols, n_in_ch))
    x = Conv2D(n_last_conv_ch//4, (4, 4), strides=1, padding='same')(d_in)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    x = Conv2D(n_last_conv_ch//2, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    x = Conv2D(n_last_conv_ch, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=leaky_relu_alpha, name='d_feature')(x)
    d_out_base = Flatten()(x)

    # real probability output
    d_out_sigmoid = Dense(1, activation='sigmoid')(d_out_base)

    # categorical output
    x = Dense(1024)(d_out_base)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    c_out_softmax = Dense(n_class, activation='softmax')(x)

    discriminator_classifier = Model(d_in, [d_out_sigmoid, c_out_softmax], name='discriminator_classifier')
    discriminator_classifier.compile(optimizer=d_opt,
                                     loss=['binary_crossentropy', 'categorical_crossentropy'],
                                     loss_weights=[1, 1])
    print('Summary of Discriminator and Classifier (for InfoGAN)')
    discriminator_classifier.summary()

    return discriminator_classifier


def build_infogan(generator, discriminator_classifier, lr=1e-4, beta_1=0.5, beta_2=0.9):
    infogan_opt = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2)
    infogan_in_noise = Input(shape=generator.layers[0].input_shape[1:])
    infogan_in_cat = Input(shape=generator.layers[1].input_shape[1:])
    gen_out = generator([infogan_in_noise, infogan_in_cat])
    set_trainable(discriminator_classifier, False)
    infogan_out_sigmoid, infogan_out_softmax = discriminator_classifier(gen_out)
    gan = Model([infogan_in_noise, infogan_in_cat], [infogan_out_sigmoid, infogan_out_softmax])
    gan.compile(optimizer=infogan_opt,
                loss=['binary_crossentropy', 'categorical_crossentropy'],
                loss_weights=[1, 1])
    print('Summary of the InfoGAN model')
    gan.summary()
    return gan


def train_infogan(image_set, label_set, generator, discriminator_classifier, gan, losses,
                  label_rate=1,
                  batch_size=32,
                  n_epochs=100,
                  save_every=10,
                  save_mode='multi_channel',
                  save_filename_prefix=None):
    label_smooth = 0.1  # label smoothing factor
    n_train = image_set.shape[0]
    dim_noise = generator.layers[0].input_shape[1]
    dim_cat = generator.layers[1].input_shape[1]
    n_ch = discriminator_classifier.layers[0].input_shape[-1]
    for ie in range(n_epochs):
        print('epoch: %d' % (ie + 1))
        idx_randperm = np.random.permutation(n_train)
        n_batches = n_train // batch_size
        progbar = generic_utils.Progbar(n_batches*batch_size)
        for ib in range(n_batches):
            set_trainable(discriminator_classifier, True)
            if ib % 2 == 0: # pick real batch
                idx_batch = idx_randperm[range(ib*batch_size, ib*batch_size+batch_size)]
                X = image_set[idx_batch]
                y = np.random.uniform(low=1-label_smooth, high=1, size=(batch_size,))  # label smoothing
                # semi-supervision
                toss = np.random.binomial(1, label_rate)
                if toss > 0:
                    label = label_set[idx_batch]
                else:
                    _, label = get_noise(dim_noise, dim_cat, batch_size=batch_size)
            else: # pick fake batch
                noise_gen, label_gen = get_noise(dim_noise, dim_cat, batch_size=batch_size)  # generate noise and labels
                X = generator.predict([noise_gen, label_gen])
                y = np.random.uniform(low=0, high=label_smooth, size=(batch_size,))  # label smoothing
                label = label_gen
            # train discriminator and classifier
            d_c_loss = discriminator_classifier.train_on_batch(X, [y, label])
            losses['d'].append(d_c_loss[1])
            losses['c'].append(d_c_loss[2])

            # train generator and classifier
            noise_train, label_train = get_noise(dim_noise, dim_cat, batch_size=batch_size)
            y_one = np.ones((batch_size,), dtype=np.float32)
            set_trainable(discriminator_classifier, False)
            g_loss = gan.train_on_batch([noise_train, label_train], [y_one, label_train])
            losses['g'].append(g_loss[1])
            losses['c'][-1] += g_loss[2]

            # update progress bar
            progbar.add(batch_size, values=[("G loss", g_loss[1]),
                                            ("D loss", d_c_loss[1]),
                                            ("C loss", losses['c'][-1])])

        # plot interim results
        if ((ie + 1) % save_every == 0) or (ie == n_epochs - 1):
            if save_mode is 'multi_channel':
                # display generated images channel by channel
                for ic in range(n_ch):
                    save_filename_image_gen = '%s_cat_gen_ch%d_epoch%d.pdf' % (save_filename_prefix, ic, ie + 1)
                    plot_gen(generator, (dim_cat, 10), (15, 15), ic, True, save_filename_image_gen, cmap='gray')
            elif save_mode is 'color':
                save_filename_image_gen = '%s_cat_gen_epoch%d.pdf' % (save_filename_prefix, ie + 1)
                plot_gen(generator, (dim_cat, 10), (15, 15), 'color', True, save_filename_image_gen)

    # plot loss
    save_filename_loss = '%s_loss_epoch%d.pdf' % (save_filename_prefix, n_epochs)
    plot_loss(losses, True, save_filename_loss)



if __name__ == '__main__':

    losses = {'g': [], 'd': [], 'c': []}

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
    dim_noise = 50
    n_class = 10
    label_rate = 1

    # output folder
    output_path = './output/'
    if not path.exists(output_path):
        os.makedirs(output_path)
    save_mode = 'multi_channel'
    save_filename_prefix = '%sMNIST_infogan_noise%d_cat%d' % (output_path, dim_noise, n_class)

    # generator
    generator = build_generator(n_class, n_row, n_col, n_out_ch=n_ch, n_first_conv_ch=256, dim_noise=dim_noise)
    plot_model(generator, to_file='%s_generator_model.pdf' % save_filename_prefix, show_shapes=True)
    # discriminator and classifier
    discriminator_classifier = build_disc_class(n_class, n_row, n_col, n_in_ch=n_ch, n_last_conv_ch=256, lr=d_lr)
    plot_model(discriminator_classifier, to_file='%s_discriminator_classifier_model.pdf' % save_filename_prefix, show_shapes=True)
    # combined InfoGAN network
    infogan = build_infogan(generator, discriminator_classifier, lr=g_lr)
    plot_model(infogan, to_file='%s_infogan_model.pdf' % save_filename_prefix, show_shapes=True)
    infogan_weights_file = '%s_weights_epoch%d.hdf' % (save_filename_prefix, n_epochs)
    if path.isfile(infogan_weights_file):
        # load InfoGAN weights that already existed
        print('loading InfoGAN model weights')
        infogan.load_weights(infogan_weights_file)
    else:
        # train InfoGAN
        print('training InfoGAN model')
        y_train = np_utils.to_categorical(y_train, num_classes=n_class)
        train_infogan(x_train, y_train, generator, discriminator_classifier, infogan, losses,
                      label_rate=label_rate,
                      batch_size=batch_size,
                      n_epochs=n_epochs,
                      save_every=n_save_every,
                      save_mode=save_mode,
                      save_filename_prefix=save_filename_prefix)
        infogan.save_weights(infogan_weights_file)

    # accuracy test
    _, y_test_softmax = discriminator_classifier.predict(x_test)
    y_test_pred = np.argmax(y_test_softmax, axis=1)
    acc_pred, w_cluster = cluster_acc(y_test, y_test_pred)
    print('Classifier accuracy: %.4f' % acc_pred)
    print(w_cluster.astype(int))

    # generate new data and save the plot
    if save_mode is 'multi_channel':
        # display generated images channel by channel
        for ic in range(n_ch):
            save_filename_image_gen = '%s_cat_gen_ch%d_epoch%d.pdf' % (save_filename_prefix, ic, n_epochs)
            plot_gen(generator, (n_class, 10), (15, 15), ic, True, save_filename_image_gen, cmap='gray')
    elif save_mode is 'color':
        save_filename_image_gen = '%s_cat_gen_epoch%d.pdf' % (save_filename_prefix, n_epochs)
        plot_gen(generator, (n_class, 10), (15, 15), 'color', True, save_filename_image_gen)

    pass