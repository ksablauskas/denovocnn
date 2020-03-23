import numpy as np
import os
import math

import matplotlib.pyplot as plt

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.regularizers import l2

from keras.models import load_model

from denovonet.settings import NUMBER_CLASSES, BATCH_SIZE, MODEL_ARCHITECTURE, DEVELOPMENNT_MODEL_PATH, WORKING_DIRECTORY
from denovonet.settings import TRAINING_X, TRAINING_Y, VALIDATION_X, VALIDATION_Y
from denovonet.settings import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS

from denovonet.variants import TrioVariant

num_classes = NUMBER_CLASSES

def cnn(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                    input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model

depth = 3 * 9 + 2

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v2(input_shape, depth, num_classes=2):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def get_model(model_name, input_shape, num_classes):
    if model_name == 'cnn':
        model = cnn(input_shape, num_classes)
    elif model_name == 'resnet_v2':
        model = resnet_v2(input_shape, depth, num_classes)
    else:
        raise Exception('NameError','Unknown model name')

    return model

def get_number_images(images_directory):    
    dnms_path = os.path.join(images_directory, 'dnm')
    ivs_path = os.path.join(images_directory, 'iv')
    print(dnms_path, len(os.listdir(dnms_path)))
    print(ivs_path, len(os.listdir(ivs_path)))

    return (len(os.listdir(dnms_path)) + len(os.listdir(ivs_path)))

def get_steps_per_epoch(images_directory):
    total_images = get_number_images(images_directory)
    steps = math.ceil(total_images / BATCH_SIZE)

    return steps

def evaluate(models, IMAGES_FOLDER, DATASET_NAME):

    from sklearn.metrics import roc_curve, auc, roc_auc_score

    plt.figure()
    lw = 2

    for model in models:

        val_folder = os.path.join(IMAGES_FOLDER, DATASET_NAME, 'val')
        dnm_paths = [os.path.join(os.path.join(val_folder, 'dnm', path)) for path in os.listdir(os.path.join(val_folder, 'dnm'))]
        iv_paths = [os.path.join(os.path.join(val_folder, 'iv', path)) for path in os.listdir(os.path.join(val_folder, 'iv'))]

        validation_image_paths = dnm_paths + iv_paths

        y_true = np.zeros((len(validation_image_paths), 2)).astype(int)
        y_pred = np.zeros((len(validation_image_paths), 2)).astype(float)

        y_true_roc = []
        y_pred_roc = []

        tps = 0
        tns = 0
        fps = 0
        fns = 0

        for image_index, validation_image_path in enumerate(validation_image_paths):
            
            prediction = TrioVariant.predict_image_path(validation_image_path, model['model'])
            argmax = np.argmax(prediction, axis=1)

            ground_truth = 0
            if 'iv' in validation_image_path:
                ground_truth = 1
                y_true[image_index, 1] = 1
                y_true_roc.append(-1) 
            else:
                y_true[image_index, 0] = 1
                y_true_roc.append(1) 
            
            y_pred[image_index] = prediction
            y_pred_roc.append(prediction[0,0])

            if argmax == 0 and ground_truth == 0:
                tps += 1
            elif argmax == 0 and ground_truth == 1:
                fps += 1
            elif argmax == 1 and ground_truth == 0:
                fns += 1
            elif argmax == 1 and ground_truth == 1:
                tns += 1
            else:
                print('error')

        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true_roc, y_pred_roc)
        # print(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)

        accuracy = (tps + tns) / (tps + tns + fps + fns)

        sensitivity = tps / (tps + fns) #Recall
        specificity = tns / (tns + fps)
        precision = tps / (tps + fps)

        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)

        distances = np.sqrt(np.square(false_positive_rate) + np.square(np.array(true_positive_rate) - 1))
        minimal_distance_index = np.argmin(distances)

        threshold = 1 - thresholds[minimal_distance_index]

        print('Model: {}'.format(model['name']))
        print('Recommended threshold: {}'.format(threshold))
        print('ROC AUC score: {}'.format(roc_auc))
        print('Accuracy: {}'.format(accuracy))
        print('True positives: {}'.format(tps))
        print('False positives: {}'.format(fps))
        print('True negatives: {}'.format(tns))
        print('False negative: {}'.format(fns))
        print('Sensitivity: {}'.format(sensitivity))
        print('Specificity: {}'.format(specificity))
        print('F1 score: {}'.format(f1_score))
        print()

        # Plot
        plt.plot(false_positive_rate, true_positive_rate, color=model['color'],
                lw=lw, label='ROC curve {} '.format(model['name']) + '(area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
    # K.clear_session()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic {}'.format(DATASET_NAME))
    plt.legend(loc="lower right")
    plt.show()

    

def train(EPOCHS, IMAGES_FOLDER, DATASET_NAME, output_model_path, continue_training=False, input_model_path=None):

    train_folder = os.path.join(IMAGES_FOLDER, DATASET_NAME, 'train')
    val_folder = os.path.join(IMAGES_FOLDER, DATASET_NAME, 'val')

    train_steps = get_steps_per_epoch(train_folder)
    val_steps = get_steps_per_epoch(val_folder)
    print('train steps:',train_steps)
    print('val_steps:',val_steps)

    batch_size = BATCH_SIZE
    num_classes = NUMBER_CLASSES

    input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

    # learning rate schedule
    def step_decay(epoch):
        initial_lrate = 0.1
        drop = 0.5
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate
    
    # param_grid = dict(
    #     bacth_size=[12,16,20,24,28,32,36]
    # )

    # grid = GridSearchCV(estimator=model, param_grid=paam_grid, n_jobs=2, verbose=1, scoring='neg_log_loss')
    # grid_result = grid.fit(x_train, y_train)

    if continue_training:
        model = load_model(input_model_path)
    else:
        model = get_model(MODEL_ARCHITECTURE, input_shape, NUMBER_CLASSES)

    adad = keras.optimizers.Adadelta()

    

    # learning schedule callback
    lrate = keras.callbacks.LearningRateScheduler(step_decay)

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                            #   min_delta=0,
                              patience=3,
                              verbose=1,
                              mode='auto')

    callbacks_list = [lrate, early_stopping]

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        # shear_range=0.2,
        zoom_range=0.2,
        # zca_whitening=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        # rotation_range=10,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(
        rescale=1./255
    )

    train_generator = train_datagen.flow_from_directory(
        train_folder,
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
            val_folder,
            target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='categorical')

    model.compile(loss=keras.losses.categorical_crossentropy ,
                optimizer=adad,
                metrics=['accuracy'])

    # model.fit(x_train, y_train,
    #         batch_size=batch_size,
    #         epochs=epochs,
    #         verbose=1,
    #         callbacks=callbacks_list,
    #         validation_data=(x_val, y_val)
    #         )

    model.fit_generator(
        train_generator,
        # steps_per_epoch=180, #snps
        # steps_per_epoch=10, #insertions
        steps_per_epoch=train_steps, #deletions
        epochs=EPOCHS,
        validation_data=validation_generator,
        # validation_steps=3 #insertions
        # validation_steps=45 #snps
        validation_steps=val_steps #deletions
    )

    model.save(output_model_path)
    print('Model saved as : {}'.format(output_model_path))

    return model