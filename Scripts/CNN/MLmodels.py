import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, ELU, PReLU
from keras.layers import BatchNormalization, Input, Add, Concatenate
from keras.models import Model


class RPNet:
    def __init__(self):
        self.layers = dict()
        self.history = None
        self.rpnet_model = None

    def summary(self):
        self.rpnet_model.summary()

    def load(self, path_model):
        from keras.models import load_model
        self.rpnet_model = load_model(path_model)

    def train(self, X_train, y_train, X_valid, y_valid, epochs=10, bs=1):
        self.history = self.rpnet_model.fit(X_train, y_train, batch_size=bs, epochs=epochs, shuffle=True,
                                            validation_data=(X_valid, y_valid))

    def save(self, path_save):
        self.rpnet_model.save(path_save)

    def predict(self, X_pred):
        output = self.rpnet_model.predict(X_pred)
        output = np.argmax(output, axis=-1)
        return output

    def build_model(self, input_shape, n_classes=2, filter_size=(128, 256), kernel_size=3):

        self.layers['input'] = Input(input_shape)
        self.layers['batch_norm_1'] = BatchNormalization()(self.layers['input'])
        self.layers['convolution_1'] = Conv2D(filter_size[0], (kernel_size, kernel_size), padding='same',
                                              kernel_initializer='he_normal')(self.layers['batch_norm_1'])
        self.layers['prelu_activation_1'] = PReLU()(self.layers['convolution_1'])
        self.layers['max_pooling'] = MaxPooling2D(pool_size=(2, 2))(self.layers['prelu_activation_1'])

        self.layers['batch_norm_2'] = BatchNormalization()(self.layers['max_pooling'])
        self.layers['convolution_2'] = Conv2D(filter_size[1], (kernel_size, kernel_size), padding='same',
                                              kernel_initializer='he_normal')(self.layers['batch_norm_2'])
        self.layers['elu_activation'] = ELU()(self.layers['convolution_2'])
        self.layers['up_sampling'] = UpSampling2D(size=(2, 2))(self.layers['elu_activation'])

        self.layers['batch_norm_3'] = BatchNormalization()(self.layers['up_sampling'])
        self.layers['deconvolution_1'] = Conv2D(filter_size[0], (kernel_size, kernel_size), padding='same',
                                                kernel_initializer='he_normal')(self.layers['batch_norm_3'])
        self.layers['prelu_activation_2'] = PReLU()(self.layers['deconvolution_1'])

        self.layers['fusion_1_3'] = Add()([self.layers['prelu_activation_1'], self.layers['prelu_activation_2']])

        self.layers['batch_norm_4'] = BatchNormalization()(self.layers['fusion_1_3'])

        self.layers['prediction'] = Conv2D(n_classes, (1, 1), padding='valid', kernel_initializer='he_normal',
                                           activation='sigmoid')(self.layers['batch_norm_4'])

        self.rpnet_model = Model(inputs=self.layers['input'], outputs=self.layers['prediction'])
        self.rpnet_model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

    def plot_history(self):
        assert (self.history is not None), "Model not trained/ a model is loaded. Try training to visualize history."
        # Plot training & validation accuracy values
        f1 = plt.figure()
        f1.plot(self.history.history['acc'])
        f1.plot(self.history.history['val_acc'])
        f1.title('Model accuracy')
        f1.ylabel('Accuracy')
        f1.xlabel('Epoch')
        f1.legend(['Train', 'Test'], loc='upper left')

        # Plot training & validation loss values
        f2 = plt.figure()
        f2.plot(self.history.history['loss'])
        f2.plot(self.history.history['val_loss'])
        f2.title('Model loss')
        f2.ylabel('Loss')
        f2.xlabel('Epoch')
        f2.legend(['Train', 'Test'], loc='upper left')

        plt.show()


class DeepFusionNet:
    """ This class is only for internal testing. It should not be imported.
    HIGHLY UNSTABLE. ONLY FOR BETA TESTING!!!! """
    def __init__(self):
        self.layers = dict()
        self.history = None
        self.dfnet_model = None

    def summary(self):
        self.dfnet_model.summary()

    def load(self, path_model):
        from keras.models import load_model
        self.dfnet_model = load_model(path_model)

    def train(self, X1_train, X2_train, y_train,X1_valid, X2_valid, y_valid, epochs=10, bs=1):
        self.history = self.dfnet_model.fit([X1_train, X2_train], y_train, batch_size=bs, epochs=epochs, shuffle=True,
                                            validation_data=([X1_valid, X2_valid], y_valid))

    def save(self, path_save):
        self.dfnet_model.save(path_save)

    def predict(self, X_pred):
        output = self.dfnet_model.predict(X_pred)
        output = np.argmax(output, axis=-1)
        return output

    def build_model(self, input_shape, disp_shape, n_classes=2, filter_size=(128, 256), kernel_size=3):

        self.layers['input_main'] = Input(input_shape)
        self.layers['batch_norm_1'] = BatchNormalization()(self.layers['input_main'])
        self.layers['convolution_1'] = Conv2D(filter_size[0], (kernel_size, kernel_size), padding='same',
                                              kernel_initializer='he_normal')(self.layers['batch_norm_1'])
        self.layers['prelu_activation_1'] = PReLU()(self.layers['convolution_1'])
        self.layers['max_pooling'] = MaxPooling2D(pool_size=(2, 2))(self.layers['prelu_activation_1'])

        self.layers['batch_norm_2'] = BatchNormalization()(self.layers['max_pooling'])
        self.layers['convolution_2'] = Conv2D(filter_size[1], (kernel_size, kernel_size), padding='same',
                                              kernel_initializer='he_normal')(self.layers['batch_norm_2'])
        self.layers['elu_activation'] = ELU()(self.layers['convolution_2'])
        self.layers['up_sampling'] = UpSampling2D(size=(2, 2))(self.layers['elu_activation'])

        self.layers['batch_norm_3'] = BatchNormalization()(self.layers['up_sampling'])
        self.layers['deconvolution_1'] = Conv2D(filter_size[0], (kernel_size, kernel_size), padding='same',
                                                kernel_initializer='he_normal')(self.layers['batch_norm_3'])
        self.layers['prelu_activation_2'] = PReLU()(self.layers['deconvolution_1'])

        self.layers['fusion_1_3'] = Add()([self.layers['prelu_activation_1'], self.layers['prelu_activation_2']])

        self.layers['batch_norm_4'] = BatchNormalization()(self.layers['fusion_1_3'])

        #self.layers['intermediate'] = Conv2D(1, (1, 1), padding='valid', kernel_initializer='he_normal',
        #                                     activation='sigmoid')(self.layers['batch_norm_4'])

        self.layers['intermediate'] = Conv2D(1, (1, 1), padding='valid', kernel_initializer='he_normal')(self.layers['batch_norm_4'])

        self.layers['input_disp'] = Input(disp_shape)
        self.layers['batch_norm_5'] = BatchNormalization()(self.layers['input_disp'])

        self.layers['add_1'] = Add()([self.layers['batch_norm_5'], self.layers['intermediate']])
        self.layers['batch_norm_6'] = BatchNormalization()(self.layers['add_1'])

        self.layers['final'] = Conv2D(n_classes, (1, 1), padding='valid', kernel_initializer='he_normal', activation='softmax')(self.layers['batch_norm_6'])

        self.dfnet_model = Model(inputs=[self.layers['input_main'], self.layers['input_disp']], outputs=self.layers['final'])
        self.dfnet_model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

    def plot_history(self):
        assert (self.history is not None), "Model not trained/ a model is loaded. Try training to visualize history."
        # Plot training & validation accuracy values
        f1 = plt.figure()
        f1.plot(self.history.history['acc'])
        f1.plot(self.history.history['val_acc'])
        f1.title('Model accuracy')
        f1.ylabel('Accuracy')
        f1.xlabel('Epoch')
        f1.legend(['Train', 'Test'], loc='upper left')

        # Plot training & validation loss values
        f2 = plt.figure()
        f2.plot(self.history.history['loss'])
        f2.plot(self.history.history['val_loss'])
        f2.title('Model loss')
        f2.ylabel('Loss')
        f2.xlabel('Epoch')
        f2.legend(['Train', 'Test'], loc='upper left')

        plt.show()









