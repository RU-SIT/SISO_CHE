from keras.models import Sequential, Model
from keras.layers import Convolution2D, Input, BatchNormalization, Conv2D, Activation, Lambda, Subtract, Conv2DTranspose, PReLU
from keras.regularizers import l2
from keras.layers import Reshape, Dense, Flatten
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
from scipy.io import loadmat
import keras.backend as K
import numpy as np
import math
from scipy import interpolate
#from scipy.misc import imresize
import matplotlib.pyplot as plt
import os
import pdb

def unit_scaling(data, label):
    """
    Scale each real and imaginary component pixelwise.
    :param data: Data to be scaled.
    :param label: Labels to be scaled.
    :return: Scaled data and labels.
    """
    data = np.array(data)
    label = np.array(label)

    denom = np.sqrt(np.sum(data ** 2, axis=-1, keepdims=True)) + 1e-8  # Avoid division by zero
    scaled_data = data / denom
    scaled_label = label / denom

    return scaled_data, scaled_label, denom

@staticmethod
def standard_scaling(x, eps=1e-8):
    """
    Scale the real and imaginary channels separately into [-1, 1].
    
    Args:
        x: np.ndarray with shape (..., 2), where last dim=2 (real, imag).
        eps: small constant to avoid division by zero.
    
    Returns:
        x_scaled: scaled data in [-1,1], same shape as x
        params: dict with min/max per channel for unscaling
    """
    # Split real/imag
    real = x[..., 0]
    imag = x[..., 1]
    
    # Compute per-channel min/max
    min_real, max_real = real.min(), real.max()
    min_imag, max_imag = imag.min(), imag.max()
    
    # Scale to [-1,1]
    real_scaled = 2.0 * (real - min_real) / (max_real - min_real + eps) - 1.0
    imag_scaled = 2.0 * (imag - min_imag) / (max_imag - min_imag + eps) - 1.0
    
    # Recombine
    x_scaled = np.stack([real_scaled, imag_scaled], axis=-1)
    
    params = {
        "min_real": min_real, "max_real": max_real,
        "min_imag": min_imag, "max_imag": max_imag
    }
    return x_scaled, params

@staticmethod
def unscale_standard(x_scaled, params, eps=1e-8):
    """
    Reverse the standard scaling to recover original values.
    
    Args:
        x_scaled: scaled array (..., 2), with values in [-1,1]
        params: dict with min/max from standard_scaling
    Returns:
        x_unscaled: recovered data, same shape
    """
    real_s = x_scaled[..., 0]
    imag_s = x_scaled[..., 1]
    
    real = (real_s + 1.0) * 0.5 * (params["max_real"] - params["min_real"] + eps) + params["min_real"]
    imag = (imag_s + 1.0) * 0.5 * (params["max_imag"] - params["min_imag"] + eps) + params["min_imag"]
    
    return np.stack([real, imag], axis=-1)
    
    
def SRCNN_model(lr):
    input_shape = (612, 14, 2)  
    x = Input(shape=input_shape)
    c1 = Conv2D(64, (9, 9), activation='tanh', kernel_initializer='he_normal', padding='same')(x)
    c2 = Conv2D(32, (1, 1), activation='tanh', kernel_initializer='he_normal', padding='same')(c1)
    c3 = Conv2D(2, (5, 5), kernel_initializer='he_normal', padding='same')(c2)  # Output to single channel
    model = Model(inputs=x, outputs=c3)
    
    # Compile the model
    adam = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    
    return model

def SRCNN_train(train_data, train_label, bsize, n_epoch):
    srcnn_model = SRCNN_model()
    print(srcnn_model.summary())

    # Save the model after the final epoch
    checkpoint = ModelCheckpoint("SRCNN_final_model.keras", monitor='val_loss', save_best_only=False, save_weights_only=False, mode='auto', verbose=1)
    callbacks_list = [checkpoint]

    # Train the model for 2000 epochs
    srcnn_model.fit(train_data, train_label, batch_size=bsize,
                    callbacks=callbacks_list, shuffle=False, epochs= n_epoch, verbose=1)

    # Save the final model after training
    srcnn_model.save_weights(f"SRCNN_final.weights.h5")

def fine_tune_SRCNN(new_train_data, new_train_label):
    # Load the pre-trained model
    srcnn_model = SRCNN_model()
    srcnn_model.load_weights(f"SRCNN_final.weights.h5")
    
    # Fine-tune the model on unseen data
    checkpoint = ModelCheckpoint("SRCNN_fine_tuned_model.keras", monitor='val_loss', save_best_only=False, save_weights_only=False, mode='auto', verbose=1)
    callbacks_list = [checkpoint]

    # Fine-tune the model for 500 epochs (or adjust as needed)
    srcnn_model.fit(new_train_data, new_train_label, batch_size=128,
                    callbacks=callbacks_list, shuffle=True, epochs=5, verbose=1)

    # Save the fine-tuned model
    srcnn_model.save_weights(f"SRCNN_fine_tuned.weights.h5")

def SRCNN_predict(input_data , num_pilots):
    srcnn_model = SRCNN_model()
    srcnn_model.load_weights("SRCNN_pred.weights.h5")
    predicted  = srcnn_model.predict(input_data)
    return predicted

  
def DNCNN_model (lr):
  
    inpt = Input(shape=(None,None,2))
    # 1st layer, Conv+tanh
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(inpt)
    x = Activation('tanh')(x)
    # 18 layers, Conv+BN+tanh
    for i in range(18):
        x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('tanh')(x)   
    # last layer, Conv
    x = Conv2D(filters=2, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = Subtract()([inpt, x])   # input - noise
    model = Model(inputs=inpt, outputs=x)
    adam = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8) 
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])    
    return model

def DNCNN_train(train_data, train_label, bsize, n_epoch):
    dncnn_model = DNCNN_model()
    print(dncnn_model.summary())

    # Save the model after the final epoch
    checkpoint = ModelCheckpoint("DNCNN_final_model.keras", monitor='val_loss', save_best_only=False, save_weights_only=False, mode='auto', verbose=1)
    callbacks_list = [checkpoint]

    # Train the model for 2000 epochs
    dncnn_model.fit(train_data, train_label, batch_size=bsize,
                    callbacks=callbacks_list, shuffle=True, epochs=n_epoch, verbose=1)

    # Save the final model after training
    dncnn_model.save_weights(f"DNCNN_final.weights.h5")
    
def fine_tune_DNCNN(new_train_data, new_train_label):
    # Load the pre-trained model
    dncnn_model = DNCNN_model()
    dncnn_model.load_weights(f"/home/CAMPUS/rghasemi/projects/related_papers/ChannelNet/DNCNN_final_model.keras")
    
    # Fine-tune the model on unseen data
    checkpoint = ModelCheckpoint("DNCNN_fine_tuned_model.keras", monitor='val_loss', save_best_only=False, save_weights_only=False, mode='auto', verbose=1)
    callbacks_list = [checkpoint]

    # Fine-tune the model for 5 epochs
    dncnn_model.fit(new_train_data, new_train_label, batch_size=128,
                    callbacks=callbacks_list, shuffle=True, epochs=5, verbose=1)

    # Return predictions after fine-tuning
    predictions = dncnn_model.predict(new_train_data)
    
    return predictions

      
def DNCNN_predict(lr, input_data, channel):
  dncnn_model = DNCNN_model(lr)
  dncnn_model.load_weights("DNCNN_prediction_" +  str(channel) + ".weights.h5")
  predicted  = dncnn_model.predict(input_data)
  return predicted
  
  
# Function to visualize and save prediction results
def visualize_and_save_results(original_data, predicted_data, title, filename):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_data, cmap='BuGn_r')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(predicted_data, cmap='BuGn_r')
    plt.title('Predicted')
    plt.axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def visualize_pilot_values(noisy_data, r, c, title, filename):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(noisy_data, cmap='gray')
    plt.scatter(c, r, marker='o', color='r', label=filename)
    plt.title(title)
    plt.axis('off')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(c, r, marker='o', color='r', label=filename)
    plt.title(filename)
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.gca().invert_yaxis()
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
