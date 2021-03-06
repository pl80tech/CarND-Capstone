'''
Script to train model with specified settings & hyperparameter

Command:
$ python train_classified_model.py arg1 arg2 arg3 arg4 arg5

    modelArch = sys.argv[1]
    dataset = sys.argv[2]
    useDropout = sys.argv[3]
    batchsize = int(sys.argv[4])
    n_epoch = int(sys.argv[5])

Input:
- arg1: model architecture
- arg2: dataset for training and validating (single or combined dataset)
- arg3: use dropout in model or not
- arg4: batch size
- arg5: number of epoch

Output:
- History file (with loss and accuracy) and model file (.h5) after training is completed
'''

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%matplotlib inline
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Conv2D, Flatten, Dense, Dropout
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

# Add layers for NVIDIA model (Architecture #1)
def Nvidia_model_1(model, useDropout):
    model.add(Conv2D(24, (5,5), strides=(2,2), activation='relu'))
    if useDropout == "True":
        model.add(Dropout(0.2))
    model.add(Conv2D(36, (5,5), strides=(2,2), activation='relu'))
    if useDropout == "True":
        model.add(Dropout(0.2))
    model.add(Conv2D(48, (5,5), strides=(2,2), activation='relu'))
    if useDropout == "True":
        model.add(Dropout(0.2))
    model.add(Conv2D(64, (3,3), activation='relu'))
    if useDropout == "True":
        model.add(Dropout(0.2))
    model.add(Conv2D(64, (3,3), activation='relu'))
    if useDropout == "True":
        model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(4))

    return model

# Add layers for NVIDIA model (Architecture #2)
def Nvidia_model_2(model, useDropout):
    model.add(Conv2D(24, (5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(36, (5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(48, (5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Flatten())
    if useDropout == "True":
        model.add(Dropout(0.2))
    model.add(Dense(100))
    if useDropout == "True":
        model.add(Dropout(0.2))
    model.add(Dense(50))
    if useDropout == "True":
        model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Dense(4))

    return model

# Create model
def create_model(modelArch, input_shape, useDropout):
    # Model definition
    model = Sequential()

    # Preprocessing layers
    model.add(Lambda(lambda x: x / 255. - 0.5, input_shape=input_shape))
    model.add(Cropping2D(cropping=((50, 20), (0, 0))))

    # Model architecture
    if modelArch == "1":
        model = Nvidia_model_1(model, useDropout)
    elif modelArch == "2":
        model = Nvidia_model_2(model, useDropout)
    else:
        print("Not defined yet. To be added.")

    # Define the loss, optimizer and metrics
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    return model

# Path to the specified dataset (single or combined)
# dataset == "3"   --> ["./dataset3/"]
# dataset == "345" --> ["./dataset3/", "./dataset4/", "./dataset5/"]
def dataset_getpath(dataset):
    datapath = []
    for i in range(len(dataset)):
        path = "./dataset" + dataset[i] + "/"
        datapath.append(path)
    return datapath

# Get list of paths to image and labels
def getData(path, label):
    paths = os.listdir(path)
    paths.sort()
    path_to_images = []
    labels = []
    for fname in paths:
        path_to_images.append(path + fname)
        labels.append(label)
    return path_to_images, labels

# Get full dataset from image folder for each label
def getFullData(path):
    X = []
    y = []
    for label_class, folder in enumerate(['Red', 'Yellow', 'Green', 'Unknown']):
        X_temp, y_temp = getData(path + folder + '/', label_class)
        X += X_temp
        y += y_temp
    return X, y

# Get combined full dataset from specified paths
def getCombinedFullData(paths):
    image_list = []
    label_list = []
    for path in paths:
        X, y = getFullData(path)
        image_list += X
        label_list += y
    return image_list, label_list

# Create generator
def generator(X, y, batchsize):
    num_samples = len(X)
    while 1:
        for offset in range(0, num_samples, batchsize):
            X_batch_filenames = X[offset:offset+batchsize]
            y_batch = y[offset:offset+batchsize]
            X_batch = read_images(X_batch_filenames)
            y_batch = np.array(y_batch)
            yield X_batch, y_batch

# A function to read the image from the list of image names
def read_images(img_list):
    images = []
    for i in range(len(img_list)):
        filepath = img_list[i]
        image = cv2.imread(filepath)
        size = (320, 240)
        image = cv2.resize(image, size)
        images.append(image)
    return np.array(images)

# Collect training information for savefile
import datetime
def get_traininfo(modelArch, dataset, useDropout, batchsize, n_epoch):
    # Get current date & time
    dt_now = datetime.datetime.now()

    # Filename (with extra information)
    saveinfo = str(dt_now.date()) + '_' + str(dt_now.hour) + str(dt_now.minute) \
             + '_modelArch' + str(modelArch) \
             + '_dataset' + str(dataset)

    if useDropout == "True":
        saveinfo += '_useDropout'
    else:
        saveinfo += '_noDropout'

    saveinfo += '_batch' + str(batchsize) + '_epoch' + str(n_epoch)

    return saveinfo

import pickle
# Save history
def save_history(history_obj, saveinfo):
    hist_pickle = {}
    hist_pickle["loss"] = history_obj.history['loss']
    hist_pickle["val_loss"] = history_obj.history['val_loss']
    hist_pickle["acc"] = history_obj.history['acc']
    hist_pickle["val_acc"] = history_obj.history['val_acc']

    # Filename (with extra information) to save history
    savefile = './trained_data/history_' + saveinfo + '.bin'

    print("Save history to pickle file: " + savefile)

    pickle.dump(hist_pickle, open(savefile,"wb"))

# Load history
def load_history(path):
    hist_pickle = pickle.load(open(path, "rb"))
    loss = hist_pickle["loss"]
    val_loss = hist_pickle["val_loss"]
    acc = hist_pickle["acc"]
    val_acc = hist_pickle["val_acc"]
    return loss, acc, val_loss, val_acc

# Save model
def model_save(model, saveinfo):
    # Filename (with extra information) to save model
    savefile = './trained_data/model_' + saveinfo + '.h5'
    print("Save model to file: " + savefile)
    model.save(savefile)

# Plot learning curve
def plot_learning_curve(path):
    plt.figure(figsize=(10,6))
    # Load calculated accuracy from history
    loss, acc, val_loss, val_acc = load_history(path)

    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('Learning curve')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training set', 'Validation set'], loc='upper left')
    plt.show()

# Plot multiple learning curves
def plot_multiple_learning_curves(paths, curve_info, legend_position, savefile):
    plt.figure(figsize=(10,6))
    for path in paths:
        # Load calculated accuracy and loss from history
        loss, acc, val_loss, val_acc = load_history(path)
        plt.plot(acc)
        plt.plot(val_acc)
    plt.title('Learning curve')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    if legend_position == "out":
        plt.legend(curve_info, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    elif legend_position == "in_UpperRight":
        plt.legend(curve_info, loc='upper right')
    elif legend_position == "in_LowerLeft":
        plt.legend(curve_info, loc='lower left')

    # Save file if the filename is specified
    if savefile != None:
        print("Save plotted image to " + savefile)
        plt.savefig(savefile)
    plt.show()

# Pipeline to train and save the model
def training_pipeline(modelArch, dataset, useDropout, batchsize, n_epoch):
    # Shape of input data
    input_shape = (240, 320, 3)

    # Create model
    model = create_model(modelArch, input_shape, useDropout)

    # Get the path to the dataset
    datapath = dataset_getpath(dataset)

    # Get single or combined data
    image_list, label_list = getCombinedFullData(datapath)

    print("length of image list =", len(image_list))

    # Shuffle and split data
    image_list_shuffle, label_list_shuffle = shuffle(image_list, label_list, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(image_list_shuffle, label_list_shuffle, test_size=0.2, random_state=0)

    # Change label to category
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)

    # Generate samples with batch size on demand (yield)
    train_generator = generator(X_train, y_train, batchsize)
    validation_generator = generator(X_val, y_val, batchsize)
    samples_per_epoch = len(X_train)
    steps_per_epoch = int(samples_per_epoch/batchsize)
    X_val_len = len(X_val)
    val_steps = int(X_val_len/batchsize)
    history_object = model.fit_generator(generator = train_generator,
                                        steps_per_epoch = steps_per_epoch,
                                        validation_data = validation_generator,
                                        validation_steps = val_steps,
                                        epochs = n_epoch, verbose = 1)

    print("-----------------------------")

    # Collect training information for savefile
    saveinfo = get_traininfo(modelArch, dataset, useDropout, batchsize, n_epoch)

    # Save model
    model_save(model, saveinfo)

    # Save history
    save_history(history_object, saveinfo)

# Main function
def main():       
    # Initial setting and hyperparameters
    modelArch = sys.argv[1]
    dataset = sys.argv[2]
    useDropout = sys.argv[3]
    batchsize = int(sys.argv[4])
    n_epoch = int(sys.argv[5])
    
    # Call the pipeline
    print("Call pipeline with " \
        + "model architecture#" + str(modelArch) \
        + ", dataset#" + str(dataset) \
        + ", useDropout = " + str(useDropout) \
        + ", batchsize = " + str(batchsize) \
        + ", n_epoch = " + str(n_epoch))

    training_pipeline(modelArch, dataset, useDropout, batchsize, n_epoch)

if __name__ == "__main__":
    main()