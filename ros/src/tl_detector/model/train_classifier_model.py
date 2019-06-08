import sys
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Conv2D, Flatten, Dense, Dropout

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