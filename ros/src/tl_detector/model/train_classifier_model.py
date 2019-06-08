# Main function
import sys

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