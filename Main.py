import Train, Generate
import os, glob, argparse


def last_generated_weights():
    list_of_files = glob.glob('*')
    latest_file = max(list_of_files, key=os.path.getctime)
    return os.path.split(latest_file)[1]


def main():
    parser = argparse.ArgumentParser(description="Train Midi files on an LSTM." + 
                                     "Note: Weights will be saved after every Epoch")
    
    parser.add_argument('-M', '--Model', default='A', type=str, help="Model A or B (Default: Model 'A')")
    parser.add_argument('-W', '--Weights', default=None, type=str, help="Load weights to continue training (Default: None)")
    parser.add_argument('-D', '--Directory', default='.', type=str, help="Directory of Midi Files (Default: '.')")
    parser.add_argument('-E', '--Epochs', default=100, type=int, help="Number of Epochs (Default: 100)")
    parser.add_argument('-O', '--Outputs', default=1, type=int, help="Number of Generated Output(s) (Default: 1)")
    parser.add_argument('-BS', '--Batch_Size', default=128, type=int, help="Batch Size (Default: 128)")
    parser.add_argument('-SL', '--Sequence_Length', default=100, type=int, help="Sequence Length (Default: 100)")
    
    args = parser.parse_args()
    
    # You may edit these variables if you are using Anaconda
    model = args.Model
    weights = args.Weights
    directory = args.Directory
    num_epochs = args.Epochs
    num_outputs = args.Outputs
    batch_size = args.Batch_Size
    sequence_length = args.Sequence_Length
    
    # Initialize Training Neural Network
    train_NN = Train.Train(directory, num_epochs, batch_size, sequence_length, model, weights)
    # Train Neural Network
    train_NN.train_network()
    
    # Gets the name of the last edited/created file in the current directory
    new_weights = last_generated_weights()
    
    print("Generating...")
    print("Generating for {}".format(new_weights))
    # Initialize Generate Neural Network
    gen = Generate.Generate(sequence_length, model, new_weights, num_outputs)
    # Generate music
    gen.generate_music()


if __name__ == '__main__':
    main()
