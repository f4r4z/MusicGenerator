import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization as BatchNorm
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ModelCheckpoint


# Location of midi files to train on
directory = "Super Mario Bros 2"

# Load weights from a previous training to continue where you left off
# You may comment this out if you would like to start training from zero
weights = "weights (10 Songs with Duration 50 Epochs).hdf5"

# Number of iterations through the Neural Network
num_epochs = 50


def train_network():
    """ Train a Neural Network to generate music """
    notes = get_notes()
    #with open('notes', 'rb') as filepath:
    #    notes = pickle.load(filepath)
    
    # get amount of pitch names
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)


def get_notes(songs='all'):
    """ Get all the notes and chords from the midi files in the directory """
    notes = []
    ite = 0

    for file in glob.glob(directory + "/*.mid"):
        if ite < 10:
            midi = converter.parse(file)

            print("Parsing %s" % file)

            notes_to_parse = None

            try: # file has instrument parts
                s2 = instrument.partitionByInstrument(midi)
                notes_to_parse = s2.parts[0].recurse() 
            except: # file has notes in a flat structure
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch) + ' ' + str(element.quarterLength))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder) + ' ' + str(element.quarterLength))
        ite += 1
        

    with open('notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes


def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = utils.to_categorical(network_output)
    
    return (network_input, network_output)


def create_network(network_input, n_vocab):
    """ Create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    try:
        model.load_weights(weights)
    except:
        print("Weights were not loaded, starting from nothing")

    return model


def train(model, network_input, network_output):
    """ Train the neural network """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=num_epochs, batch_size=128, callbacks=callbacks_list)


if __name__ == '__main__':
    train_network()
