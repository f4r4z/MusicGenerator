import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm

from keras.layers import Flatten
from keras.layers import Bidirectional
#from keras_self_attention import SeqSelfAttention

from keras import utils
from keras.callbacks import ModelCheckpoint


class Train():
    def __init__(self, directory, epochs, batch_size, sequence_length, model, weights_file=None):
        self.directory = directory
        self.num_epochs = epochs
        self.batch_Size = batch_size
        self.sequence_length = sequence_length
        self.model = model
        self.weights = weights_file
        
        
    def train_network(self):
        """ Train a Neural Network to generate music """
        notes = self.get_notes()
        #with open('notes', 'rb') as filepath:
        #    notes = pickle.load(filepath)
        
        # get amount of pitch names
        n_vocab = len(set(notes))
    
        network_input, network_output = self.prepare_sequences(notes, n_vocab)
        
        print("Training...")
        if self.model == 'A':
            print("Using Model A")
            model = self.model_A(network_input, n_vocab)
        elif self.model == 'B':
            print("Using Model B")
            model = self.model_B(network_input, n_vocab)
    
        self.train(model, network_input, network_output)
    
    
    def get_notes(self):
        """ Get all the notes and chords from the midi files in the directory """
        notes = []
        for file in glob.glob(self.directory + "/*.mid"):
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
            break
            
        if len(notes) == 0:
            raise ValueError("No Midi files were found in {} directory".format(self.directory))
            
        with open('notes', 'wb') as filepath:
            pickle.dump(notes, filepath)
    
        return notes
    
    
    def prepare_sequences(self, notes, n_vocab):
        """ Prepare the sequences used by the Neural Network """
        sequence_length = self.sequence_length
    
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
    
    
    def model_A(self, network_input, n_vocab):
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
        
        if self.weights != None:
            model.load_weights(self.weights)
        else:
            print("Weights were not loaded, starting from nothing")  
            
        return model
    
    
    def model_B(self, network_input, n_vocab):
        """ Create the structure of the neural network """
        model = Sequential()
        model.add(Bidirectional(LSTM(
            512,
            input_shape=(network_input.shape[1], network_input.shape[2]),
            return_sequences=True
        )))
        #model.add(SeqSelfAttention(attention_activation='sigmoid'))
        model.add(Dropout(0.3))
        
        model.add(LSTM(512, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(Flatten())
        
        model.add(Dense(n_vocab))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    
        try:
            model.load_weights(self.weights)
        except:
            print("Weights were not loaded, starting from nothing")
    
        return model

    
    def train(self, model, network_input, network_output):
        """ Train the neural network """
        filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
        checkpoint = ModelCheckpoint(
            filepath,
            period=1,
            monitor='loss',
            verbose=1,
            save_best_only=False,
            mode='min'
        )
        callbacks_list = [checkpoint]
        
        model.fit(network_input, network_output, epochs=self.num_epochs, batch_size=self.batch_Size, callbacks=callbacks_list)
