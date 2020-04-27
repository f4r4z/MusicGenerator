# LSTM Music Generator

## About
Machine Learning project to generate music using LSTM.
Currently, it is being trained on Midi files of *Super Mario Bros 2*

## Running Program
You may use `Main.py --help` to see all the options
```C
Optional Arguments:
  -h, --help            Show this help message and exit
  -M MODEL, --Model MODEL
                        Model A or B (Default: Model 'A')
  -W WEIGHTS, --Weights WEIGHTS
                        Load weights to continue training (Default: None)
  -D DIRECTORY, --Directory DIRECTORY
                        Directory of Midi Files (Default: '.')
  -E EPOCHS, --Epochs EPOCHS
                        Number of Epochs (Default: 100)
  -O OUTPUTS, --Outputs OUTPUTS
                        Number of Generated Output(s) (Default: 1)
  -BS BATCH_SIZE, --Batch_Size BATCH_SIZE
                        Batch Size (Default: 128)
  -SL SEQUENCE_LENGTH, --Sequence_Length SEQUENCE_LENGTH
                        Sequence Length (Default: 100)
```

* There are two models, A and B. Checkout code to see the difference.
* Weights are optional to load. If you want to continue training from certain weights, you may use this feature.
* Set a directory where the MIDI file are contained.
* Set the number of Epochs, the default is 10.
* The Output is how many sound that will be generated after training. We recommend >= 3 
* Set Batch Size.
* Set Sequence Length.

All of these options have a set default.

> **Example Runs**

`Main.py --Model B --Directory Undertale --Epochs 10 --Outputs 5 --Batch_Size 128 --Sequence_Length 80`

## Recommendations
* We **strongly** recommend for the user to run this code and train on Anaconda, due to a drastic performance increase while using Anaconda.

## Contributors
* Yousif Kako
* Faraz Heravi
* Dr. Andrew Rosen
