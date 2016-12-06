# shape
Neural Network to detect 2D shapes in images using a GANN approach. This combines the heuristic approach of a Genetic Algorithm, and the precision of backpropagation, to reach optimum convergence.

There are 3 options to run this program:

train:
    This will train the network using the "training_data/" directory. Each subdirectory will be considered as the label for a class. This option takes 1 argument which is the error value to reach before exiting.

test:
    This option will force the net to test itself with the data under test_data, which is assumed to have the same labels as the training data. The neural net will test its performance using this.

predict:
    This option takes an image file as an argument, and classifies the image. It is assumed that the image has the same dimensions as the training_data.
