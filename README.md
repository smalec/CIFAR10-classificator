# CIFAR10-classificator

### About CIFAR-10
CIFAR-10 is a dataset consisting of 60.000 32x32 images. Each of them belongs to one out of ten categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship or truck. The problem of classification CIFAR-10 images is one of the most popular classification problem for neural networks.

### Downloading code
If you are on Linux type in terminal `git clone --recursive https://github.com/smalec/CIFAR10-classificator.git`. Adding `--recursive` to clone command will resolve all submodules dependencies and download required libraries.

### Setting environment
Linux users should execute `source set-env.sh` in order to set some environment variables, which are necessary to run the code.

### Approach to the problem

This solution is based on an idea of Convolutional Neural Networks. In convolutional layer, unlike in fully connected layer, we don't connect each output from previous layer with all neurons, but we only make connections between some of them. If we think of input for convolutional layer as of square of neurons, the output for each neuron will depend only on few neurons lying nearest to this neuron. Let us assume, that we want each output to be dependent only on 5x5 neighborhood. We will call this 5x5 square of weights a *filter*. What is more, this filter will be shared across all neurons in convolutional layer. The intuition behind this is that each filter is able to detect some features (e.g., vertical or horizontal edges) and we want to detect this particular feature in a whole image. So in fully connected layer we store unique set of weights and biases for each neuron, and here we've got only 25 weights (and only one overall bias) for a single filter (we can define few filters in one layer -- the output will then consist of few *feature maps*). For example: if our input is a 32x32 image, as it is in a case of CIFAR-10, a fully connected layer will need `1024*number_of_neurons` weights and `number_of_neurons` biases (actually it will require more parameters since images have 3 channels) while convolutional layer will only need `25*number_of_filters + number_of_filters` parameters.  
Convolutional layers go hand in hand with pooling layers, which are very often used just after them. Pooling is about simplifying the output of convolutional layer. One of the most popular approach to pooling is *max-pooling*. For example, for every 2x2 region in a single feature map from convolutional layer, max-pooling will output only the biggest value. The idea of pooling is based on the fact that it should be more important for us to know what is the relative location of some feature to the other ones rather than the information about exact location of feature. Also, it's worth noting that pooling layer decreases the size of feature map (e.g., 2x2 max-pooling will halve both dimensions of image), so we will need fewer paramaters to train in successive layers.

### Network model

The network, which obtained over 76% accuracy, consists of:
* 3 convolutional-pool layers
  * each convolution is run in `border_mode='same'`, which means that each feature map produced by convolution has the same resolution as the input image. Actually it's a missing feature of Theano library. This mode is obtained in this code by applying convolution in `full` mode, what makes feature maps bigger than input images, and then trimming the output to the expected size. 
  * each convolution is followed by ReLU activation function (`f(x) = max(0, x)`) before pooling
  * each layer uses 2x2 max-pooling
  * layers uses `32`, `64` and `128` 5x5 filters respectively
* affine (fully-connected) layer -- with `128*4*4` inputs, `512` outputs and ReLU activation function
* soft-max layer -- with `512` inputs and `10` outputs (each output corresponds to probability of belonging to one of classes); actually it's an affine layer with soft-max as an activation function

The network is trained with *stochastic gradient descent* algorithm. Each batch consists of 100 samples from training set which includes 40.000 images. We use 10.000 set for validation. Monitoring validation error will help us to choose best model by stopping training when validation error starts to increase (this approach is called *early stopping*). Implementation of SGD includes weight decay and momentum. The policy for learning rate is: `initial_rate * 2000/max(2000, batch_counter)`.

As the very last plot in `CIFAR10-classificator.ipynb` shows, model seems to have big overfitting but it's still able to achieve quite good accuracy within its simplicity.
