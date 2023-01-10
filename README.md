# A Shallow CNN for COVID-19 Detection using Preprocessed CXR Images
It is a convolutional neural network which classifies chest x-rays images as if they are COVID-19 positive or not. Chest x-ray images firstly preprocessed with a max-min normalizator and contrast limited adaptive histogram equalizator, and then fed into the network. The accuracy results on the test dataset with 128x128 images are 94.63%.

The convolutional neural network that I've created follows the popular pattern of doubling filter size:
- 2D Convolutional Layer with 32 Filters and 3x3 Kernel (No Padding)
- Max Pooling Layer with 2x2 Kernel, with 2x2 Stride.
- Dropout with 10% probability.
- 2D Convolutional Layer with 64 Filters and 3x3 Kernel (No Padding)
- Max Pooling Layer with 2x2 Kernel, with 2x2 Stride.
- Dropout with 10% probability.
- 2D Convolutional Layer with 128 Filters and 3x3 Kernel (No Padding)
- Max Pooling Layer with 2x2 Kernel, with 2x2 Stride.
- Dense layer with 128 neurons and ReLU activation.
- Dense layer with 10 neurons and ReLU activation.
- Dense layer with 1 neuron and sigmoid activation.
