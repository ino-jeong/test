Portfolio
=

정인오 / inoh.jung@gmail.com

- MNIST with CNN implementation</h4>
    - https://github.com/ino-jeong/Portfolio/tree/master/MNIST_CNN
    - Test set accuracy : 98.39% ~ 98.67% (if number of epoch is increased)
    - 구현환경 : Python 3.5 with Tensorflow 1.1, Mac OS
    - CNN을 통한 MNIST classifier 구현 (하기 reference 참조) :
      + Tensorflow official tutorial https://www.tensorflow.org/get_started/mnist/pros
      + 'DeepLearningZeroToAll' lecture by prof. Sunghun-Kim
    http://hunkim.github.io/ml/
    - Training set : as per MNIST specification (28 X 28 pixel, grayscale)
    - Model : Convolusion Neural Network :
      - 1st layer :
        convolution with 3x3 filter, 1 channel in / 32 channel out → ReLu → Max-Pooling with 2x2 filter
      - 2nd layer :
      convolution with 3x3 filter, 32 channel in / 64 channel out → ReLu → Max-Pooling with 2x2 filter
      - 3rd later :
      Fully connected layer
