Portfolio
=

정인오 / inoh.jung@gmail.com / 010-9907-9386




1. OCR implementation, multi-class classification (Coursera)
    - https://github.com/ino-jeong/Portfolio/tree/master/OCR(multiclass_classification)
    - Test set accuracy : 94.86%
    - Octave(추천) 또는 Matlab에서 main.m 실행
    - 구현환경 : GNU Octave 3.8, Mac OS
    - Coursera Machine Learning 과정 구현 과제
    - Training set : 20 X 20 pixel, grayscale, 5000 examples of handwritten digits
    - Model : Multi-class classification
    - Cost function 및 Training / Prediction 과정 구현 :
        + lrCostFunction.m
        + oneVsAll.m
        + predictOneVsAll.m



2. OCR implementation, neural-net (Coursera)

    - https://github.com/ino-jeong/Portfolio/tree/master/OCR(neural_net)
    - Test set accuracy : 95~96% (up to random initialization)
    - Octave(추천) 또는 Matlab에서 main.m 실행
    - 구현환경 : GNU Octave 3.8, Mac OS
    - Coursera Machine Learning 과정 구현 과제
    - Training set : 20 X 20 pixel, grayscale, 5000 examples of handwritten digits (1번과 동일 set)
    - Model : Neural Net, 3 layer (1 hidden layer)
    - Layer 구성 및 backpropagation 구현 :
        + sigmoidGradient.m
        + nnCostFunction.m


3. K-means clustering (Coursera)

    - https://github.com/ino-jeong/Portfolio/tree/master/k_means(K_means)
    - Octave(추천) 또는 Matlab에서 main.m 실행
    - Basic k-means clustering implementation (left : iteration 1 / right : after 8 iteration)
    - 구현환경 : GNU Octave 3.8, Mac OS
    - Coursera Machine Learning 과정 구현 과제
    - Model : K-means
    - K-means clustering algorithm 구현 :
        + computeCentroids.m
        + findClosestCentroids.m
        + kMeansInitCentroids.m
