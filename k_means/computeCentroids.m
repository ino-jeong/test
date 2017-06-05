% This is Machine Learning Online Class from Coursera, Exercise 7
%
% codes implemented by applicant as assignment of online course are :
%
%     computeCentroids.m
%     findClosestCentroids.m
%     kMeansInitCentroids.m
%
% computeCentroids() returs the new centroids by computing the means of the
% data points assigned to each centroid.


function centroids = computeCentroids(X, idx, K)

% Useful variables
[m n] = size(X);
centroids = zeros(K, n);


% ====================== YOUR CODE HERE (Coursera) ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.

for i = 1:K

    mat_k = (idx==i);   % mask matrix. for each example x, mat_k has value 1 if its centroid is 'i'
    m_k = sum(mat_k);   % get total number of examples which has centroid as 'i'
    centroids(i,:) = sum((X .* mat_k), 1) / m_k;     % calculate mean of ith centroid

end
% =============================================================

end

