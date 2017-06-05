% This is Machine Learning Online Class from Coursera, Exercise 7
%
% codes implemented by applicant as assignment of online course are :
%
%     computeCentroids.m
%     findClosestCentroids.m
%     kMeansInitCentroids.m
%
% findClosestCentroids computes the centroid memberships for every example.
% and returns the closest centroids as idx.


function idx = findClosestCentroids(X, centroids)

% set some variables
K = size(centroids, 1);
m = size(X,1);
idx = zeros(m, 1);

% ====================== YOUR CODE HERE (Coursera) ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.

for i = 1:m
    % for each example, simultaneously calculate (x - u_j)^2, where j = 1...K for k centroids
    % and check out which centroids is closest one from 1...kth centroids

    [min_value,min_idx] = min(sum(((X(i,:)-centroids).^2),2));
    idx(i) = min_idx(1);
end
% =============================================================

end

