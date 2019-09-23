function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
c_s = zeros(8, 8);

for i = 1:size(C_vec, 2)
  for j = 1:size(sigma_vec, 2)
    model = svmTrain(X, y, C_vec(1, i), @(x1, x2) gaussianKernel(x1, x2, sigma_vec(1, j)));
    predictions = svmPredict(model, Xval);
    c_s(i, j) = mean(double(predictions ~= yval));
    fprintf(["C = %f, sigma = %f"], C_vec(1, i), sigma_vec(1, j));
    visualizeBoundary(X, y, model);
  endfor
endfor

c_s

[min_row, C_indexs] = min(c_s);
[mins, sigma_index] = min(min_row);
C = C_vec(1, C_indexs(sigma_index));
sigma = sigma_vec(1, sigma_index);





% =========================================================================

end
