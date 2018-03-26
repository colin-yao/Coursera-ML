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

% using values suggested in assignment sheet
steps = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% init final resulting errors for 8 * 8 steps
error_results = eye(64, 3);
iter_counter = 1;

% test combinations for corresponding error value
for test_C = steps
    for test_sigma = steps
        model = svmTrain (X, y, test_C, @(x1, x2) gaussianKernel(x1, x2, test_sigma));
        prediction = svmPredict(model, Xval);
        error = mean(double(prediction ~= yval));
        error_results(iter_counter, :) = [error, test_C, test_sigma];
        
        iter_counter = iter_counter + 1;
    end
end

% return optimal C and sigma based on lowest error
sorted_error_results = sortrows(error_results, 1);
C = sorted_error_results(1, 2);
sigma = sorted_error_results(1, 3);

% =========================================================================

end
