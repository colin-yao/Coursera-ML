function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% -------------------------------------------------------------

%% PART 1: FEED FORWARD PASS

    % pad with column of ones
    a1 = [ones(m, 1) X];
    
    % calculate a2 and pad with column of ones
    z2 = a1 * Theta1';
    a2 = sigmoid(z2);
    a2 = [ones(size(a2, 1), 1), a2];
    
    % calculate a3
    z3 = a2 * Theta2';
    a3 = sigmoid(z3);
    
    % create matrix of y vectors as rows
    y_vectors = zeros(m, num_labels);
    for i = 1:m
        y_vectors(i, y(i)) = 1;
    end
    
    % compute cost J
    cost_summation_K = sum((-y_vectors .* log(a3)) - ((1 - y_vectors) .* log (1 - a3)), 1);
    cost_summation_m = sum(cost_summation_K, 2);
    J = 1/m * cost_summation_m;
    
    % add regularization term
    num_cols_Theta1 = size(Theta1, 2);
    num_cols_Theta2 = size(Theta2, 2);
    regularization_term_Theta1 = sum(sum(Theta1(:, 2:num_cols_Theta1) .^ 2, 2), 1);
    regularization_term_Theta2 = sum(sum(Theta2(:, 2:num_cols_Theta2) .^ 2, 2), 1);
    
    J = J + lambda/(2*m) * (regularization_term_Theta1 + regularization_term_Theta2);
    
%% PART 2: BACKPROPAGATION

    for i = 1:m
        % set input layer values
        a1 = [1; X(i, :)'];
        
        % calculate a2 and a3
        z2 = Theta1 * a1;
        a2 = sigmoid(z2);
        a2 = [1; a2];
        
        z3 = Theta2 * a2;
        a3 = sigmoid(z3);
        
        % calculate errors and accumulated gradients
        delta_output = a3 - y_vectors(i, :)';
        delta_2 = (Theta2' * delta_output) .* [1; sigmoidGradient(z2)];
        delta_2 = delta_2(2: end);
        
        Theta1_grad = Theta1_grad + delta_2 * a1';
        Theta2_grad = Theta2_grad + delta_output * a2';
        
    end
    
%% PART 3: UNREGULARIZED GRADIENT

    % strip bias unit and add padding
    regularization_Theta1 = [zeros(size(Theta1, 1), 1), Theta1(:, 2:end)];
    regularization_Theta2 = [zeros(size(Theta2, 1), 1), Theta2(:, 2:end)];

    Theta1_grad = 1/m * Theta1_grad + lambda/m * regularization_Theta1;
    Theta2_grad = 1/m * Theta2_grad + lambda/m * regularization_Theta2;

%% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
