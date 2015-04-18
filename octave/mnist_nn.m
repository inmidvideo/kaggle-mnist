%% Machine Learning Online Class - Exercise 4 Neural Network Learning
%% adapted for kaggle mnist challenge

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 784;  % 28x28 Input Images of Digits
hidden_layer_size = 256;  % hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('mnist_train.mat');
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
displayData(X(sel(1:100), :));

% train/test split
ti = round(m*0.75);
X_train = X(sel(1:ti),:);
y_train = y(sel(1:ti));
X_test = X(sel(ti+1:end),:);
y_test = y(sel(ti+1:end));

fprintf('Program paused. Press enter to continue.\n');
pause;

% initialize parameters
fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

% train nn
fprintf('\nTraining Neural Network... \n')

% After you have completed the assignment, change the MaxIter to a larger
% value to see how more training helps.
options = optimset('MaxIter', 100);

% You should also try different values of lambda
lambda = 10;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_train, y_train, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

% visualize weights
fprintf('\nVisualizing Neural Network... \n')

displayData(Theta1(:, 2:end));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% predict
pred_train = predict(Theta1, Theta2, X_train);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred_train == y_train)) * 100);

pred_test = predict(Theta1, Theta2, X_test);
fprintf('\nTesting Set Accuracy: %f\n', mean(double(pred_test == y_test)) * 100);

%load('mnist_test.mat');
%prediction = predict(Theta1, Theta2, X_test);




