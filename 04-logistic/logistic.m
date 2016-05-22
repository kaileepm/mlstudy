%% Machine Learning Class Exercise 2
%% Logistic Regression (Classification)

function logistic()

%%	
%% Load data
%%

% data file has three field: x1, x2, y
data = load('ex2data1.txt');
X = data(:, 1:2);
y = data(:, 3);

% find indices of positive and negative samples
pos_ids = find(y == 1);
neg_ids = find(y == 0);

%%
%% Logistic regression
%% Use gradient descent
%%

% decision boundary
% theta_0 + theta_1 * x1 + theta_2 * x2 + ...

[ m, n ] = size(X);
X = [ ones(m, 1) X ];
theta = zeros(n + 1, 1);

[cost, grad] = MyComputeCost(theta, X, y);
fprintf('Cost at initial theta: %f\n', cost);
fprintf('Gradient at initial theta: %f, %f, %f\n', grad(1), grad(2), grad(3));

% Use advanced algorithm to solve gradient descent
% fminunc()

options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = fminunc(@(t)(MyComputeCost(t, X, y)), theta, options);
fprintf('Cost at final theta: %f\n', cost);
fprintf('Final theta: %f %f %f\n', theta(1), theta(2), theta(3));

% plot positive and negative samples
figure;
hold on;
plot(X(pos_ids, 2), X(pos_ids, 3), "bo");
plot(X(neg_ids, 2), X(neg_ids, 3), "kx");
xlabel("Exam 1");
ylabel("Exam 2");

% plot the decision boundary
plot_x = [min(X(:,2)) - 2,  max(X(:,2)) + 2];
plot_y = (-1 ./ theta(3)) .* (theta(2) .* plot_x + theta(1));
plot(plot_x, plot_y, "r-")
legend("Positive", "Negative", "Decision boundary");
hold off;

%%
%% Evaluate
%%

predict_right_count = 0;
h = sigmoid(X * theta);

for i = 1:m
	if (h(i) > 0.5 && y(i) == 1) || (h(i) < 0.5 && y(i) == 0)
		++predict_right_count;
	endif
endfor

fprintf('Got %d right out of %d : %2.f percent\n', predict_right_count, m, predict_right_count * 100 / m);

% hypothesis
% X_i_t: column vector of features
% theta_t: row vector of theta (transpose of theta vector)
function h = hypothesis(X_i_t, theta_t)
	h = sigmoid(theta_t * X_i_t);
endfunction

% sigmoid
% g(z) = 1 / (1 + e^-z)
function ret = sigmoid(z)
	ret = 1.0 ./ (1.0 + exp(-z));
endfunction

% cost function
function [J, grad] = MyComputeCost(theta, X, y)
	m = length(y);
	% h_theta-x: hypothesis in vector format
	h_theta_x = sigmoid(X * theta);
	% cost J
	J = (1 / m) * (-y' * log(h_theta_x) - (1 - y') * log(1 - h_theta_x));
	% gradient
	grad = (1 / m) * (X' * (h_theta_x - y));
endfunction

endfunction
