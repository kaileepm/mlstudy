%% Exercise step 2
% Run gradient descent, and plot the hypothesis and cost
% gradient descent algorithm: LMS (Least Mean Squares)

function step_02()

% Gradient descent
% data file has a pair of x1 and y
% feature (matrix): X [ x0, x1 ], x0 = 1
% target (vector): y
% parameters (or weights): theta [ theta_0, theta_1 ]
% hypothesis: h
%   h(theta)(x) = theta_0 * x0 + theta_1 * x1
% learning rate: alpha

% Initial value
theta_init_0 = 0;
theta_init_1 = 0;
alpha = 0.01;

iter_1 = 100;
iter_2 = 900;
iter_3 = 4000;

% Load data
data = load('ex1data1.txt');

y = data(:, 2);
m = length(y);

X = [ ones(m, 1), data(:, 1) ]; % m x 2
theta = [ theta_init_0; theta_init_1 ]; % 2 x 1

% Gradient descent
[theta_hist_1, J_history_1] = MyGradientDescent(X, y, theta, alpha, iter_1);
[theta_hist_2, J_history_2] = MyGradientDescent(X, y, theta_hist_1, alpha, iter_2);
[theta_hist_3, J_history_3] = MyGradientDescent(X, y, theta_hist_2, alpha, iter_3);
theta = theta_hist_3;

% Print the result
fprintf("parameters: %f %f\n", theta(1), theta(2));
fprintf("cost: %f\n", MyComputeCost(X, y, theta));

% Plot of theta
figure;
hold on;
plot(data(:, 1), y, "k.");
plot(data(:, 1), X * theta_hist_1, "r-");
plot(data(:, 1), X * theta_hist_2, "g-");
plot(data(:, 1), X * theta_hist_3, "b-");
xlabel("Population of City in 10,000s");
ylabel("Profit in $10,000s");
legend("Training data", "First", "Second", "Third");
hold off;

% Plot of cost
figure;
hold on;
plot(1:iter_1, J_history_1, "r-");
plot(iter_1 + 1:iter_1 + iter_2, J_history_2, "g-");
plot(iter_1 + iter_2 + 1:iter_1 + iter_2 + iter_3, J_history_3, "b-");
xlabel("Iterations");
ylabel("Cost");
legend("First", "Second", "Third");
hold off;

function J = MyComputeCost(X, y, theta)
	% J(theta) = (1/2m) * sum((h(theta)(x(i)) - y(i)) ^ 2)
	% sum of (prediction - target) ^ 2
	% h(theta)(x) = theta * X
	m = length(y);
	% X: m x 2 matrix, theta: 2 x 1 vector
	% X * theta: m dimensional vector
	prediction = X * theta;
	% errors: m dimensional vector
	errors = (prediction - y);
	sqrErrors = errors .^ 2; % element-wise square
	sumSqrErrors = sum(sqrErrors);

	J = 1 / (2 * m) * sumSqrErrors;
endfunction

function theta_new = MyGradientDescentLMSOnce(X, y, theta, alpha)
	% adjust theta
	% theta(j) = theta(j) - alpha * (1/m) * sum((errors * xj)

	m = length(y);
	prediction = X * theta; % m x 1
	errors = (prediction - y); % m x 1

	% x0 = 1
	theta_0 = theta(1) - alpha * (1 / m) * sum(errors);
	% x1
	x1 = X(:, 2);
	theta_1 = theta(2) - alpha * (1 / m) * sum(errors .* x1); % element wise multiplication

	theta_new = [ theta_0; theta_1 ];
endfunction

function [theta_new, J_history] = MyGradientDescent(X, y, theta, alpha, iterations)
	for i = 1:iterations
		theta = MyGradientDescentLMSOnce(X, y, theta, alpha);
		J_history(i) = MyComputeCost(X, y, theta);
		% fprintf('iteration %d: theta = (%f, %f), cost = %f\n', i, theta(1), theta(2), J_history(i));
	end

	theta_new = theta;
endfunction

endfunction
