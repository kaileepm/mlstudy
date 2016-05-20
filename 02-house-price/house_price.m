%% Exercise 2

function house_price()

% Initial value
theta_zero = 0;
theta_one = 0;
theta_two = 0;
alpha = 0.01;
converge_margin = 0.0001;

% Load data
data = load('ex1data2.txt');

X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Plot the original data
figure;
plot3(data(:, 1), data(:, 2), y, "k.");
xlabel('Area');
ylabel('Number of rooms');
zlabel('Price of the house');
grid;

% Normalize the features
% - Subtract the mean value of each feature from the dataset.
% - After subtracting the mean, additionally scale (divide) the feature values
%   by their respective “standard deviations.”
% After normalization, mean value becomes 0, and the standard deviation becomes 1.

% size(object, dim) : number of rows
% mu: mean value
% sigma: standard devication	

fprintf('Normalizing the features...\n');

mu = mean(X);
sigma = std(X);

for i = 1:size(X, 2)
	X_norm(:, i) = (X(:, i) - mu(i)) / sigma(i);
	fprintf('Feature %d: mean %f -> %f, std %f -> %f\n', i, mu(i), mean(X_norm(:, i)), sigma(i), std(X_norm(:, i)));
end

% Plot the normalized data
figure;
plot3(X_norm(:, 1), X_norm(:, 2), y, "k.");
xlabel('Normalized Area');
ylabel('Normalized Number of rooms');
zlabel('Price of the house');
grid;

% Prepare for gradient descent
fprintf('Running gradient descent...\n');
X_norm = [ ones(m, 1) X_norm ];
theta = [ theta_zero; theta_one; theta_two ]; % 3 x 1

% Gradient descent
[theta, J_history] = MyGradientDescentMulti(X_norm, y, theta, alpha, converge_margin);

% Plot of cost
figure;
plot(1:numel(J_history), J_history, "r-");
xlabel("Iterations");
ylabel("Cost");

% Prediction
fprintf('Predicting...\n');
MyPredictPrint(1650, 3, theta, mu, sigma);
MyPredictPrint(1000, 2, theta, mu, sigma);
MyPredictPrint(2000, 4, theta, mu, sigma);

% Plot
x1_space = linspace(0, 5000, 100);
x2_space = linspace(0, 5, 100);
z_value = zeros(size(x1_space, 2), size(x2_space, 2));

for i = 1:size(x1_space, 2)
	for j = 1:size(x2_space, 2)
		z_value(i, j) = MyPredict(x1_space(i), x2_space(j), theta, mu, sigma);
	end
end

figure;
mesh(x1_space, x2_space, z_value);

function J = MyComputeCost(X, y, theta)
	% J(theta) = (1/2m) * sum((h(theta)(x(i)) - y(i)) ^ 2)
	% sum of (prediction - target) ^ 2
	% h(theta)(x) = theta * X
	m = length(y);
	prediction = X * theta;
	errors = (prediction - y);
	sqrErrors = errors .^ 2; % element-wise square
	sumSqrErrors = sum(sqrErrors);

	J = 1 / (2 * m) * sumSqrErrors;
endfunction

function theta_new = MyGradientDescentMultiLMSOnce(X, y, theta, alpha)
	% adjust theta
	% theta(j) = theta(j) - alpha * (1/m) * sum((errors * xj)

	m = length(y);
	prediction = X * theta;
	errors = (prediction - y);

	% x0 = 1
	theta_zero = theta(1) - alpha * (1 / m) * sum(errors);
	% x1
	x1 = X(:, 2);
	theta_one = theta(2) - alpha * (1 / m) * sum(errors .* x1); % element wise multiplication
	% x2
	x2 = X(:, 3);
	theta_two = theta(3) - alpha * (1 / m) * sum(errors .* x2); % element wise multiplication

	theta_new = [ theta_zero; theta_one; theta_two ];
endfunction

function [theta_new, J_history] = MyGradientDescentMulti(X, y, theta, alpha, converge_margin)
	cost_old = MyComputeCost(X, y, theta);
	i = 1;
	while (1)
		theta = MyGradientDescentMultiLMSOnce(X, y, theta, alpha);
		cost_new = MyComputeCost(X, y, theta);
		J_history(i) = cost_new;
		if (cost_old - cost_new < converge_margin)
			break;
		endif
		cost_old = cost_new;
		% fprintf('theta: %f %f %f, cost: %f\n', theta(1), theta(2), theta(3), cost_new);
		++i;
	endwhile

	theta_new = theta;
endfunction

function price = MyPredict(x1, x2, theta, mu, sigma)
	% normalize
	sample = [ 1, (x1 - mu(1)) / sigma(1), (x2 - mu(2)) / sigma(2) ];
	% calculate
	price = sample * theta;
endfunction

function MyPredictPrint(x1, x2, theta, mu, sigma)
	price = MyPredict(x1, x2, theta, mu, sigma);
	fprintf('Prediction for (%d, %d): %f\n', x1, x2, price);
endfunction

endfunction
