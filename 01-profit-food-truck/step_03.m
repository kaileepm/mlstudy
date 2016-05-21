%% Exercise step 3
% Run gradient descent, and plot the hypothesis and cost
% Plot surface and contour

function step_03()

% Gradient descent
% data file has a pair of x1 and y
% feature: X [ x0, x1 ], x0 = 1
% target: y
% parameters (or weights): theta [ theta_0, theta_1 ]
% hypothesis: h
%   h(theta)(x) = theta_0 * x0 + theta_1 * x1
% learning rate: alpha
% convergence margin: stop training if cost function diff is less than this

% Initial value
theta_init_0 = 0;
theta_init_1 = 0;
alpha = 0.01;
converge_margin = 0.0001;

% Load data
data = load('ex1data1.txt');

y = data(:, 2);
m = length(y);

X = [ ones(m, 1), data(:, 1) ]; % m x 2
theta = [ theta_init_0; theta_init_1 ]; % 2 x 1

% Gradient descent
[theta, J_history] = MyGradientDescent(X, y, theta, alpha, converge_margin);
fprintf("theta_0: %f theta_1: %f\n", theta(1), theta(2));

% Plot of theta
figure;
hold on;
plot(data(:, 1), y, "k.");
plot(data(:, 1), X * theta, "r-");
xlabel("Population of City in 10,000s");
ylabel("Profit in $10,000s");
legend("Training data", "Linear regression");
hold off;

% Plot of cost
figure;
plot(1:numel(J_history), J_history, "r-");
xlabel("Iterations");
ylabel("Cost");

%% Visualize J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);
J_vals = zeros(length(theta0_vals), length(theta1_vals));

for i = 1:length(theta0_vals)
	for j = 1:length(theta1_vals)
		t = [theta0_vals(i); theta1_vals(j)];
		J_vals(i,j) = MyComputeCost(X, y, t);
	end
end

J_vals = J_vals';

% Surface plot
% With linear regression, only global optima exists (no local optima)
% Thus this should be a bowl shape
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0');
ylabel('\theta_1');

% Contour plot
figure;
hold on;
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0');
ylabel('\theta_1');
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
hold off;

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

function theta_new = MyGradientDescentLMSOnce(X, y, theta, alpha)
	% adjust theta
	% theta(j) = theta(j) - alpha * (1/m) * sum((errors * xj)
	% Batch gradient descent: adjust theta using entire samples

	m = length(y);
	prediction = X * theta;
	errors = (prediction - y);

	% x0 = 1
	theta_0 = theta(1) - alpha * (1 / m) * sum(errors);
	% x1
	x1 = X(:, 2);
	theta_1 = theta(2) - alpha * (1 / m) * sum(errors .* x1); % element wise multiplication

	theta_new = [ theta_0; theta_1 ];
endfunction

function [theta_new, J_history] = MyGradientDescent(X, y, theta, alpha, converge_margin)
	cost_old = MyComputeCost(X, y, theta);
	i = 1;
	while (1)
		theta = MyGradientDescentLMSOnce(X, y, theta, alpha);
		cost_new = MyComputeCost(X, y, theta);
		J_history(i) = cost_new;
		if (cost_old - cost_new < converge_margin)
			break;
		endif
		cost_old = cost_new;
		++i;
	endwhile

	theta_new = theta;
endfunction

endfunction
