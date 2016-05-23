%% Machine Learning Class Exercise 2
%% Logistic Regression (Classification)

function logistic_cost_hist()

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
cost_hist(1) = cost;
cost_old = cost;
iter = 1;
iter_per_fminunc = 40;
converge_threshold = 0.00001;

options = optimset('GradObj', 'on', 'MaxIter', iter_per_fminunc);
[theta, cost] = fminunc(@(t)(MyComputeCost(t, X, y)), theta, options);

while (1)
	++iter;
	cost_hist(iter) = cost;
	if abs(cost_old - cost) < converge_threshold
		break;
	endif

	cost_old = cost;
	[theta, cost] = fminunc(@(t)(MyComputeCost(t, X, y)), theta, options);
endwhile

fprintf('Iterations: %d\n', iter);
fprintf('Final theta: %f %f %f\n', theta(1), theta(2), theta(3));

% plot the learning curve
figure;
plot(linspace(1, iter * iter_per_fminunc, iter), cost_hist, "r-");
xlabel("Iterations");
ylabel("Cost");
legend('Learning curve');

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
