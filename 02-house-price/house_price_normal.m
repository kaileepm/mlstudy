%% Exercise 2 using normal equation

function house_price_normal()

% Load data
data = load('ex1data2.txt');

y = data(:, 3);
m = length(y);
X = data(:, 1:2);

% Feature scaling
mu = mean(X);
sigma = std(X);

for i = 1:size(X, 2)
	X_norm(:, i) = (X(:, i) - mu(i)) / sigma(i);
end

X = [ ones(m, 1) X_norm ];

% Normal equation
theta = (pinv(X' * X)) * (X') * y;
fprintf('With feature scaling\n');
fprintf('theta: %f %f %f\n', theta(1), theta(2), theta(3));

% Prediction
MyPredictPrint(1650, 3, theta, mu, sigma);
MyPredictPrint(1000, 2, theta, mu, sigma);
MyPredictPrint(2000, 4, theta, mu, sigma);

% Normal equation without feature scaling
X = [ ones(m, 1) data(:, 1:2) ];
theta = (pinv(X' * X)) * (X') * y;
fprintf('Without feature scaling\n');
fprintf('theta: %f %f %f\n', theta(1), theta(2), theta(3));
MyPredictPrintNoScaling(1650, 3, theta);
MyPredictPrintNoScaling(1000, 2, theta);
MyPredictPrintNoScaling(2000, 4, theta);

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

function price = MyPredictNoScaling(x1, x2, theta)
	price = [ 1 x1 x2 ] * theta;
endfunction

function MyPredictPrintNoScaling(x1, x2, theta, mu, sigma)
	price = MyPredictNoScaling(x1, x2, theta);
	fprintf('Prediction for (%d, %d): %f\n', x1, x2, price);
endfunction

endfunction
