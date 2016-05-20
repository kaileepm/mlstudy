%% Exercise step 1

% Load data and plot
% ex1data1.txt file has two field: feature (X) and target (y)

data = load('ex1data1.txt');

% X: feature
% y: target
% m: number of training examples
X = data(:, 1);
y = data(:, 2);
m = length(y);

% plotting
% marker: point, color: black
plot(X, y, "k.");
xlabel('Population of City in 10,000s');
ylabel('Profit in $10,000s');
