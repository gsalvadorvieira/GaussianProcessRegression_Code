clear; close all; startup;

%% Position points where measurements happens
train.t = (1:7)';
% train.t = [1,2,4.5,5,6.2,8]';
% train.t = [1 4 5 7 9];

train.t = reshape(train.t,[length(train.t),1]); % force it to be a colum vector

%% Define true function that generates data
% train.y = exp(-train.t);
train.y = sin(train.t);
% train.y = -(train.t-10).*train.t;


%% Set amplitude of Gaussian noise

% sigmaN = eps; % no measurement noise
% sigmaN = .05;
% sigmaN = .1;
sigmaN = max(abs(train.y(:)))*.05; % relative error

sigmaN2 = sigmaN^2;

% train.err = sigmaN*randn(size(train.t));
train.err = sigmaN*ones(size(train.t));

figure
errorbar(train.t, train.y, train.err,'ok', 'LineWidth', 2, 'MarkerSize',10)
xlabel('$t$'); ylabel('$y$');
xlim([0 10])

%% Fit GP model

% kernel function: squared exponential with sigmaL = theta(1), sigmaF =
% theta(2) + Noise based on number

% kfcn = @(XN,XM,theta) theta(2)^2*exp(-0.5*(pdist2(XN,XM).^2)/(theta(1)^2)) + sigmaN2* (pdist2(XN,XM) == 0);
kfcn = @(XN,XM,theta) theta(2)^2*exp(-0.5*(pdist2(XN,XM).^2)/(theta(1)^2));


theta0 = [1,1]; % initialize hyperparameters
% gprMdl = fitrgp(train.t, train.y, 'KernelFunction', kfcn, 'KernelParameters', theta0, ...
%   'Basis','constant','FitMethod','exact',...
%   'PredictMethod','exact',...
%   'Sigma', 1e-16, 'ConstantSigma', true, ...
%   'SigmaLowerBound', 1e-16);
gprMdl = fitrgp(train.t, train.y, 'KernelFunction', kfcn, 'KernelParameters', theta0, ...
  'Sigma', sigmaN, 'ConstantSigma', true, ...
  'SigmaLowerBound', 1e-16);

% gprMdl = fitrgp(train.t, train.y, 'KernelFunction', kfcn, 'KernelParameters', theta0, ...
%   'SigmaLowerBound', 1e-16);

fprintf('Optimal sigmaL = %g\n', gprMdl.KernelInformation.KernelParameters(1))
fprintf('Optimal sigmaF = %g\n', gprMdl.KernelInformation.KernelParameters(2))

%% Make predictions for uniform grid of test points and get standard deviation

test.t = linspace(0,10, 201);
test.t = reshape(test.t,[length(test.t),1]); % colum vector always

[test.ypred, test.ysd] = predict(gprMdl, test.t);


%% Plot prediction and uncertainty (2*sigma) (filling area in between)

figure

curve1 = test.ypred - test.ysd;
curve2 = test.ypred + test.ysd;
plot(test.t, curve1, '-r', 'LineWidth', 1.);
hold on;
plot(test.t, curve2, '-r', 'LineWidth', 1.);
S = patch([test.t; flipud(test.t)], [curve1; flipud(curve2)], 'r');
S.FaceAlpha = 0.25;
S.LineStyle = 'none';

plot(test.t, test.ypred, '-r', 'LineWidth',2.5)
xlabel('$t$'); ylabel('$y$');
errorbar(train.t, train.y, train.err,'ok', 'LineWidth', 2, 'MarkerSize',10)
xlabel('$t$'); ylabel('$y$');
box on
