clear; close all;

%% Generate data

% train.t = (1:6)';
% train.t = [1,2,4.5,5,6.2,8]';
train.t = [1 4 5 9];

train.t = reshape(train.t,[length(train.t),1]); % colum vector always

%%% Set function that generates data

train.y = exp(-train.t);
% train.y = sin(train.t);

%%% Set noise
sigmaN = eps;
% sigmaN = .1;
% sigmaN = .05;
sigmaN2 = sigmaN^2;

% train.err = sigmaN*randn(size(train.t));
train.err = sigmaN*ones(size(train.t));

figure
errorbar(train.t,train.y,train.err,'ob')
hold on

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


fprintf('Optimal sigmaL = %g\n', gprMdl.KernelInformation.KernelParameters(1))
fprintf('Optimal sigmaF = %g\n', gprMdl.KernelInformation.KernelParameters(2))

%% Make predictions and plot confidence interval

test.t = linspace(0,10, 101);
  
test.t = reshape(test.t,[length(test.t),1]); % colum vector always

[test.ypred, test.ysd] = predict(gprMdl, test.t);


%%% fancy plot filling area in between

curve1 = test.ypred - test.ysd;
curve2 = test.ypred + test.ysd;
plot(test.t, curve1, 'r', 'LineWidth', 2);
hold on;
plot(test.t, curve2, 'r', 'LineWidth', 2);
S = patch([test.t; flipud(test.t)], [curve1; flipud(curve2)], 'r');
S.FaceAlpha = 0.25;

% errorbar(test.t,test.ypred,test.ysd,'-r')
errorbar(train.t,train.y,train.err,'ob','LineWidth',2)

