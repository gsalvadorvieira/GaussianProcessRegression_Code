clear; close all

% Mesh domain
[x1G,x2G] = meshgrid(linspace(0,10,101),linspace(0,5,51));

%% Define function
fG = sin(x1G)+sin(x2G);
% fG = (x1G/2).^2;
% fG = sin(x1G)-sin(x2G);
% fG = x1G.^2-x2G.^3;
% fG = sin(x1G).*sin(x2G);

%% Noisy parameters in the measurements
% sigman = eps;
% sigman = .25;
sigman = .05;
% sigman = max(abs(fG(:)))*.05; % relative error

sigman2 = sigman^2;

%% Plot function

figure('Position',[270 535 1320 420]);

ax11 = subplot(1,2,1);

surf(x1G,x2G,fG)
hold on
shg

%% Select training positions

%%% Select random points with noise
% Ntrain = 50;
% ind = randsample(length(fG(:)), Ntrain);

%%% Select uniformly
ind = find(abs(rem(x1G(:),1))<0.01 & abs(rem(x2G(:),1))<0.01);
Ntrain = length(ind);

train.x1 = x1G(ind);
train.x2 = x2G(ind);
train.fG = fG(ind) + sigman * randn(size(fG(ind)));

plot3(train.x1,train.x2,train.fG(:),'*r')
shg

%% Fit regression model using all training points. Specify kernel function and parameters

% kernel function: squared exponential with sigmaL = theta(1), sigmaF =
% theta(2) + Gaussian Noise

kfcn_withNoise = @(XN,XM,theta) theta(2)^2*exp(-0.5*(pdist2(XN,XM).^2)/(theta(1)^2)) + sigman2* (pdist2(XN,XM) == 0);
kfcn = @(XN,XM,theta) theta(2)^2*exp(-0.5*(pdist2(XN,XM).^2)/(theta(1)^2));

Ntest = 1000;

ind_random = randsample(length(fG(:)), Ntest);

theta0 = [1,1]; % initialize hyperparameters
gprMdl = fitrgp([train.x1, train.x2], train.fG, 'KernelFunction', kfcn, 'KernelParameters', theta0, ...
      'Sigma', sigman, 'ConstantSigma', true, ...
        'SigmaLowerBound', 1e-16);

fprintf('Optimal sigmaL = %g\n', gprMdl.KernelInformation.KernelParameters(1))
fprintf('Optimal sigmaF = %g\n', gprMdl.KernelInformation.KernelParameters(2))

test.x1 = x1G(ind_random);
test.x2 = x2G(ind_random);

[test.fGpred,test.fGsd] = predict(gprMdl, [test.x1, test.x2]);

plot3(test.x1, test.x2, test.fGpred, 'ob')
gca.Title = 'Using all possible sensors';
shg
colormap(ax11,'gray')

fprintf('Average error using all the sensors = %g\n', norm(test.fGpred-fG(ind_random))/Ntest)

%% Plot variance

ax12 = subplot(1,2,2);
scatter(test.x1, test.x2, [], test.fGsd.^2, '.'); box on;
colormap(ax12,'jet'); colorbar;

%% Optimize points to ask for

k = 15; % nb. of sensors to be employed

Selectible = 1:length(train.fG);
Selected(1) = datasample(Selectible,1);
Selectible = setdiff(Selectible, Selected);

figure('Position',[677 133 560 316])

plot(train.x1,train.x2,'.k'); shg;
hold on
xlim([0,10])
ylim([0,5])

plot(train.x1(Selected),train.x2(Selected),'or','markersize',9);
title('Sensor Distribution');
hold on
% pause

%%

for j=2:k
    
    dy = -Inf;
    
    SigmaAA = kfcn_withNoise(train.fG(Selected), train.fG(Selected), gprMdl.KernelInformation.KernelParameters);
    invSigmaAA = inv(SigmaAA);
    SigmaAbAb = kfcn_withNoise(train.fG(Selectible), train.fG(Selectible), gprMdl.KernelInformation.KernelParameters);
    invSigmaAbAb = inv(SigmaAbAb);
    
    for i = Selectible
                
        sigmay2 = kfcn_withNoise(train.fG(i), train.fG(i), gprMdl.KernelInformation.KernelParameters);
        SigmayA = kfcn_withNoise(train.fG(i), train.fG(Selected), gprMdl.KernelInformation.KernelParameters);        
        SigmaAy = SigmayA';
        SigmayAb = kfcn_withNoise(train.fG(i), train.fG(Selectible), gprMdl.KernelInformation.KernelParameters); 
        SigmaAby = SigmayAb';
        
        %% Mutual Information (MI) Criterium
%         dy_new = (sigmay2 - SigmayA*invSigmaAA*SigmaAy) / ...
%                     (sigmay2 - SigmayAb*invSigmaAbAb*SigmaAby);
        %% Entropy Criterium
        dy_new = sigmay2 - SigmayA*invSigmaAA*SigmaAy;
        
        if dy_new > dy
            dy = dy_new;
            iMax = i;
        end
        
    end
    
    Selectible = setdiff(Selectible, iMax);
    Selected(j) = iMax;
    
%     plot(train.x1(Selected(j)),train.x2(Selected(j)),'or','markersize',9);
%     shg
%     pause
    
end

plot(train.x1(Selected),train.x2(Selected),'or','markersize',9);
shg

g = sprintf('%d ', Selected);
fprintf('Selected list = [%s]\n', g)


%% Get distribution only from selected data

gprMdl_final = fitrgp([train.x1(Selected), train.x2(Selected)], train.fG(Selected), 'KernelFunction', kfcn, 'KernelParameters', gprMdl.KernelInformation.KernelParameters, ...
  'Sigma', sigman, 'ConstantSigma', true, ...
        'SigmaLowerBound', 1e-16);


fprintf('New Optimal sigmaL = %g\n', gprMdl_final.KernelInformation.KernelParameters(1))
fprintf('New Optimal sigmaF = %g\n', gprMdl_final.KernelInformation.KernelParameters(2))

% Evaluate in test data
[test.fGpred_final, test.fGsd_final] = predict(gprMdl_final, [test.x1, test.x2]);

figure('Position',[270 535 1320 420]);

ax21 = subplot(1,2,1);

surf(x1G,x2G,fG)
hold on
plot3(train.x1(Selected), train.x2(Selected), train.fG(Selected), '*r');
plot3(test.x1, test.x2, test.fGpred_final, 'ob')
gca.Title = 'Using only selected sensors';
shg
colormap(ax21,'gray')

fprintf('Average error using only selected sensors = %g\n', norm(test.fGpred_final-fG(ind_random))/Ntest)

%% Plot variance of final model

ax22 = subplot(1,2,2);
scatter(test.x1, test.x2, [], test.fGsd_final, '.'); box on;
colormap(ax22,'jet'); colorbar;
title('Standard Deviation')
