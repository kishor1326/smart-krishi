%% Uzhavan AI - MATLAB Crop Health Engine (final-round module)
% Run section by section. Requires: Image Processing, Deep Learning, Statistics & ML Toolboxes
% + "Deep Learning Toolbox Model for ResNet-18 Network" support package (Add-On Explorer).
addpath(fileparts(mfilename('fullpath')));
%% 1. Train (put tomato class folders in dataset/tomato first)
trainModel;
%% 2. Evaluate on held-out split (accuracy, P/R/F1, confusion matrix, latency)
M = evaluateModel;
%% 3. Single image (writes results/panel.png)
res = predictDisease(fullfile(fileparts(fileparts(mfilename('fullpath'))),'samples','leaf1.jpg'));
disp(res); imshow(fullfile(uzhavanConfig().resultsDir,'panel.png'));
%% 4. Batch (use YOUR OWN phone photos to report real-world behaviour)
% T = batchPredict(fullfile(uzhavanConfig().root,'samples'));
%% 5. Irrigation what-if
simulateIrrigation;                              % baseline
simulateIrrigation('irrigMult',0.8);             % what if irrigation -20%
simulateIrrigation('rainMult',1.5);              % what if rainfall +50%
