function cfg = uzhavanConfig()
% Central settings. Thresholds marked TUNE must be calibrated on YOUR phone photos.
cfg.root       = fileparts(fileparts(mfilename('fullpath')));
cfg.datasetDir = fullfile(cfg.root,'dataset','tomato');   % one sub-folder per class
cfg.modelFile  = fullfile(cfg.root,'models','uzhavanNet.mat');
cfg.resultsDir = fullfile(cfg.root,'results');
cfg.inputSize  = [224 224 3];
cfg.backbone   = "resnet18";      % or "mobilenetv2" (smaller/faster). Needs the support package.
cfg.maxImagesPerClass = 350;      % subsample => fast training; raise if you have a GPU
cfg.epochs     = 4;
cfg.miniBatch  = 32;
cfg.learningRate = 1e-4;
cfg.validationFrequency = 30;
cfg.seed       = 42;
cfg.minConfidence = 0.70;         % below this => "low confidence", no diagnosis
cfg.minMargin     = 0.20;         % top1 - top2 must exceed this
cfg.blurThreshold = 40;           % TUNE: Laplacian variance
cfg.minBrightness = 0.15;
cfg.maxBrightness = 0.92;
cfg.minLeafFraction = 0.10;
end
