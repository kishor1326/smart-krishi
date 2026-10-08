function trainModel()
% Uzhavan AI - robust R2024a+ training pipeline
% Uses augmentedImageDatastore + trainnet + imagePretrainedNetwork.
cfg = uzhavanConfig();
rng(cfg.seed);

if ~isfolder(cfg.datasetDir)
    error("Dataset folder not found: %s", cfg.datasetDir);
end

imds = imageDatastore(cfg.datasetDir, ...
    IncludeSubfolders=true, LabelSource="foldernames");

if numel(imds.Files) < 10
    error("Dataset is empty or too small. Put the tomato class folders inside: %s", cfg.datasetDir);
end

tbl = countEachLabel(imds);
disp(tbl);

if height(tbl) < 2
    error("At least 2 class folders are required. Found %d class.", height(tbl));
end

n = min(cfg.maxImagesPerClass, min(tbl.Count));
imds = splitEachLabel(imds, n, randomized=true);

[imdsTr, imdsVa, imdsTe] = splitEachLabel(imds, 0.70, 0.15, randomized=true);
classes = categories(imdsTr.Labels);

if ~isfolder(cfg.resultsDir), mkdir(cfg.resultsDir); end
if ~isfolder(fileparts(cfg.modelFile)), mkdir(fileparts(cfg.modelFile)); end

testFiles = imdsTe.Files;
testLabels = imdsTe.Labels;
save(fullfile(cfg.resultsDir,"testSplit.mat"),"testFiles","testLabels");

aug = imageDataAugmenter( ...
    RandRotation=[-25 25], ...
    RandXReflection=true, ...
    RandYReflection=true, ...
    RandXScale=[0.8 1.2], ...
    RandYScale=[0.8 1.2], ...
    RandXTranslation=[-20 20], ...
    RandYTranslation=[-20 20]);

imageSize = cfg.inputSize;

augTr = augmentedImageDatastore(imageSize(1:2), imdsTr, ...
    DataAugmentation=aug);
augVa = augmentedImageDatastore(imageSize(1:2), imdsVa);

fprintf("Loading ResNet-18...\n");
net = imagePretrainedNetwork(cfg.backbone, NumClasses=numel(classes));

opts = trainingOptions("adam", ...
    InitialLearnRate=cfg.learningRate, ...
    MaxEpochs=cfg.epochs, ...
    MiniBatchSize=cfg.miniBatch, ...
    Shuffle="every-epoch", ...
    ValidationData=augVa, ...
    ValidationFrequency=cfg.validationFrequency, ...
    Metrics="accuracy", ...
    Plots="training-progress", ...
    Verbose=false);

fprintf("Training %d classes with %d images/class max...\n", numel(classes), n);
[net, info] = trainnet(augTr, net, "crossentropy", opts);

save(cfg.modelFile,"net","classes","cfg","info","-v7.3");
fprintf("Model saved to:\n%s\n",cfg.modelFile);

try
    plotCurves(info,cfg);
catch ME
    warning("Training curve export skipped: %s",ME.message);
end
end

function plotCurves(info,cfg)
if ~isfield(info,"TrainingHistory")
    return
end
th = info.TrainingHistory;
vh = info.ValidationHistory;

f = figure("Visible","off","Position",[100 100 1100 420]);
tiledlayout(1,2,"TileSpacing","compact");

nexttile;
if istable(th) && all(ismember(["Iteration","Loss"],string(th.Properties.VariableNames)))
    plot(th.Iteration,th.Loss); hold on;
end
if istable(vh) && all(ismember(["Iteration","Loss"],string(vh.Properties.VariableNames)))
    v = vh(~isnan(vh.Loss),:);
    if ~isempty(v), plot(v.Iteration,v.Loss,"o-"); end
end
grid on; xlabel("Iteration"); ylabel("Loss"); title("Loss");

nexttile;
if istable(th) && all(ismember(["Iteration","Accuracy"],string(th.Properties.VariableNames)))
    plot(th.Iteration,th.Accuracy); hold on;
end
if istable(vh) && all(ismember(["Iteration","Accuracy"],string(vh.Properties.VariableNames)))
    v = vh(~isnan(vh.Accuracy),:);
    if ~isempty(v), plot(v.Iteration,v.Accuracy,"o-"); end
end
grid on; xlabel("Iteration"); ylabel("Accuracy"); title("Accuracy");

exportgraphics(f,fullfile(cfg.resultsDir,"training_curves.png"),Resolution=150);
close(f);
end
