function setup_check()
% SETUP_CHECK Verifies MATLAB environment, required toolboxes, support packages,
% project folders, dataset files, and model status for Smart Krishi (Uzhavan AI).

fprintf("========================================================\n");
fprintf("      Smart Krishi (Uzhavan AI) - Setup & Health Check  \n");
fprintf("========================================================\n\n");

%% 1. MATLAB Environment
fprintf("[1/5] Checking MATLAB Release & Environment...\n");
fprintf("      MATLAB Version: %s\n", version);
fprintf("      MATLAB Root:    %s\n\n", matlabroot);

%% 2. Toolboxes & Support Packages
fprintf("[2/5] Checking Required Toolboxes & Support Packages...\n");
toolboxes = [
    struct('name', "Deep Learning Toolbox", 'code', "nnet"), ...
    struct('name', "Image Processing Toolbox", 'code', "images"), ...
    struct('name', "Statistics and Machine Learning Toolbox", 'code', "stats") ...
];

allToolboxesOk = true;
for tb = toolboxes
    v = ver(tb.code);
    if ~isempty(v)
        fprintf("      [PASS] %-40s (v%s)\n", tb.name, v.Version);
    else
        fprintf("      [FAIL] %-40s (NOT INSTALLED)\n", tb.name);
        allToolboxesOk = false;
    end
end

% Check ResNet-18 Backbone availability
cfg = uzhavanConfig();
hasBackbone = false;
try
    if ~isempty(which("imagePretrainedNetwork"))
        hasBackbone = ~isempty(which("resnet18")) || ~isempty(which("imagePretrainedNetwork"));
    elseif ~isempty(which("resnet18"))
        hasBackbone = true;
    end
catch
    hasBackbone = false;
end

if hasBackbone
    fprintf("      [PASS] %-40s (Available)\n", "ResNet-18 Backbone Support");
else
    fprintf("      [WARN] %-40s (May require Add-On installation)\n", "ResNet-18 Backbone");
    fprintf("             Install via Add-On Explorer: 'Deep Learning Toolbox Model for ResNet-18 Network'\n");
end

% Check core deep learning and image processing functions
coreFuncs = ["imagePretrainedNetwork", "trainnet", "augmentedImageDatastore", ...
             "imageDatastore", "gradCAM", "jitterColorHSV"];
fprintf("\n      Core Functions Check:\n");
for k = 1:numel(coreFuncs)
    w = which(coreFuncs(k));
    if isempty(w)
        statusStr = "[MISSING]";
    else
        statusStr = "[FOUND]";
    end
    fprintf("      %-9s %-28s : %s\n", statusStr, coreFuncs(k), w);
end
fprintf("\n");

%% 3. Required Project Folders
fprintf("[3/5] Checking Required Project Folders...\n");
fprintf("      Project Root: %s\n", cfg.root);

modelsDir  = fileparts(cfg.modelFile);
resultsDir = cfg.resultsDir;
datasetDir = cfg.datasetDir;
classFolders = [
    fullfile(datasetDir, "Healthy"), ...
    fullfile(datasetDir, "Early_Blight"), ...
    fullfile(datasetDir, "Late_Blight") ...
];

reqDirs = [
    struct('path', datasetDir, 'desc', "Dataset Root (tomato)"), ...
    struct('path', classFolders(1), 'desc', "Class: Healthy"), ...
    struct('path', classFolders(2), 'desc', "Class: Early_Blight"), ...
    struct('path', classFolders(3), 'desc', "Class: Late_Blight"), ...
    struct('path', modelsDir,  'desc', "Models Folder"), ...
    struct('path', resultsDir, 'desc', "Results Folder") ...
];

for i = 1:numel(reqDirs)
    d = reqDirs(i);
    if ~isfolder(d.path)
        mkdir(d.path);
        fprintf("      [CREATED] %-25s : %s\n", d.desc, d.path);
    else
        fprintf("      [OK]      %-25s : %s\n", d.desc, d.path);
    end
end
fprintf("\n");

%% 4. Dataset Inspection
fprintf("[4/5] Checking Dataset Images...\n");
totalImages = 0;

if isfolder(cfg.datasetDir)
    % Count image files across supported image extensions
    imgExts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'};
    classCounts = zeros(1, numel(classFolders));
    for c = 1:numel(classFolders)
        cFolder = classFolders(c);
        if isfolder(cFolder)
            fList = [];
            for e = 1:numel(imgExts)
                fList = [fList; dir(fullfile(cFolder, ['*' imgExts{e}])); dir(fullfile(cFolder, ['*' upper(imgExts{e})]))]; %#ok<AGROW>
            end
            classCounts(c) = numel(fList);
        end
    end
    totalImages = sum(classCounts);

    if totalImages > 0
        try
            imds = imageDatastore(cfg.datasetDir, IncludeSubfolders=true, ...
                LabelSource="foldernames", FileExtensions=[".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]);
            fprintf("      Total images detected: %d\n", numel(imds.Files));
            disp(countEachLabel(imds));
        catch
            fprintf("      Total images detected: %d\n", totalImages);
            for c = 1:numel(classFolders)
                [~, cName] = fileparts(classFolders(c));
                fprintf("        - %-15s : %d\n", cName, classCounts(c));
            end
        end
    else
        fprintf("      Total images detected: 0\n");
    end
end

if totalImages == 0
    fprintf("      [NOTE] Dataset folders are currently empty.\n");
    fprintf("             Place real tomato leaf photos in:\n");
    fprintf("               - %s\n", classFolders(1));
    fprintf("               - %s\n", classFolders(2));
    fprintf("               - %s\n", classFolders(3));
elseif totalImages < 10
    fprintf("      [WARN] Only %d images found. trainModel requires at least 10 images across classes.\n", totalImages);
else
    fprintf("      [PASS] Sufficient images detected for training.\n");
end
fprintf("\n");

%% 5. Model Status
fprintf("[5/5] Checking Trained Model File...\n");
fprintf("      Model target: %s\n", cfg.modelFile);
if isfile(cfg.modelFile)
    fprintf("      [EXISTS] Trained model is ready for evaluation and inference.\n");
else
    fprintf("      [NOT FOUND] No pre-trained model file found. Train the model using trainModel once dataset is loaded.\n");
end

%% Summary & Next Steps
fprintf("\n========================================================\n");
fprintf("                      Summary                           \n");
fprintf("========================================================\n");
if ~allToolboxesOk
    fprintf(" -> Install missing Toolboxes from MATLAB Home > Add-Ons.\n");
end
if totalImages == 0
    fprintf(" -> Add tomato leaf images to dataset/tomato/ subfolders.\n");
elseif totalImages >= 10 && ~isfile(cfg.modelFile)
    fprintf(" -> Ready to train: run 'trainModel' to train uzhavanNet.\n");
elseif isfile(cfg.modelFile)
    fprintf(" -> Ready to run: run 'evaluateModel' or 'main' for disease diagnosis.\n");
end
fprintf("========================================================\n\n");
end
