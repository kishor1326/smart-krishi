function main()
% Uzhavan AI - Tomato Disease Detection Demo
% Uses an existing trained model; does NOT retrain automatically.

clc;
close all;

cfg = uzhavanConfig();

% Check that the trained model exists
if ~isfile(cfg.modelFile)
    error("Trained model not found: %s. Run trainModel once first.", ...
        cfg.modelFile);
end

% Select a tomato leaf image
[file, folder] = uigetfile( ...
    {'*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff', ...
    'Tomato leaf images (*.jpg, *.jpeg, *.png, *.bmp, *.tif, *.tiff)'}, ...
    'Select a tomato leaf image');

% Stop if the user cancels
if isequal(file, 0)
    disp("Image selection cancelled.");
    return;
end

imgPath = fullfile(folder, file);

fprintf("\nSelected image: %s\n", imgPath);
fprintf("Running disease prediction...\n");

% Run the trained model
res = predictDisease(imgPath);

% Display key results
fprintf("\n========== UZHAVAN AI RESULT ==========\n");
fprintf("Status     : %s\n", char(res.status));

if isfield(res, 'condition') && strlength(res.condition) > 0
    fprintf("Prediction : %s\n", ...
        char(strrep(res.condition, "_", " ")));
end

if isfield(res, 'confidence') && ~isnan(res.confidence)
    fprintf("Confidence : %.2f%%\n", 100 * res.confidence);
end

if isfield(res, 'health') && isstruct(res.health) ...
        && isfield(res.health, 'score')
    fprintf("Health index: %d/100\n", res.health.score);
end

if isfield(res, 'messageEn') && strlength(res.messageEn) > 0
    fprintf("Message    : %s\n", char(res.messageEn));
end

fprintf("Time       : %.2f seconds\n", res.timeSec);
fprintf("=======================================\n\n");

% Open the generated result panel
panelFile = fullfile(cfg.resultsDir, 'panel.png');

if isfile(panelFile)
    figure('Name', 'Uzhavan AI - Prediction Result');
    imshow(imread(panelFile));
    title('Uzhavan AI - Tomato Disease Detection');
else
    warning("Prediction completed, but panel.png was not found: %s", ...
        panelFile);
end

end