function setup_check()
% Run this first. It checks the MATLAB release and required components.
fprintf("MATLAB: %s\n",version);
fprintf("Root: %s\n\n",matlabroot);

names = ["imagePretrainedNetwork","trainnet","augmentedImageDatastore", ...
         "imageDatastore","gradCAM","jitterColorHSV"];
for k=1:numel(names)
    fprintf("%-30s : %s\n",names(k),which(names(k)));
end

cfg = uzhavanConfig();
fprintf("\nProject root: %s\n",cfg.root);
fprintf("Dataset:      %s\n",cfg.datasetDir);
fprintf("Model file:   %s\n",cfg.modelFile);

if ~isfolder(cfg.datasetDir)
    warning("Dataset folder does not exist.");
end

if isfolder(cfg.datasetDir)
    imds = imageDatastore(cfg.datasetDir,IncludeSubfolders=true,LabelSource="foldernames");
    fprintf("Images found: %d\n",numel(imds.Files));
    if ~isempty(imds.Files)
        disp(countEachLabel(imds));
    end
end

fprintf("\nNext step: install the ResNet-18 support package, then run trainModel.\n");
end
