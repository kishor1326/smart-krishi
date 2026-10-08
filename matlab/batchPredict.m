function T = batchPredict(folder)
cfg = uzhavanConfig();
files = [dir(fullfile(folder,'*.jpg')); dir(fullfile(folder,'*.jpeg')); dir(fullfile(folder,'*.png'))];
n = numel(files); File=strings(n,1); Status=File; Condition=File; Conf=nan(n,1); Score=nan(n,1); Sec=nan(n,1);
for k = 1:n
    [~,baseName,~] = fileparts(files(k).name);
    out = fullfile(cfg.resultsDir,'batch',baseName);
    r = predictDisease(fullfile(folder,files(k).name), out);
    File(k)=files(k).name; Status(k)=r.status; Condition(k)=r.condition; Conf(k)=r.confidence; Sec(k)=r.timeSec;
    if ~isempty(r.health), Score(k)=r.health.score; end
end
T = table(File,Status,Condition,Conf,Score,Sec);
writetable(T, fullfile(cfg.resultsDir,'batch_predictions.csv')); disp(T);
end
