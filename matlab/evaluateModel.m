function M = evaluateModel()
% Held-out test metrics + confusion matrix + latency. Run once after training.
cfg = uzhavanConfig(); S = load(cfg.modelFile); T = load(fullfile(cfg.resultsDir,'testSplit.mat'));
net = S.net; classes = S.classes; n = numel(T.testFiles);
pred = strings(n,1); conf = zeros(n,1); ms = zeros(n,1);
for k = 1:n
    rgb = readRGB(T.testFiles{k}); t = tic;
    p = runNet(net, rgb, cfg.inputSize); ms(k) = 1000*toc(t);
    [conf(k), ix] = max(p); pred(k) = string(classes{ix});
end
truth = categorical(T.testLabels, classes); predC = categorical(pred, classes);
C = confusionmat(truth, predC);
tp = diag(C); prec = tp./max(sum(C,1)',1); rec = tp./max(sum(C,2),1); f1 = 2*prec.*rec./max(prec+rec,eps);
M.accuracy = sum(tp)/sum(C(:));
M.perClass = table(string(classes), prec, rec, f1, sum(C,2), 'VariableNames',{'Class','Precision','Recall','F1','Support'});
M.macroF1 = mean(f1); M.medianInferenceMs = median(ms(2:end)); M.meanConfidence = mean(conf);
M.fractionBelowThreshold = mean(conf < cfg.minConfidence);
disp(M.perClass); fprintf('Test accuracy %.2f%% | macro-F1 %.3f | median %.1f ms/img (excl. warm-up)\n', 100*M.accuracy, M.macroF1, M.medianInferenceMs);
f = figure('Visible','off','Position',[100 100 900 800]);
confusionchart(truth, predC, 'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('Test-set confusion matrix (lab-style held-out split)');
exportgraphics(f, fullfile(cfg.resultsDir,'confusion_matrix.png'),'Resolution',150); close(f);
save(fullfile(cfg.resultsDir,'metrics.mat'),'M');
end
