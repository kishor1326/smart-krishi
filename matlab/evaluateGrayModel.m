
function G = evaluateGrayModel()
% Evaluate grayscale-input experiment on the same held-out test split.
% Uses existing RGB-trained network; does not retrain.

cfg = uzhavanConfig();
S = load(cfg.modelFile);
T = load(fullfile(cfg.resultsDir, 'testSplit.mat'));

net = S.net;
classes = S.classes;
n = numel(T.testFiles);

pred = strings(n,1);
conf = zeros(n,1);

for k = 1:n
    rgb = readRGB(T.testFiles{k});

    p = runNetGray(net, rgb, cfg.inputSize);

    [conf(k), ix] = max(p);
    pred(k) = string(classes{ix});
end

truth = categorical(T.testLabels, classes);
predC = categorical(pred, classes);

C = confusionmat(truth, predC);

tp = diag(C);
prec = tp ./ max(sum(C,1)', 1);
rec = tp ./ max(sum(C,2), 1);
f1 = 2 .* prec .* rec ./ max(prec + rec, eps);

G.accuracy = sum(tp) / sum(C(:));
G.perClass = table(string(classes), prec, rec, f1, sum(C,2), ...
    'VariableNames', {'Class','Precision','Recall','F1','Support'});
G.macroF1 = mean(f1);
G.confusionMatrix = C;
G.meanConfidence = mean(conf);

disp(G.perClass);
fprintf('Grayscale-input accuracy: %.2f%% | Macro-F1: %.3f\n', ...
    100*G.accuracy, G.macroF1);

save(fullfile(cfg.resultsDir, 'grayMetrics.mat'), 'G');
end
