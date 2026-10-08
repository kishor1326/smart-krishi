function res = predictDisease(imgPath, outDir)
% Full pipeline: quality gate -> segmentation -> CNN -> confidence gate -> Grad-CAM -> health index.
persistent MODEL
cfg = uzhavanConfig();
if isempty(MODEL)
    if ~isfile(cfg.modelFile)
        error("Trained model not found: %s. Run trainModel first.",cfg.modelFile);
    end
    MODEL = load(cfg.modelFile);
end
if nargin < 2 || isempty(outDir), outDir = cfg.resultsDir; end
if ~isfolder(outDir), mkdir(outDir); end
t0 = tic;
rgb = readRGB(imgPath);
seg = segmentLeaf(rgb, cfg.inputSize(1));
q   = assessImageQuality(rgb, seg, cfg);
res = struct('status',"ok",'messageEn',"",'messageTa',"",'imageQuality',q,'top',[],'confidence',NaN, ...
             'condition',"",'isHealthy',false,'health',[],'flags',strings(0,1),'panel',"panel.png");
cam = uint8(seg.resized);
if ~q.ok
    res.status = "retake"; res.messageEn = q.messageEn; res.messageTa = q.messageTa;
else
    p = runNet(MODEL.net, rgb, cfg.inputSize);
    [ps, ix] = sort(p,'descend'); k = min(3,numel(ps));
    res.top = repmat(struct('label',"",'prob',0),1,k);
    for j = 1:k
        res.top(j).label = string(MODEL.classes{ix(j)});
        res.top(j).prob = ps(j);
    end
    res.confidence = ps(1); res.condition = string(MODEL.classes{ix(1)});
    isH = contains(string(MODEL.classes),'healthy','IgnoreCase',true);
    pH  = sum(p(isH)); res.isHealthy = isH(ix(1));
    if ps(1) < cfg.minConfidence || (ps(1)-ps(2)) < cfg.minMargin
        res.status = "low_confidence";
        res.messageEn = "Possible crop stress detected. Confidence is low. Please capture another image.";
        res.messageTa = "பயிர் அழுத்தம் இருக்கலாம். நம்பகத்தன்மை குறைவு. மீண்டும் தெளிவான படம் எடுக்கவும்.";
    else
        res.health = healthScore(pH, seg.lesionFraction);
        if res.isHealthy && seg.lesionFraction > 0.15
            res.flags(end+1,1) = "visible_damage_but_model_says_healthy (possible nutrient/abiotic stress - verify on field)";
        end
    end
    [cam, ~] = generateGradCAM(MODEL.net, rgb, ix(1), cfg.inputSize);
end
res.timeSec = toc(t0);
savePanel(fullfile(outDir,'panel.png'), seg, cam, res);
end

function savePanel(file, seg, cam, res)
f = figure('Visible','off','Position',[50 50 1500 780]); tl = tiledlayout(2,3,'TileSpacing','compact');
nexttile; imshow(seg.resized); title('1. Input (resized)');
nexttile; imshow(seg.masked);  title(sprintf('2. Leaf segmentation (%.0f%% of frame)',100*seg.leafFraction));
L = uint8(seg.leafMask); L(seg.lesionMask) = 2;
nexttile; imshow(labeloverlay(seg.resized, L, 'Colormap',[0 0.8 0; 1 0 0], 'Transparency',0.55));
title(sprintf('3. Visible damage = %.1f%% of leaf',100*seg.lesionFraction));
nexttile; imshow(cam); title('4. Grad-CAM: where the model looked');
nexttile;
if ~isempty(res.top)
    b = barh(flipud([res.top.prob]')); yticks(1:numel(res.top)); yticklabels(flipud(strrep({res.top.label}','_',' ')));
    xlim([0 1]); xlabel('Probability'); title('5. Top predictions');
else, axis off; text(0.1,0.5,'No prediction (image rejected)'); end
nexttile; axis off;
s = "Status: " + res.status + newline;
if res.status=="ok", s = s + "Condition: " + strrep(res.condition,'_',' ') + newline + sprintf("Confidence: %.1f%%\n",100*res.confidence) ...
    + sprintf("Health index: %d/100 (%s)\n",res.health.score,res.health.band) + "Severity: " + res.health.severity;
else, s = s + res.messageEn; end
text(0,1,s,'VerticalAlignment','top','FontSize',13); title('6. Result');
exportgraphics(f, file, 'Resolution',110); close(f);
end
