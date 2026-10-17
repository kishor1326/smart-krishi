function js = predictDiseaseJSON(imgPath, outDir)
% Entry point called from Python or batch. Always returns a JSON string.
if nargin < 2, outDir = []; end
try
    js = jsonencode(predictDisease(imgPath, outDir));
catch ME
    js = jsonencode(struct('status',"error",'messageEn',string(ME.message)));
end
end
