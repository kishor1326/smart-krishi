function js = predictDiseaseJSON(imgPath, outDir)
% Entry point called from Python (MATLAB Engine). Always returns a JSON string.
try
    js = jsonencode(predictDisease(imgPath, outDir));
catch ME
    js = jsonencode(struct('status',"error",'messageEn',string(ME.message)));
end
end
