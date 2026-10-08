function q = assessImageQuality(rgb, seg, cfg)
g   = double(im2gray(imresize(rgb,[384 NaN])));
lap = imfilter(g, fspecial('laplacian',0.2), 'replicate');
q.blurScore    = var(lap(:));
q.brightness   = mean(g(:))/255;
q.leafFraction = seg.leafFraction;
q.ok = true; q.reason = "ok";
if q.blurScore < cfg.blurThreshold,          q.ok=false; q.reason="blurry";
elseif q.brightness < cfg.minBrightness,     q.ok=false; q.reason="too_dark";
elseif q.brightness > cfg.maxBrightness,     q.ok=false; q.reason="too_bright";
elseif q.leafFraction < cfg.minLeafFraction, q.ok=false; q.reason="no_leaf";
end
switch q.reason
    case "blurry",    q.messageEn="Please upload a clearer crop image."; q.messageTa="தயவுசெய்து தெளிவான படத்தை பதிவேற்றவும்.";
    case "too_dark",  q.messageEn="Image is too dark. Retake in daylight."; q.messageTa="படம் இருட்டாக உள்ளது. பகல் வெளிச்சத்தில் மீண்டும் எடுக்கவும்.";
    case "too_bright",q.messageEn="Image is overexposed. Avoid direct glare."; q.messageTa="படம் அதிக வெளிச்சத்தில் உள்ளது. நேரடி ஒளியைத் தவிர்க்கவும்.";
    case "no_leaf",   q.messageEn="No clear leaf found. Fill the frame with one leaf."; q.messageTa="இலை தெளிவாகத் தெரியவில்லை. ஒரு இலையை நெருக்கமாகப் படம் எடுக்கவும்.";
    otherwise,        q.messageEn=""; q.messageTa="";
end
end
