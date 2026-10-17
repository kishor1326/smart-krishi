function p = runNetGray(net, rgb, inputSize)
% Grayscale-input experiment using the existing 3-channel network.
% Converts to grayscale, then replicates grayscale into 3 channels.

gray = im2gray(rgb);
gray3 = repmat(gray, 1, 1, 3);

X = single(imresize(gray3, inputSize(1:2)));

scores = predict(net, X);
scores = extractdata(scores);

p = double(scores(:))';
p = p / max(sum(p), eps);
end