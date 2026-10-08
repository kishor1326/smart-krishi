function p = runNet(net, rgb, inputSize)
% Single-image prediction for a dlnetwork returned by trainnet.
X = single(imresize(rgb,inputSize(1:2)));
scores = predict(net,X);
scores = extractdata(scores);
p = double(scores(:))';
p = p / max(sum(p),eps);
end
