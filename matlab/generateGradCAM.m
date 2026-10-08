function [overlay, map] = generateGradCAM(net, rgb, classIdx, inputSize)
X = single(imresize(rgb, inputSize(1:2)));
overlay = uint8(X); map = [];
try
    map  = mat2gray(gradCAM(net, X, classIdx));
    heat = ind2rgb(uint8(255*map), jet(256));
    overlay = uint8(0.55*X + 0.45*255*heat);
catch ME
    warning('Grad-CAM failed: %s', ME.message);
end
end
