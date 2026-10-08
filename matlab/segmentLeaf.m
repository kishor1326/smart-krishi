function seg = segmentLeaf(rgb, sz)
% Classical IPT segmentation: saturation-Otsu foreground + hue/value lesion rule.
% Heuristic: good on single leaf / plain background; weaker on cluttered field scenes.
if nargin<2, sz = 224; end
I  = im2double(imresize(rgb,[sz sz]));
Is = imgaussfilt(I,1);                       % noise suppression
hsv = rgb2hsv(Is); H = hsv(:,:,1); S = hsv(:,:,2); V = hsv(:,:,3);
fg = imbinarize(S, graythresh(S)) & V > 0.10;
fg = imfill(fg,'holes');
fg = imopen(fg, strel('disk',2));
fg = bwareaopen(fg, round(0.01*sz*sz));
lesion = fg & (H < 0.19 | H > 0.95 | V < 0.22);   % yellow/brown/red/dark = non-green tissue
lesion = bwareaopen(lesion, 20);
seg.resized = uint8(255*I);
seg.masked  = uint8(255*(I .* double(fg)));
seg.leafMask = fg; seg.lesionMask = lesion;
seg.leafFraction   = nnz(fg)/numel(fg);
seg.lesionFraction = nnz(lesion)/max(nnz(fg),1);
end
