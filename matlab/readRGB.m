function rgb = readRGB(path)
rgb = imread(path);
try   % honour phone EXIF rotation
    info = imfinfo(path);
    if isfield(info,'Orientation')
        switch info(1).Orientation
            case 3, rgb = rot90(rgb,2);
            case 6, rgb = rot90(rgb,-1);
            case 8, rgb = rot90(rgb,1);
        end
    end
catch
end
if size(rgb,3)==1, rgb = repmat(rgb,1,1,3);
elseif size(rgb,3)==4, rgb = rgb(:,:,1:3); end
if ~isa(rgb,'uint8'), rgb = im2uint8(rgb); end
end
