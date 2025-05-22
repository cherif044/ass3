function featureTracking
% Main function for Part 1.1: Keypoint Detection
% Loads the first frame of the hotel sequence, detects keypoints using Harris
% corner detector, and visualizes them with green dots.

% Specify the folder containing the hotel sequence images
folder = './images'; % Replace with the actual path to your images

% Read the first frame (hotel.seq0.png)
im = readImages(folder, 0);

% Extract the first frame from cell array
im = im{1};

% Initial threshold for Harris corner detection
tau = 0.06;

% Detect keypoints
[pt_x, pt_y] = getKeypoints(im, tau);

% Adjust tau to get 200–500 keypoints if needed
if length(pt_x) < 200 || length(pt_x) > 500
    fprintf('Adjusting tau to get 200–500 keypoints...\n');
    % Compute Harris response to adjust tau dynamically
    if size(im, 3) == 3
        im_gray = rgb2gray(im);
    else
        im_gray = im;
    end
    [Ix, Iy] = gradient(im_gray);
    Ix2 = Ix.^2; Iy2 = Iy.^2; Ixy = Ix .* Iy;
    g = fspecial('gaussian', [5 5], 2);
    Sx2 = imfilter(Ix2, g, 'same');
    Sy2 = imfilter(Iy2, g, 'same');
    Sxy = imfilter(Ixy, g, 'same');
    k = 0.04;
    R = (Sx2 .* Sy2 - Sxy.^2) - k * (Sx2 + Sy2).^2;
    tau = 0.1 * max(R(:)); % Dynamic threshold (adjust multiplier as needed)
    [pt_x, pt_y] = getKeypoints(im, tau);
end

% Visualize keypoints on the first frame
figure;
imshow(im);
hold on;
plot(pt_x, pt_y, 'g.', 'LineWidth', 3); % Green dots as required
hold off;
title('Keypoints on First Frame (Part 1.1)');
axis image;

% Nested function to read images
function im = readImages(folder, nums)
    im = cell(numel(nums),1);
    t = 0;
    for k = nums
        t = t+1;
        im{t} = imread(fullfile(folder, ['hotel.seq' num2str(k) '.png']));
        im{t} = im2single(im{t});
    end
end

% Nested function to detect keypoints
function [pt_x, pt_y] = getKeypoints(im, tau)
    % Convert to grayscale if needed
    if size(im, 3) == 3
        im = rgb2gray(im);
    end

    % Compute gradients
    [Ix, Iy] = gradient(im);
    Ix2 = Ix.^2;
    Iy2 = Iy.^2;
    Ixy = Ix .* Iy;

    % Gaussian filter (use 5x5 kernel with sigma = 2)
    g = fspecial('gaussian', [5 5], 2);
    Sx2 = imfilter(Ix2, g, 'same');
    Sy2 = imfilter(Iy2, g, 'same');
    Sxy = imfilter(Ixy, g, 'same');

    % Harris response
    k = 0.04;
    R = (Sx2 .* Sy2 - Sxy.^2) - k * (Sx2 + Sy2).^2;

    % Threshold
    R(R < tau) = 0;

    % Non-maximum suppression (use 5x5 window)
    Rmax = imdilate(R, strel('square', 5));
    corners = (R == Rmax) & (R > 0);
    [pt_y, pt_x] = find(corners);

    fprintf('Number of keypoints detected: %d\n', length(pt_x));
end

end
