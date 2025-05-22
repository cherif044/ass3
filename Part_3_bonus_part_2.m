function featureMatching
im1 = imread('stop1.jpg');
im2 = imread('stop2.jpg');

if size(im1,3) == 3
    im1 = im2double(rgb2gray(im1));
else
    im1 = im2double(im1);
end

if size(im2,3) == 3
    im2 = im2double(rgb2gray(im2));
else
    im2 = im2double(im2);
end

data = load('SIFT_features.mat');
Descriptor1 = double(data.Descriptor1);
Descriptor2 = double(data.Descriptor2);
Frame1 = data.Frame1;
Frame2 = data.Frame2;

distMatrix = pdist2(Descriptor1', Descriptor2', 'euclidean');

distanceThreshold = 200;
[minDists, minIdx] = min(distMatrix, [], 2);
validMatchesIdx = find(minDists < distanceThreshold);
matchesThreshold = [validMatchesIdx'; minIdx(validMatchesIdx)'];

figure('Name', 'Nearest Neighbor Distance Threshold Matches');
plotMatches(im1, im2, Frame1, Frame2, matchesThreshold);
title(sprintf('Matches with Distance Threshold < %d (%d matches)', distanceThreshold, size(matchesThreshold,2)));

ratioThreshold = 0.8;
[sortedDists, sortedIdx] = sort(distMatrix, 2, 'ascend');
ratios = sortedDists(:,1) ./ sortedDists(:,2);
validRatioMatchesIdx = find(ratios < ratioThreshold);
matchesRatio = [validRatioMatchesIdx'; sortedIdx(validRatioMatchesIdx,1)'];

figure('Name', 'Lowe''s Ratio Test Matches');
plotMatches(im1, im2, Frame1, Frame2, matchesRatio);
title(sprintf('Matches with Lowe''s Ratio Test < %.2f (%d matches)', ratioThreshold, size(matchesRatio,2)));

fprintf('Matching summary:\n');
fprintf('-----------------\n');
fprintf('Nearest neighbor threshold matches: %d\n', size(matchesThreshold,2));
fprintf('Lowe''s ratio test matches: %d\n\n', size(matchesRatio,2));
fprintf(['Difference: The ratio test is more selective and filters out ambiguous matches by\n' ...
         'comparing the closest and second closest descriptor distances, while thresholding\n' ...
         'only considers absolute distance values.\n']);

% Example bounding box in image 1
x1 = 150;
y1 = 120;
w1 = 80;
h1 = 100;

if ~isempty(matchesRatio)
    idx1 = matchesRatio(1,1);
    idx2 = matchesRatio(2,1);

    u1 = Frame1(1, idx1);
    v1 = Frame1(2, idx1);
    s1 = Frame1(3, idx1);
    theta1 = Frame1(4, idx1);

    u2 = Frame2(1, idx2);
    v2 = Frame2(2, idx2);
    s2 = Frame2(3, idx2);
    theta2 = Frame2(4, idx2);

    x2 = x1 + (u2 - u1);
    y2 = y1 + (v2 - v1);
    w2 = w1 * (s2 / s1);
    h2 = h1 * (s2 / s1);
    o2 = theta2 - theta1;

    fprintf('Predicted bounding box in image 2:\n');
    fprintf('Center: (%.2f, %.2f)\n', x2, y2);
    fprintf('Width: %.2f\n', w2);
    fprintf('Height: %.2f\n', h2);
    fprintf('Orientation: %.2f radians\n', o2);
end

end

function plotMatches(I1, I2, F1, F2, matches)
height = max(size(I1,1), size(I2,1));
width = size(I1,2) + size(I2,2);
composite = zeros(height, width);
composite(1:size(I1,1), 1:size(I1,2)) = I1;
composite(1:size(I2,1), size(I1,2)+1:end) = I2;

imshow(composite);
hold on;

for k = 1:size(matches,2)
    idx1 = matches(1,k);
    idx2 = matches(2,k);
    
    x1 = F1(1, idx1);
    y1 = F1(2, idx1);
    x2 = F2(1, idx2) + size(I1,2);
    y2 = F2(2, idx2);
    
    line([x1 x2], [y1 y2], 'Color', 'g', 'LineWidth', 1);
    plot(x1, y1, 'ro', 'MarkerSize', 6, 'LineWidth', 1.5);
    plot(x2, y2, 'bo', 'MarkerSize', 6, 'LineWidth', 1.5);
end

hold off;
axis image off;

end
