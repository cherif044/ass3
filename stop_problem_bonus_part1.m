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

distanceThreshold = 200;
ratioThreshold = 0.8;

tic
distMatrix = pdist2(Descriptor1', Descriptor2', 'euclidean');
[minDists, minIdx] = min(distMatrix, [], 2);
validMatchesIdx = find(minDists < distanceThreshold);
matchesThreshold = [validMatchesIdx'; minIdx(validMatchesIdx)'];
timeBruteForceThreshold = toc;

tic
[sortedDists, sortedIdx] = sort(distMatrix, 2, 'ascend');
ratios = sortedDists(:,1) ./ sortedDists(:,2);
validRatioMatchesIdx = find(ratios < ratioThreshold);
matchesRatio = [validRatioMatchesIdx'; sortedIdx(validRatioMatchesIdx,1)'];
timeBruteForceRatio = toc;

tic
mdl = KDTreeSearcher(Descriptor2');
[I, D] = knnsearch(mdl, Descriptor1', 'K', 2);
ratiosKD = D(:,1) ./ D(:,2);
validIdxKD = find(ratiosKD < ratioThreshold);
matchesKD = [validIdxKD'; I(validIdxKD)'];
timeKDTree = toc;

figure('Name', 'Nearest Neighbor Distance Threshold Matches');
plotMatches(im1, im2, Frame1, Frame2, matchesThreshold);
title(sprintf('Distance Threshold < %d (%d matches)', distanceThreshold, size(matchesThreshold,2)));

figure('Name', 'Brute-force Lowe''s Ratio Test Matches');
plotMatches(im1, im2, Frame1, Frame2, matchesRatio);
title(sprintf('Brute-force Lowe''s Ratio Test < %.2f (%d matches)', ratioThreshold, size(matchesRatio,2)));

figure('Name', 'kd-tree Lowe''s Ratio Test Matches');
plotMatches(im1, im2, Frame1, Frame2, matchesKD);
title(sprintf('kd-tree Lowe''s Ratio Test < %.2f (%d matches)', ratioThreshold, size(matchesKD,2)));

fprintf('Matching summary:\n');
fprintf('-----------------\n');
fprintf('Nearest neighbor threshold matches: %d\n', size(matchesThreshold,2));
fprintf('Brute-force Lowe''s ratio test matches: %d\n', size(matchesRatio,2));
fprintf('kd-tree Lowe''s ratio test matches: %d\n\n', size(matchesKD,2));

fprintf('Runtime summary (seconds):\n');
fprintf('Brute-force threshold matching: %.4f\n', timeBruteForceThreshold);
fprintf('Brute-force ratio test matching: %.4f\n', timeBruteForceRatio);
fprintf('kd-tree ratio test matching: %.4f\n', timeKDTree);

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
