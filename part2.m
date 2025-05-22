function shape_alignment
% Main script to perform shape alignment, display results, and report errors/runtimes
% Save as shape_alignment.m and run by typing: shape_alignment
% Ensure the 'data' folder is in the same directory as this script
% Ensure MATLAB's working directory is set to this folder (use: cd /path/to/this/folder)

clear; clf;
imgPath = 'data';
objList = {'apple', 'bat', 'bell', 'bird', 'Bone', 'bottle', 'brick', ...
    'butterfly', 'camel', 'car', 'carriage', 'cattle', 'cellular_phone', ...
    'chicken', 'children', 'device7', 'dog', 'elephant', 'face', 'fork', 'hammer', ...
    'Heart', 'horse', 'jar', 'turtle'};
numObj = length(objList);
err_align = zeros(numObj, 1);
runtimes = zeros(numObj, 1);
failed_images = {};

% Verify data folder exists
if ~exist(imgPath, 'dir')
    error('Data folder "%s" not found in current directory: %s\nSet MATLAB working directory using: cd /path/to/folder', imgPath, pwd);
end

% Check data folder contents
files = dir(fullfile(imgPath, '*.png'));
if isempty(files)
    error('No PNG files found in data folder: %s', imgPath);
end
fprintf('Found %d files in data folder: %s\n', length(files), imgPath);
for i = 1:length(files)
    fprintf('File %d: %s\n', i, files(i).name);
end

figure(1);
ha = tight_subplot(5,5,[.01 .03],[.1 .01],[.01 .01]);
for i = 1:numObj
    objName = objList{i};
    file1 = fullfile(imgPath, [objName, '_1.png']);
    file2 = fullfile(imgPath, [objName, '_2.png']);
    
    % Check if files exist
    if ~exist(file1, 'file') || ~exist(file2, 'file')
        fprintf('Warning: Image files for "%s" not found (%s, %s). Skipping.\n', objName, file1, file2);
        failed_images{end+1} = objName;
        err_align(i) = NaN;
        axes(ha(i)); title(objName, 'FontSize', 8);
        continue;
    end
    
    % Load and validate images
    try
        im1 = imread(file1);
        im2 = imread(file2);
        fprintf('Loaded "%s": im1 size=%s, im2 size=%s\n', objName, mat2str(size(im1)), mat2str(size(im2)));
        
        % Convert to grayscale if RGB
        if size(im1, 3) == 3
            im1 = rgb2gray(im1);
        end
        if size(im2, 3) == 3
            im2 = rgb2gray(im2);
        end
        % Convert to binary
        im1 = im1 > 0;
        im2 = im2 > 0;
        
        % Validate image data
        if isempty(im1) || isempty(im2) || ~ismatrix(im1) || ~ismatrix(im2)
            fprintf('Warning: Invalid image data for "%s". im1=%s, im2=%s. Skipping.\n', ...
                objName, mat2str(size(im1)), mat2str(size(im2)));
            failed_images{end+1} = objName;
            err_align(i) = NaN;
            axes(ha(i)); title(objName, 'FontSize', 8);
            continue;
        end
    catch e
        fprintf('Warning: Failed to read images for "%s": %s. Skipping.\n', objName, e.message);
        failed_images{end+1} = objName;
        err_align(i) = NaN;
        axes(ha(i)); title(objName, 'FontSize', 8);
        continue;
    end
    
    % Perform alignment
    try
        tic;
        T = align_shape(im1, im2);
        aligned = imtransform(im1, maketform('projective', double(T')), ...
            'XData', [1 size(im1,2)], 'YData', [1 size(im1,1)]);
        runtimes(i) = toc;
    catch e
        fprintf('Warning: Alignment failed for "%s": %s. Skipping.\n', objName, e.message);
        failed_images{end+1} = objName;
        err_align(i) = NaN;
        axes(ha(i)); title(objName, 'FontSize', 8);
        continue;
    end
    
    % Display alignment
    axes(ha(i));
    c = double(repmat(im1, [1 1 3]));
    c(:,:,2) = im2; c(:,:,3) = aligned;
    imshow(c);
    title(objName, 'FontSize', 8);
    
    % Compute and display error
    err_align(i) = evalAlignment(aligned, im2);
    fprintf('Error for aligning "%s": %f, Runtime: %f seconds\n', objName, err_align(i), runtimes(i));
end

% Display failed images
if ~isempty(failed_images)
    fprintf('Failed to process %d/%d images (%.1f%%):\n', length(failed_images), numObj, 100*length(failed_images)/numObj);
    for i = 1:length(failed_images)
        fprintf('  - %s\n', failed_images{i});
    end
else
    fprintf('All %d images processed successfully.\n', numObj);
end

% Display error bar chart
figure(2);
bar(err_align);
set(gca, 'XTick', 1:numObj);
set(gca, 'XTickLabel', objList, 'FontSize', 12, 'XTickLabelRotation', 45);
xlabel('Object', 'FontSize', 16);
ylabel('Alignment error', 'FontSize', 16);
title('Alignment Errors per Object', 'FontSize', 16);

% Compute and report averages
valid_err = err_align(~isnan(err_align));
valid_times = runtimes(~isnan(err_align));
if ~isempty(valid_err)
    fprintf('Averaged alignment error = %f\n', mean(valid_err));
    fprintf('Averaged runtime = %f seconds\n', mean(valid_times));
else
    fprintf('No valid alignments completed.\n');
end

function T = align_shape(im1, im2)
% im1: input edge image 1
% im2: input edge image 2
% Output: transformation T [3] x [3]

% Validate inputs
if nargin < 2
    error('align_shape requires two input arguments: im1 and im2');
end
if isempty(im1) || isempty(im2) || ~ismatrix(im1) || ~ismatrix(im2)
    error('Invalid input images: im1=%s, im2=%s', mat2str(size(im1)), mat2str(size(im2)));
end

% Find non-zero points
[y1, x1] = find(im1);
[y2, x2] = find(im2);
if isempty(x1) || isempty(x2)
    fprintf('Warning: No edge points found in one or both images for alignment.\n');
    T = eye(3);
    return;
end
pts1 = [x1, y1];
pts2 = [x2, y2];

% Ensure same number of points for correspondence using nearest neighbors
n = min(size(pts1, 1), size(pts2, 1));
if n < 4
    fprintf('Warning: Fewer than 4 points for alignment. Returning identity matrix.\n');
    T = eye(3);
    return;
end

% Find nearest neighbors for correspondence
idx = knnsearch(pts2, pts1, 'K', 1); % Find closest points in im2 for each point in im1
pts1 = pts1(1:n, :);
pts2 = pts2(idx(1:n), :);

% Add homogeneous coordinates
pts1 = [pts1, ones(n, 1)];
pts2 = [pts2, ones(n, 1)];

% Estimate affine transformation using least squares
A = zeros(2*n, 6);
b = zeros(2*n, 1);
for i = 1:n
    A(2*i-1, :) = [pts1(i, 1), pts1(i, 2), 1, 0, 0, 0];
    A(2*i, :) = [0, 0, 0, pts1(i, 1), pts1(i, 2), 1];
    b(2*i-1) = pts2(i, 1);
    b(2*i) = pts2(i, 2);
end
x = pinv(A) * b;
T = [x(1), x(2), x(3); x(4), x(5), x(6); 0, 0, 1];

function dispim = displayAlignment(im1, im2, aligned1, thick)
if ~exist('thick', 'var')
    thick = false;
end
if thick
    dispim = cat(3, ordfilt2(im1, 9, ones(3)), ordfilt2(aligned1, 9, ones(3)), ordfilt2(im2, 9, ones(3)));
else
    dispim = cat(3, im1, aligned1, im2);
end

function err = evalAlignment(aligned1, im2)
d2 = bwdist(im2);
err1 = mean(d2(logical(aligned1)));
d1 = bwdist(aligned1);
err2 = mean(d1(logical(im2)));
err = (err1+err2)/2;

function ha = tight_subplot(Nh, Nw, gap, marg_h, marg_w)
if nargin<3; gap = .02; end
if nargin<4 || isempty(marg_h); marg_h = .05; end
if nargin<5; marg_w = .05; end
if numel(gap)==1
    gap = [gap gap];
end
if numel(marg_w)==1
    marg_w = [marg_w marg_w];
end
if numel(marg_h)==1
    marg_h = [marg_h marg_h];
end
axh = (1-sum(marg_h)-(Nh-1)*gap(1))/Nh;
axw = (1-sum(marg_w)-(Nw-1)*gap(2))/Nw;
py = 1-marg_h(2)-axh;
ha = zeros(Nh*Nw,1);
ii = 0;
for ih = 1:Nh
    px = marg_w(1);
    for ix = 1:Nw
        ii = ii+1;
        ha(ii) = axes('Units','normalized', ...
            'Position',[px py axw axh], ...
            'XTickLabel','', ...
            'YTickLabel','');
        px = px+axw+gap(2);
    end
    py = py-axh-gap(1);
end
