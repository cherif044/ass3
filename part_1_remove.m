function featureTracking
% Main function for Part 1.1 (Keypoint Detection) and Part 1.2 (Feature Tracking)
% Detects keypoints in the first frame of the hotel sequence and tracks them
% across the sequence using the Kanade-Lucas-Tomasi (KLT) tracker.

% Specify the folder containing the hotel sequence images
folder = './images'; % Replace with the actual path to your images

% Read frames 0 to 50 (51 frames total)
im = readImages(folder, 0:50);

% --- Part 1.1: Keypoint Detection ---
% Initial threshold for Harris corner detection
tau = 0.1;

% Detect keypoints in the first frame
[pt_x, pt_y] = getKeypoints(im{1}, tau);

% Adjust tau to get 200–500 keypoints if needed
multipliers = [0.1, 0.05, 0.01, 0.005, 0.001];
for m = multipliers
    if length(pt_x) >= 200 && length(pt_x) <= 500
        break;
    end
    fprintf('Adjusting tau to get 200–500 keypoints with multiplier %.3f...\n', m);
    tau = m;
    [pt_x, pt_y] = getKeypoints(im{1}, tau);
end

if isempty(pt_x)
    error('No keypoints detected. Adjust threshold or check image.');
end

% Visualize keypoints on the first frame
figure('Visible', 'on');
imshow(im{1});
hold on;
plot(pt_x, pt_y, 'g.', 'LineWidth', 3);
hold off;
title('Keypoints on First Frame (Part 1.1)');
axis image;
drawnow;
pause(0.1);

% --- Part 1.2: Feature Tracking ---
% Track keypoints across all frames
ws = 15;
fprintf('Starting tracking with %d keypoints...\n', length(pt_x));
[track_x, track_y, valid] = trackPoints(pt_x, pt_y, im, ws);
fprintf('Tracking completed for %d frames.\n', size(track_x, 2));

% Display keypoints at frame 1 (green) and tracked positions at frame 2 (red)
figure('Visible', 'on');
imshow(im{1});
hold on;
plot(track_x(:, 1), track_y(:, 1), 'g.', 'LineWidth', 3);
if size(track_x, 2) > 1 && any(valid(:, 2))
    plot(track_x(valid(:, 2), 2), track_y(valid(:, 2), 2), 'r.', 'LineWidth', 3);
else
    fprintf('Warning: No valid keypoints tracked to frame 2.\n');
end
hold off;
title('Keypoints (Green) and Tracked Positions at Frame 2 (Red) on First Frame');
axis image;
drawnow;
pause(0.1);

% For 20 random keypoints, draw their 2D paths over the sequence
num_random = 20;
N = length(pt_x);
if num_random > N
    num_random = N;
end
random_indices = randperm(N, num_random);
colors = hsv(num_random); % Distinct colors for trajectories
figure('Visible', 'on');
imshow(im{1});
hold on;
for i = 1:num_random
    idx = random_indices(i);
    valid_frames = find(valid(idx, :));
    if length(valid_frames) > 1
        plot(track_x(idx, valid_frames), track_y(idx, valid_frames), ...
             'Color', colors(i, :), 'LineWidth', 2);
    else
        plot(track_x(idx, 1), track_y(idx, 1), '.', ...
             'Color', colors(i, :), 'LineWidth', 3);
        fprintf('Warning: Only frame 1 data for keypoint %d.\n', idx);
    end
end
hold off;
title('Trajectories of 20 Random Keypoints Over the Sequence');
axis image;
drawnow;
pause(0.1);

% Plot points that have moved out of the frame at some point
[height, width] = size(im{1});
out_of_frame = false(N, 1);
for t = 1:size(track_x, 2)
    out_of_frame = out_of_frame | ...
        (track_x(:, t) < 1 | track_x(:, t) > width | ...
         track_y(:, t) < 1 | track_y(:, t) > height);
end
out_x = track_x(out_of_frame, 1);
out_y = track_y(out_of_frame, 1);
figure('Visible', 'on');
imshow(im{1});
hold on;
plot(out_x, out_y, 'b.', 'LineWidth', 3);
hold off;
title('Points That Moved Out of Frame (Initial Positions)');
axis image;
drawnow;
pause(0.1);

% Nested function to read images
function im = readImages(folder, nums)
    im = cell(numel(nums),1);
    t = 0;
    for k = nums
        t = t+1;
        try
            img = imread(fullfile(folder, ['hotel.seq' num2str(k) '.png']));
            if size(img, 3) == 3
                img = rgb2gray(img);
            end
            im{t} = im2single(img);
            if isempty(im{t}) || all(im{t}(:) == 0)
                error('Image %d is empty or invalid.', k);
            end
        catch ME
            fprintf('Error loading image %d: %s\n', k, ME.message);
            im{t} = zeros(480, 640, 'single');
        end
    end
end

% Nested function to detect keypoints (Part 1.1)
function [pt_x, pt_y] = getKeypoints(im, tau)
    if size(im, 3) == 3
        im = rgb2gray(im);
    end
    [Ix, Iy] = gradient(im);
    Ix2 = Ix.^2;
    Iy2 = Iy.^2;
    Ixy = Ix .* Iy;
    g = fspecial('gaussian', [5 5], 2);
    Sx2 = imfilter(Ix2, g, 'same');
    Sy2 = imfilter(Iy2, g, 'same');
    Sxy = imfilter(Ixy, g, 'same');
    k = 0.04;
    R = (Sx2 .* Sy2 - Sxy.^2) - k * (Sx2 + Sy2).^2;
    R = R / max(R(:)); % Normalize Harris response
    fprintf('Min Harris response: %.2f, Max Harris response: %.2f\n', min(R(:)), max(R(:)));
    R(R < tau) = 0;
    Rmax = imdilate(R, strel('square', 5));
    corners = (R == Rmax) & (R > 0);
    [pt_y, pt_x] = find(corners);
    fprintf('Threshold tau: %.2f\n', tau);
    fprintf('Number of keypoints detected: %d\n', length(pt_x));
end

% Nested function to track points across the sequence (Part 1.2)
function [track_x, track_y, valid] = trackPoints(pt_x, pt_y, im, ws)
    N = numel(pt_x);
    nim = numel(im);
    track_x = zeros(N, nim);
    track_y = zeros(N, nim);
    valid = true(N, nim);
    track_x(:, 1) = pt_x(:);
    track_y(:, 1) = pt_y(:);

    for t = 1:nim-1
        fprintf('Tracking frame %d to %d...\n', t, t+1);
        tic;
        [track_x(:, t+1), track_y(:, t+1), valid(:, t+1)] = ...
            getNextPoints(track_x(:, t), track_y(:, t), valid(:, t), im{t}, im{t+1}, ws);
        fprintf('Frame %d to %d took %.2f seconds.\n', t, t+1, toc);
    end
end

% Nested function to implement KLT tracker between two frames (Part 1.2)
function [x2, y2, valid] = getNextPoints(x, y, prev_valid, im1, im2, ws)
    N = length(x);
    x2 = x;
    y2 = y;
    valid = prev_valid;
    half_ws = floor(ws / 2);
    [height, width] = size(im1);
    [X, Y] = meshgrid(1:width, 1:height);
    max_iter = 10;
    thresh = 0.01;

    [Ix, Iy] = gradient(im1);

    for i = 1:N
        if ~valid(i)
            x2(i) = x(i);
            y2(i) = y(i);
            continue;
        end

        x_curr = x(i);
        y_curr = y(i);
        if x_curr < 1 || x_curr > width || y_curr < 1 || y_curr > height
            valid(i) = false;
            x2(i) = x_curr;
            y2(i) = y_curr;
            continue;
        end

        x_min = max(1, floor(x_curr - half_ws));
        x_max = min(width, floor(x_curr + half_ws));
        y_min = max(1, floor(y_curr - half_ws));
        y_max = min(height, floor(y_curr + half_ws));
        [wx, wy] = meshgrid(x_min:x_max, y_min:y_max);

        Ix_patch = interp2(X, Y, Ix, wx, wy, 'linear', 0);
        Iy_patch = interp2(X, Y, Iy, wx, wy, 'linear', 0);
        Sxx = sum(Ix_patch(:).^2);
        Syy = sum(Iy_patch(:).^2);
        Sxy = sum(Ix_patch(:) .* Iy_patch(:));
        detM = Sxx * Syy - Sxy^2;
        if abs(detM) < eps
            fprintf('Singular matrix at point %d (pre-check)\n', i);
            valid(i) = false;
            continue;
        end

        u = 0;
        v = 0;
        x_new = x_curr;
        y_new = y_curr;

        for iter = 1:max_iter
            I1_patch = interp2(X, Y, im1, wx, wy, 'linear', 0);
            I2_patch = interp2(X, Y, im2, wx + u, wy + v, 'linear', 0);
            It_patch = I2_patch - I1_patch;
            bx = -sum(Ix_patch(:) .* It_patch(:));
            by = -sum(Iy_patch(:) .* It_patch(:));
            u_new = (Syy * bx - Sxy * by) / detM;
            v_new = (-Sxy * bx + Sxx * by) / detM;
            u = u + u_new;
            v = v + v_new;
            x_new = x_curr + u;
            y_new = y_curr + v;
            if sqrt(u_new^2 + v_new^2) < thresh
                break;
            end
        end

        x2(i) = x_new;
        y2(i) = y_new;
        if x_new < 1 || x_new > width || y_new < 1 || y_new > height
            valid(i) = false;
        end
    end
end
end
