clear all; close all;
%% init parameter
addpath('AKM');
run('vlfeat\toolbox\vl_setup.m');
datasetDir = 'oxford\images\';
num_words = 1000;
num_iterations = 5;
num_trees = 8;
dim = 128;
if_weight = 'tfidf';
if_norm = 'l1';
if_dist = 'l1';
verbose=1;
%% Compute SIFT features
if ~exist('oxford\feat\feature.bin', 'file')
    fprintf('Computing SIFT features:\n');
    
    features = zeros(128, 2000000);
    nfeat = 0;
    files = dir(fullfile(datasetDir, '*.jpg'));
    nfiles = length(files);
    features_per_image = zeros(1,nfiles);
    for i=1:nfiles
        fprintf('Extracting features %d/%d\n', i, nfiles);
        imgPath = strcat(datasetDir, files(i).name);
        I = im2single(rgb2gray(imread(imgPath)));
        I = imresize(I, 0.6);
        [frame, sift] = vl_covdet(I, 'method', 'Hessian', 'estimateAffineShape', true);
        
        if nfeat+size(sift,2) > size(features,2)
            features = [features zeros(128,1000000)];
        end
        features(:,nfeat+1:nfeat+size(sift,2)) = sift;
        nfeat = nfeat+size(sift,2);
        features_per_image(i) = size(sift, 2);
    end
    features = features(:,1:nfeat);
    fid = fopen('oxford\feat\feature.bin', 'w');
    fwrite(fid, features, 'float');
    fclose(fid);
    
    save('oxford\feat\feat_info.mat', 'features_per_image', 'files');
else
    fprintf('Loading SIFT features:\n');
    file = dir('oxford\feat\feature.bin');
    %features = zeros(128, file.bytes/(4*128), 'single');

    fid = fopen('oxford\feat\feature.bin', 'r');
    features = fread(fid, [128, file.bytes/(4*128)], 'float');
    fclose(fid);
    
    load('oxford\feat\feat_info.mat');
end

%% compute rootSIFT
fprintf('Computing rootSIFT features:\n');
num_features = size(features, 2);
%rootSIFT = zeros(dim, num_features);

% matlabpool('open',4);
% for k = 1:5000000:num_features
%     eIdx = k+5000000-1;
%     if eIdx > num_features
%         eIdx = num_features;
%     end
%     parfor i=k:eIdx
%         features(:, i) = sqrt(features(:, i) / sum(features(:,i)));
%     end
% end
% matlabpool close;

%% Run AKM to build dictionary
fprintf('Building the dictionary:\n');
num_images = length(files);
dict_params =  {num_iterations, 'kdt', num_trees};

% build the dictionary
if exist('oxford\feat\dict.mat', 'file')
    load('oxford\feat\dict.mat');
else
    randIndex = randperm(size(features,2));
    dict_words = ccvBowGetDict(features(:,randIndex(1:100000)), [], [], num_words, 'flat', 'akmeans', ...
        [], dict_params);
    save('oxford\feat\dict.mat', 'dict_words');
end

% compute sparse frequency vector
fprintf('Computing the words\n');
dict = ccvBowGetWordsInit(dict_words, 'flat', 'akmeans', [], dict_params);

if exist('oxford\feat\words.mat', 'file')
    load('oxford\feat\words.mat');
else
    words = cell(1, num_images);
    for i=1:num_images
        fprintf('Quantizing %d/%d images\n', i, num_images);
        if i==1
            bIndex = 1;
        else
            bIndex = sum(features_per_image(1:i-1))+1;
        end
        eIndex = bIndex + features_per_image(i)-1;
        words{i} = ccvBowGetWords(dict_words, features(:, bIndex:eIndex), [], dict);
    end;
    save('oxford\feat\words.mat', 'words');
end
%fprintf('Computing sparse frequency vector\n');
%dict = ccvBowGetWordsInit(dict_words, 'flat', 'akmeans', [], dict_params);
%words = ccvBowGetWords(dict_words, root_sift, [], dict);
%ccvBowGetWordsClean(dict);

% create an inverted file for the images
fprintf('Creating and searching an inverted file\n');
inv_file = ccvInvFileInsert([], words, num_words);
ccvInvFileCompStats(inv_file, if_weight, if_norm);
%save('inverted_file.mat', 'if_weight', 'if_norm', 'if_dist', 'inv_file');

%% Query images
q_files = dir(fullfile('oxford\groundtruth', '*query.txt'));
%oxc1_all_souls_000013 136.5 34.1 648.5 955.7
ntop = 0;
% load query image
for k=1:length(q_files)
    k
    fid = fopen(strcat('oxford\groundtruth\', q_files(k).name), 'r');
    str = fgetl(fid);
    [image_name, remain] = strtok(str, ' ');
    fclose(fid);
    numbers = str2num(remain);
    x1 = numbers(1);
    y1 = numbers(2);
    x2 = numbers(3);
    y2 = numbers(4);
    file = strcat('oxford\images\', image_name(6:end), '.jpg');
    I = im2single(rgb2gray(imread(file)));
    %imshow(I); hold on;
    %plot([x1 x2], [y1 y1], 'g');
    %plot([x2 x2], [y1 y2], 'g');
    %plot([x2 x1], [y2 y2], 'g');
    %plot([x1 x1], [y2 y1], 'g');
    %hold off;
    % compute rootSIFT features
    [frame, sift] = vl_covdet(I, 'method', 'Hessian', 'estimateAffineShape', true);
    sift = sift(:,(frame(1,:)<=x2) &  (frame(1,:) >= x1) & (frame(2,:) <= y2) & (frame(2,:) >= y1));
    
    % Test on an image
    q_words = cell(1,1);
    q_words{1} = ccvBowGetWords(dict_words, double(sift), [], dict);
    [ids dists] = ccvInvFileSearch(inv_file, q_words(1), if_weight, if_norm, if_dist, ntop);
    % visualize
    if verbose ==1
        close all;
        hold on; subplot(3,5,3); imshow(I);
        title(image_name(6:end));
    end
    fid = fopen('oxford\groundtruth\rank_list.txt', 'w');
    for i=1:size(ids{1},2)
        % Show only 10 highest score images
        if verbose==1 && i<=10
            subplot(3, 5, 5+i); 
            imshow(imread(fullfile('oxford\images\', files(ids{1}(i)).name)));
            title(files(ids{1}(i)).name);
        end
        fprintf(fid, '%s\n', files(ids{1}(i)).name(1:end-4));
    end
    fclose(fid);
    script = ['oxford\groundtruth\Test.exe oxford\groundtruth\', ...
        q_files(k).name(1:end-10), ...
        ' oxford\groundtruth\rank_list.txt',...
        ' >oxford\result\', image_name(6:end), '_result.txt']; %q_files(k).name(1:end-10)
    system(script);
    if verbose==1
        pause;
    end
end

r_files = dir(fullfile('.\result\', '*.txt'));
acc = [];
for i=1:length(r_files)
    file = ['.\result\' r_files(i).name];
    fid = fopen(file, 'r');
    acc = [acc fscanf(fid, '%f')];
    fclose(fid);
end
mean(acc)

% clear inv file
ccvInvFileClean(inv_file);