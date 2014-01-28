clear all; close all;
%% init parameter
addpath('AKM');
run('vlfeat\toolbox\vl_setup.m');
datasetDir = 'C:\oxford-images\';
isComputeSIFT = 0;
num_words = 1000000;
num_iterations = 5;
num_trees = 8;
dim = 128;

%% Compute SIFT features
if isComputeSIFT == 1
    fprintf('Computing SIFT features:\n');
    features = [];
    files = dir(fullfile(datasetDir, '*.jpg'));
    nfiles = length(files);
    features_per_image = zeros(1,nfiles);
    for i=1:nfiles
        i
        imgPath = strcat(datasetDir, files(i).name);
        I = im2single(rgb2gray(imread(imgPath)));
        [frame, sift] = vl_covdet(I, 'method', 'Hessian', 'estimateAffineShape', true);
        features = [features sift];
        features_per_image(i) = size(sift, 2);
    end
    fid = fopen('C:\oxford-feat\feature.bin', 'w');
    fwrite(fid, features, 'float');
    fclose(fid);
    
    save('C:\oxford-feat\feat_info.mat', 'features_per_image', 'files');
else
    fprintf('Loading SIFT features:\n');
    file = dir('C:\oxford-feat\feature.bin');
    %features = zeros(128, file.bytes/(4*128), 'single');

    fid = fopen('C:\oxford-feat\feature.bin', 'r');
    features = fread(fid, [128, file.bytes/(4*128)], 'float');
    fclose(fid);
    
    load('C:\oxford-feat\feat_info.mat');
end

%% load SIFT features (should be recomputed) -> bo do dung lai SIFT cua oxford cung cap
%fid = fopen('C:/oxford-feat/feat_oxc1_hesaff_sift.bin');
%data = zeros(16334970, 128, 'uint8');
%SIFT_feat = fread(fid, [128, 16334970], '*uint8');
%data = fread(fid, [16334970, 128], '*uint8'); %????
%fclose(fid);

%% compute rootSIFT
fprintf('Computing rootSIFT features:\n');
num_features = size(features, 2);
%rootSIFT = zeros(dim, num_features);

matlabpool('open',4);
for k = 1:5000000:num_features
    eIdx = k+5000000-1;
    if eIdx > num_features
        eIdx = num_features;
    end
    parfor i=k:eIdx
        features(:, i) = sqrt(features(:, i) / sum(features(:,i)));
    end
end
matlabpool close;

%% Run AKM to build dictionary
fprintf('Building the dictionary:\n');
num_images = length(files);
dict_params =  {num_iterations, 'kdt', num_trees};

% build the dictionary
if exist('C:\oxford-feat\dict.mat', 'file')
    load('C:\oxford-feat\dict.mat');
else
    dict_words = ccvBowGetDict(features, [], [], num_words, 'flat', 'akmeans', ...
        [], dict_params);
    save('dict.mat', 'dict_words');
end

% compute sparse frequency vector
fprintf('Computing the words\n');
dict = ccvBowGetWordsInit(dict_words, 'flat', 'akmeans', [], dict_params);

if exist('words.mat', 'file')
    load('words.mat');
else
    words = cell(1, num_images);
    for i=1:num_images   
        if i==1
            bIndex = 1;
        else
            bIndex = sum(features_per_image(1:i-1))+1;
        end
        eIndex = bIndex + features_per_image(i)-1;
        words{i} = ccvBowGetWords(dict_words, features(:, bIndex:eIndex), [], dict);
    end;
    save('words.mat', 'words');
end
%fprintf('Computing sparse frequency vector\n');
%dict = ccvBowGetWordsInit(dict_words, 'flat', 'akmeans', [], dict_params);
%words = ccvBowGetWords(dict_words, root_sift, [], dict);
%ccvBowGetWordsClean(dict);

% create an inverted file for the images
fprintf('Creating and searching an inverted file\n');
if exist('inverted_file.mat', 'file')
    load('inverted_file.mat');
else
    if_weight = 'tfidf';
    if_norm = 'l2';
    if_dist = 'l2';
    inv_file = ccvInvFileInsert([], words, num_words);
    ccvInvFileCompStats(inv_file, if_weight, if_norm);
    %save('inverted_file.mat', 'if_weight', 'if_norm', 'if_dist', 'inv_file');
end
%% Query images
verbose=0;
q_files = dir(fullfile('C:\oxford-groundtruth', '*query.txt'));
%oxc1_all_souls_000013 136.5 34.1 648.5 955.7
ntop = 200;
% load query image
for k=1:length(q_files)
    k
    fid = fopen(strcat('C:\oxford-groundtruth\', q_files(k).name), 'r');
    str = fgetl(fid);
    [image_name, remain] = strtok(str, ' ');
    fclose(fid);
    file = strcat('C:\oxford-images\', image_name(6:end), '.jpg');
    I = im2single(rgb2gray(imread(file)));
    % compute rootSIFT features
    [frame, sift] = vl_covdet(I, 'method', 'Hessian', 'estimateAffineShape', true);
    root_sift = zeros(size(sift));
    nfeat = size(sift, 2);
    for i=1:nfeat
        root_sift(:,i) = sqrt(sift(:,i) / sum(sift(:,i)));
    end
    % Test on an image
    q_words = cell(1,1);
    q_words{1} = ccvBowGetWords(dict_words, root_sift, [], dict);
    [ids dists] = ccvInvFileSearch(inv_file, q_words(1), if_weight, if_norm, if_dist, ntop);
    % visualize
    if verbose ==1
        close all;
        hold on; subplot(3,5,3); imshow(I);
        title(image_name(6:end));
    end
    fid = fopen('c:\oxford-groundtruth\rank_list.txt', 'w');
    for i=1:ntop
        % Show only 10 highest score image
        if verbose==1 && i<=10
            subplot(3, 5, 5+i); 
            imshow(imread(strcat('C:\oxford-images\', files(ids(i)).name)));
            title(files(ids(i)).name);
        end
        fprintf(fid, '%s\n', files(ids(i)).name(1:end-4));
    end
    fclose(fid);
    script = ['c:\oxford-groundtruth\Test.exe c:\oxford-groundtruth\', ...
        q_files(k).name(1:end-10), ...
        ' c:\oxford-groundtruth\rank_list.txt',...
        ' >result\', q_files(k).name(1:end-10), '_result.txt'];
    system(script);
end

q_files = dir(fullfile('.\result\', '*.txt'));
acc = [];
for i=1:length(q_files)
    file = ['.\result\' q_files(i).name];
    fid = fopen(file, 'r');
    acc = [acc fscanf(fid, '%f')];
    fclose(fid);
end
mean(acc)
% clear inv file
ccvInvFileClean(inv_file);