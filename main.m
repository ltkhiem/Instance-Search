%% init parameter
addpath('AKM');
run('vlfeat\toolbox\vl_setup.m');

datasetDir = 'C:\oxford-images\';
isComputeSIFT = 1;
%% Compute SIFT features
if isComputeSIFT == 1
    features = [];
    files = dir(fullfile(datasetDir, '*.jpg'));
    nfiles = length(files);
    features_per_image = zeros(1,nfiles);
    for i=1:round(nfiles/2)
        i
        imgPath = strcat(datasetDir, files(i).name);
        I = im2single(rgb2gray(imread(imgPath)));
        [frame, sift] = vl_covdet(I, 'method', 'Hessian', 'estimateAffineShape', true);
        features = [features sift];
        features_per_image(i) = size(sift, 2);
    end
end
save('sift_feat.mat', 'features', 'features_per_image', 'files');
%% load SIFT features (should be recomputed)
fprintf('Loading SIFT features:\n');
fid = fopen('C:/oxford-feat/feat_oxc1_hesaff_sift.bin');
%data = zeros(16334970, 128, 'uint8');
SIFT_feat = fread(fid, [128, 16334970], '*uint8');
%data = fread(fid, [16334970, 128], '*uint8'); %????
fclose(fid);

%% compute rootSIFT
fprintf('Computing rootSIFT features:\n');
dim = 128;
num_features = 16334970;
rootSIFT = zeros(dim, num_features);
for i=1:num_features 
    rootSIFT(:, i) = sqrt(double(SIFT_feat(:, i)) / double(sum(SIFT_feat(:,i))));
end
clear SIFT_feat;

%% Run AKM to build dictionary
fprintf('Building the dictionary:\n');
num_images = 5062;
%labels = reshape(repmat(uint32(1:num_images), features_per_image, 1), [], 1)';

num_words = 1000000;
num_iterations = 5;
num_trees = 8;
dict_params =  {num_iterations, 'kdt', num_trees};

% build the dictionary
if exist('dict.mat', 'file')
    load('dict.mat');
else
    dict_words = ccvBowGetDict(rootSIFT, [], [], num_words, 'flat', 'akmeans', ...
        [], dict_params);
    save('dict.mat', 'dict_words');
end

%% Query images
%files = dir(fullfile('C:\oxford-groundtruth', '*query.txt'));
%oxc1_all_souls_000013 136.5 34.1 648.5 955.7
% load query image
file = strcat('C:\oxford-images\', 'all_souls_000013.jpg');
I = im2single(rgb2gray(imread(file)));
% compute rootSIFT features
[frame, sift] = vl_covdet(I, 'method', 'Hessian', 'estimateAffineShape', true);
root_sift = zeros(size(sift));
nfeat = size(sift, 2);
for i=1:nfeat
    root_sift(:,i) = sqrt(double(sift(:,i)) / double(sum(sift(:,i))));
end

% compute sparse frequency vector
fprintf('Computing sparse frequency vector\n');
dict = ccvBowGetWordsInit(dict_words, 'flat', 'akmeans', [], dict_params);
words = ccvBowGetWords(dict_words, root_sift, [], dict);
ccvBowGetWordsClean(dict);

% compute inverse document frequency
