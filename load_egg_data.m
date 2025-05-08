% This script will read images from dir 'dir_path' and store them into a
% tensor X, one for training the model and one for testing.
classes = ["crack", "empty", "good"];

img_size = [120 160 3];

train_files = cell(1,3);
test_files = cell(1,3);
trainY = cell(1,3);
testY = cell(1,3);

for i = 1:3
    train_files{i} = dir(fullfile("dataset\train\", classes(i), '*.jpg'));
    test_files{i} = dir(fullfile("dataset\test\", classes(i), '*.jpg'));
    trainY{i} = i * ones(size(train_files{i}));
    testY{i} = i * ones(size(test_files{i}));
end
%test_files = vertcat
train_files = vertcat(train_files{:});
test_files = vertcat(test_files{:});
trainY = vertcat(trainY{:});
testY = vertcat(testY{:});

num_train_samples = length(trainY);
num_test_samples = length(testY);
trainX = tensor(zeros([img_size,num_train_samples]));
testX = tensor(zeros([img_size,num_test_samples]));
% now we read training samples ...
tic;
for i = 1:num_train_samples
    img = train_files(i);
    img = imresize(imread(fullfile(img.folder, img.name)), img_size(1:2));
    trainX(:,:,:,i) = double(img) / 255;
end
fprintf('Loading training data took ');
toc;
tic;
% ... and now testing.
for i = 1:num_test_samples
    img = test_files(i);
    img = imresize(imread(fullfile(img.folder, img.name)), img_size(1:2));
    testX(:,:,:,i) = double(img) / 255;
end
fprintf('loading testing data took ');
toc;

clear test_files train_files