function imdb = cnn_ntu_setup_data(varargin)
 

%%
opts.dataDir = fullfile('data','UCLA_alldepth') ;
opts.lite = false ;
opts = vl_argparse(opts, varargin) ;

%% ------------------------------------------------------------------------
%                                                  Load categories metadata
% -------------------------------------------------------------------------

% find metadata
metaPath = fullfile(opts.dataDir, 'classInd.txt') ;

fprintf('using metadata %s\n', metaPath) ;
tmp = importdata(metaPath);
nCls = numel(tmp);

if nCls ~= 12
  error('Wrong meta file %s',metaPath);
end

cats = cell(1,nCls);
for i=1:numel(tmp)
  t = strsplit(tmp{i});
  cats{i} = t{2};
end

imdb.classes.name = cats ;
imdb.imageDir.train = fullfile(opts.dataDir, 'train') ;
imdb.imageDir.test = fullfile(opts.dataDir, 'test') ;

%% -----------------------------------------------------------------
%                                              load image names and labels
% -------------------------------------------------------------------------

name = {};
labels = {} ;
imdb.images.sets = [] ;
%%
fprintf('searching training images ...\n') ;

train_label_path = fullfile(opts.dataDir, 'train_label.txt') ;
train_label_temp = importdata(train_label_path);
temp_l = train_label_temp.data;
for i=1:numel(temp_l)
    train_label{i} = temp_l(i);
    train_img{i} = train_label_temp.textdata{i};
end

i = 1;
file_train = dir(fullfile(imdb.imageDir.train, '*.jpg'));
% if length(train_label) ~= length(file_train)/55 % 每一个视频55张dynamic image
%     error('training data is not equal to its label!!!');
% end

for d = 1:55:length(file_train)
    name{end+1} = [file_train(d).name(1:15),'.jpg'];%
    labels{end+1} = train_label{i} ;
    imdb.images.sets(end+1) = 1;%train

    if mod(numel(name), 10) == 0, fprintf('.') ; end
    if mod(numel(name), 500) == 0, fprintf('\n') ; end
    i = i+1;
end
%%
fprintf('searching testing images ...\n') ;

test_label_path = fullfile(opts.dataDir, 'test_label.txt') ;
test_label_temp = importdata(test_label_path);
temp_l = test_label_temp.data;
for i=1:numel(temp_l)
    test_label{i} = temp_l(i);
    test_img{i} = test_label_temp.textdata{i};
end

i = 1;
file_test = dir(fullfile(imdb.imageDir.test, '*.jpg'));
% if length(test_label) ~= length(file_test)/55 % 每一个视频55张dynamic image
%     error('testing data is not equal to its label!!!');
% end
for d = 1:55:length(file_test)
        name{end+1} = [file_test(d).name(1:15),'.jpg'];%
        labels{end+1} = test_label{i};%not a similar class
        imdb.images.sets(end+1) = 3;%test
    if mod(numel(name), 10) == 0, fprintf('.') ; end
    if mod(numel(name), 500) == 0, fprintf('\n') ; end
    i = i+1;
end

%%
labels = horzcat(labels{:}) ;

imdb.images.id = 1:numel(name) ;
imdb.images.name = name ;
imdb.images.label = labels ;




