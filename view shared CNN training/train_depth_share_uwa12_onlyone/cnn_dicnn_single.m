function [net, info] = cnn_dicnn_single(varargin)
%CNN_DICNN Demonstrates fine-tuning a pre-trained CNN on UCF101 dataset

run(fullfile('F:\chenjun\dynamic','matconvnet-1.0-beta23\matconvnet-1.0-beta23', 'matlab', 'vl_setupnn.m')) ;

% addpath Layers Datasets

opts.dataDir = fullfile('data','UWA3D12') ;
% opts.expDir  = fullfile('exp', 'ntu_1+2') ;
opts.expDir  = fullfile('exp', 'uwa12_onlyone') ;

opts.modelPath = fullfile('models','imagenet-vgg-f.mat');
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.numFetchThreads = 12 ;

opts.lite = false ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');

opts.train = struct() ;
opts.train.gpus = [3] ;
opts.train.batchSize = 8*3 ;
opts.train.numSubBatches = 4 ;
opts.train.learningRate = 1e-4 * [ones(1,10), 0.1*ones(1,5)];

opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------
nCls = 30;
net = load(opts.modelPath);
net = prepareDINet(net,nCls,opts);
% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------

if exist(opts.imdbPath,'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = cnn_ntu_setup_data('dataDir', opts.dataDir, 'lite', opts.lite) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

imdb.images.set = imdb.images.sets;

% Set the class names in the network
net.meta.classes.name = imdb.classes.name ;
net.meta.classes.description = imdb.classes.name ;

% Compute image statistics (mean, RGB covariances, etc.)
imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
if exist(imageStatsPath)
    load(imageStatsPath, 'averageImage') ;
    opts.averageImage = averageImage ;
else
    opts.averageImage = [];
    averageImage = getImageStats(opts, net.meta, imdb) ;
    save(imageStatsPath, 'averageImage') ;
end
% % ����?% averageImage = getImageStats(opts, net.meta, imdb) ;
net.meta.normalization.averageImage = averageImage;
% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------
opts.train.train = find(imdb.images.set==1) ;
opts.train.val = find(imdb.images.set==3) ;

[net, info] = cnn_train_dag_dicnn(net, imdb, getBatchFn(opts, net.meta), ...
                      'expDir', opts.expDir, ...
                      opts.train) ;

% -------------------------------------------------------------------------
%                                                                    Deploy
% -------------------------------------------------------------------------

net = cnn_imagenet_deploy(net) ;
modelPath = fullfile(opts.expDir, 'net-deployed.mat');

net_ = net.saveobj() ;
save(modelPath, '-struct', 'net_') ;
clear net_ ;

% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------
useGpu = numel(opts.train.gpus) > 0 ;

bopts.numThreads = opts.numFetchThreads ;
bopts.imageSize = meta.normalization.imageSize ;
bopts.border = meta.normalization.border ;
bopts.averageImage = opts.averageImage ;
% bopts.averageImage = meta.normalization.averageImage ;

fn = @(x,y) getDagNNBatch(bopts,useGpu,x,y) ;


% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, useGpu, imdb, batch)
% -------------------------------------------------------------------------
nn = 1;
for i = 1:length(batch)
    name = imdb.images.name(batch(i));
    name = name{1};
    aa = randperm(5);
    for j = 1:11 %single
        indd = aa(1);
        name_t = [name(1:end-4),'_',num2str(j),'_',num2str(indd),name(end-3:end)];% ��ȡ5������
        if imdb.images.set(batch(i)) == 1 %1Ϊѵ�������ļ���
            images{nn} = strcat([imdb.imageDir.train filesep] , name_t);
        else
            images{nn} = strcat([imdb.imageDir.test filesep] , name_t);
        end
        nn = nn+1;
    end
end
% images = strcat([imdb.imageDir.train filesep] , imdb.images.name(batch)) ;
isVal = ~isempty(batch) && imdb.images.set(batch(1)) ~= 1 ;

if ~isVal
  % training
  im = cnn_imagenet_get_batch_single(images, opts, ...
                              'prefetch', nargout == 0) ;
else
  % validation: disable data augmentation
  im = cnn_imagenet_get_batch_single(images, opts, ...
                              'prefetch', nargout == 0, ...
                              'transformation', 'none') ;
end

if nargout > 0
  if useGpu
    im = gpuArray(im) ;
  end
  labels = imdb.images.label(batch) ;
  inputs = {'input', im, 'label', labels} ;
end
% -------------------------------------------------------------------------
function averageImage = getImageStats(opts, meta, imdb)
% -------------------------------------------------------------------------
train = find(imdb.images.set == 1) ;
batch = 1:length(train);
fn = getBatchFn(opts, meta) ;
train = train(1: 100: end);
avg = {};
for i = 1:length(train)-1
    disp(['calculate averageImage...',num2str(i/length(train)*100),'%']);
    temp = fn(imdb, batch(train(i):train(i)+99)) ;
    temp = temp{2};
%     temp1 = mean(temp,4);
    avg{end+1} = mean(temp,4); 
end
% z = reshape(permute(temp,[3 1 2 4]),3,[]) ;
% n = size(z,2) ;
averageImage = mean(cat(4,avg{:}),4) ;
averageImage = gather(averageImage) ;

