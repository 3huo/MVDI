function [net, info] = cnn_dicnn(varargin)
%CNN_DICNN Demonstrates fine-tuning a pre-trained CNN on UCF101 dataset


run(fullfile('F:\chenjun\dynamic','matconvnet-1.0-beta23\matconvnet-1.0-beta23', 'matlab', 'vl_setupnn.m')) ;

% addpath Layers Datasets

opts.dataDir = fullfile('data','ntu_subject_MBB') ;
opts.expDir_view  = fullfile('exp', 'ntu_subject_onlyone') ;
% opts.expDir_view2  = fullfile('exp', 'ntu_sharesubject_view2') ;
% opts.expDir_view3  = fullfile('exp', 'ntu_sharesubject_view3') ;
% opts.expDir_view4  = fullfile('exp', 'ntu_sharesubject_view4') ;
% opts.expDir_view5  = fullfile('exp', 'ntu_sharesubject_view5') ;

opts.modelPath = fullfile('models','imagenet-vgg-f.mat');
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.numFetchThreads = 12 ;

opts.lite = false ;

opts.imdbPath_view = fullfile(opts.expDir_view, 'imdb.mat');
% opts.imdbPath_view2 = fullfile(opts.expDir_view2, 'imdb.mat');
% opts.imdbPath_view3 = fullfile(opts.expDir_view3, 'imdb.mat');
% opts.imdbPath_view4 = fullfile(opts.expDir_view4, 'imdb.mat');
% opts.imdbPath_view5 = fullfile(opts.expDir_view5, 'imdb.mat');

opts.train = struct() ;
opts.train.gpus = [1];
opts.train.batchSize = 8*3 ;
opts.train.numSubBatches = 4 ;
opts.train.learningRate = 1e-4 * [ones(1,10), 0.1*ones(1,5)];

opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------
%  note:ntu:60  UWA3D£º30  UCLA:12 
nCls = 60;
net_view = load(opts.modelPath);
net_view = prepareDINet(net_view,nCls,opts);
% net_view2 = load(opts.modelPath);
% net_view2 = prepareDINet(net_view2,nCls,opts);
% net_view3 = load(opts.modelPath);
% net_view3 = prepareDINet(net_view3,nCls,opts);
% net_view4 = load(opts.modelPath);
% net_view4 = prepareDINet(net_view4,nCls,opts);
% net_view5 = load(opts.modelPath);
% net_view5 = prepareDINet(net_view5,nCls,opts);

% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------

if exist(opts.imdbPath_view,'file')
  imdb_view = load(opts.imdbPath_view) ;
%   imdb_view2 = load(opts.imdbPath_view2) ;
%   imdb_view3 = load(opts.imdbPath_view3) ;
%   imdb_view4 = load(opts.imdbPath_view4) ;
%   imdb_view5 = load(opts.imdbPath_view5) ;
else 
  imdb = cnn_ntu_setup_data('dataDir', opts.dataDir, 'lite', opts.lite) ;  
  mkdir(opts.expDir_view) ;
  save(opts.imdbPath_view, '-struct', 'imdb') ;
%   mkdir(opts.expDir_view2) ;
%   save(opts.imdbPath_view2, '-struct', 'imdb') ;
%   mkdir(opts.expDir_view3) ;
%   save(opts.imdbPath_view3, '-struct', 'imdb') ;
%   mkdir(opts.expDir_view4) ;
%   save(opts.imdbPath_view4, '-struct', 'imdb') ;
%   mkdir(opts.expDir_view5) ;
%   save(opts.imdbPath_view5, '-struct', 'imdb') ;
end

imdb_view = load(opts.imdbPath_view) ;
% imdb_view2 = load(opts.imdbPath_view2) ;
% imdb_view3 = load(opts.imdbPath_view3) ;
% imdb_view4 = load(opts.imdbPath_view4) ;
% imdb_view5 = load(opts.imdbPath_view5) ;
  
imdb_view.images.set = imdb_view.images.sets;
% imdb_view2.images.set = imdb_view2.images.sets;
% imdb_view3.images.set = imdb_view3.images.sets;
% imdb_view4.images.set = imdb_view4.images.sets;
% imdb_view5.images.set = imdb_view5.images.sets;

% Set the class names in the network
net_view.meta.classes.name = imdb_view.classes.name ;
net_view.meta.classes.description = imdb_view.classes.name ;
% net_view2.meta.classes.name = imdb_view2.classes.name ;
% net_view2.meta.classes.description = imdb_view2.classes.name ;
% net_view3.meta.classes.name = imdb_view3.classes.name ;
% net_view3.meta.classes.description = imdb_view3.classes.name ;
% net_view4.meta.classes.name = imdb_view4.classes.name ;
% net_view4.meta.classes.description = imdb_view4.classes.name ;
% net_view5.meta.classes.name = imdb_view5.classes.name ;
% net_view5.meta.classes.description = imdb_view5.classes.name ;

% Compute image statistics (mean, RGB covariances, etc.)
imageStatsPath_view = fullfile(opts.expDir_view, 'imageStats_view.mat') ;
% imageStatsPath_view2 = fullfile(opts.expDir_view2, 'imageStats_view2.mat') ;
% imageStatsPath_view3 = fullfile(opts.expDir_view3, 'imageStats_view3.mat') ;
% imageStatsPath_view4 = fullfile(opts.expDir_view4, 'imageStats_view4.mat') ;
% imageStatsPath_view5 = fullfile(opts.expDir_view5, 'imageStats_view5.mat') ;

if exist(imageStatsPath_view1)&& exist(imageStatsPath_view2)&&exist(imageStatsPath_view3)...
        &&exist(imageStatsPath_view4)&&exist(imageStatsPath_view5)
    load(imageStatsPath_view1, 'averageImage_view1') ;
    load(imageStatsPath_view2, 'averageImage_view2') ;
    load(imageStatsPath_view3, 'averageImage_view3') ;
    load(imageStatsPath_view4, 'averageImage_view4') ;
    load(imageStatsPath_view5, 'averageImage_view5') ;
    
else
    [averageImage_view1,averageImage_view2,averageImage_view3,...
       averageImage_view4,averageImage_view5] = getImageStats(opts, net_view1.meta, imdb_view1) ;
    save(imageStatsPath_view1, 'averageImage_view1') ;
    save(imageStatsPath_view2, 'averageImage_view2') ;
    save(imageStatsPath_view3, 'averageImage_view3') ;
    save(imageStatsPath_view4, 'averageImage_view4') ;
    save(imageStatsPath_view5, 'averageImage_view5') ;
end
net_view1.meta.normalization.averageImage = averageImage_view1;
net_view2.meta.normalization.averageImage = averageImage_view2;
net_view3.meta.normalization.averageImage = averageImage_view3;
net_view4.meta.normalization.averageImage = averageImage_view4;
net_view5.meta.normalization.averageImage = averageImage_view5;

% param_all.net1.average = averageImage_view1;
% param_all.net2.average = averageImage_view2;
% param_all.net3.average = averageImage_view3;
% param_all.net4.average = averageImage_view4;
% param_all.net5.average = averageImage_view5;
% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------
opts.train.train = find(imdb_view1.images.set==1) ;
opts.train.val = find(imdb_view1.images.set==3) ;

[net_view1,net_view2,net_view3,net_view4,net_view5,...
    stats_view1, stats_view2,stats_view3,stats_view4,stats_view5]...
    = cnn_train_dag_dicnn(net_view1,net_view2,net_view3,net_view4,net_view5,...
    imdb_view1,imdb_view2,imdb_view3,imdb_view4,imdb_view5,...
    getBatchFn(opts, net_view1.meta),...
    'expDir_view1',opts.expDir_view1,'expDir_view2',opts.expDir_view2,'expDir_view3',opts.expDir_view3, ...
    'expDir_view4',opts.expDir_view4,'expDir_view5',opts.expDir_view5,...
                      opts.train) ;

% -------------------------------------------------------------------------
%                                                                    Deploy
% -------------------------------------------------------------------------

%---------------1--------------
% for layer = 11:16
%     net_view1.params(layer).value = param_all.net1.layer(layer).value;
% end
% net_view1.meta.normalization.averageImage = param_all.net1.average;

net_view1 = cnn_imagenet_deploy(net_view1) ;
modelPath = fullfile(opts.expDir_view1, 'net-deployed-view1.mat');
net_ = net_view1.saveobj() ;
save(modelPath, '-struct', 'net_') ;
clear net_ ;
%-------------2----------------
% for layer = 11:16
%     net_view2.params(layer).value = param_all.net2.layer(layer).value;
% end
% net_view2.meta.normalization.averageImage = param_all.net2.average;

net_view2 = cnn_imagenet_deploy(net_view2) ;
modelPath = fullfile(opts.expDir_view2, 'net-deployed-view2.mat');
net_ = net_view2.saveobj() ;
save(modelPath, '-struct', 'net_') ;
clear net_ ;
%--------------3---------------
% for layer = 11:16
%     net_view3.params(layer).value = param_all.net3.layer(layer).value;
% end
% net_view3.meta.normalization.averageImage = param_all.net3.average;

net_view3 = cnn_imagenet_deploy(net_view3) ;
modelPath = fullfile(opts.expDir_view3, 'net-deployed-view3.mat');
net_ = net_view3.saveobj() ;
save(modelPath, '-struct', 'net_') ;
clear net_ ;
%--------------4---------------
% for layer = 11:16
%     net_view4.params(layer).value = param_all.net4.layer(layer).value;
% end
% net_view4.meta.normalization.averageImage = param_all.net4.average;

net_view4 = cnn_imagenet_deploy(net_view4) ;
modelPath = fullfile(opts.expDir_view4, 'net-deployed-view4.mat');
net_ = net_view4.saveobj() ;
save(modelPath, '-struct', 'net_') ;
clear net_ ;
%-----------------------------
% for layer = 11:16
%     net_view5.params(layer).value = param_all.net5.layer(layer).value;
% end
% net_view5.meta.normalization.averageImage = param_all.net5.average;

net_view5 = cnn_imagenet_deploy(net_view5) ;
modelPath = fullfile(opts.expDir_view5, 'net-deployed-view5.mat');
net_ = net_view5.saveobj() ;
save(modelPath, '-struct', 'net_') ;
clear net_ ;


% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------
useGpu = numel(opts.train.gpus) > 0 ;

bopts.numThreads = opts.numFetchThreads ;
bopts.imageSize = meta.normalization.imageSize ;
bopts.border = meta.normalization.border ;
% bopts.averageImage = []; 
bopts.averageImage = meta.normalization.averageImage ;
% bopts.rgbVariance = meta.augmentation.rgbVariance ;
% bopts.transformation = meta.augmentation.transformation ;


fn = @(x,y,z,aveimg) getDagNNBatch(bopts,useGpu,x,y,z,aveimg) ;


% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, useGpu, imdb, index, batch,aveimg)
% -------------------------------------------------------------------------
opts.averageImage = aveimg;
nn = 1;
for i = 1:length(batch)
    name = imdb.images.name(batch(i));
    name = name{1};
    if index == 1
        name_t = {[name(1:end-4),'_1_1.jpg'],[name(1:end-4),'_2_1.jpg']};
        cc = randperm(length(name_t));
        cc = cc(1);
        name_t1 = name_t{cc};
    elseif index == 2
        name_t = {[name(1:end-4),'_3_1.jpg'],[name(1:end-4),'_4_1.jpg'],[name(1:end-4),'_5_1.jpg']};
        cc = randperm(length(name_t));
        cc = cc(1);
        name_t1 = name_t{cc};
    elseif index == 3
        name_t = {[name(1:end-4),'_6_1.jpg']};
        cc = randperm(length(name_t));
        cc = cc(1);
        name_t1 = name_t{cc};
    elseif index == 4
        name_t = {[name(1:end-4),'_7_1.jpg'],[name(1:end-4),'_8_1.jpg'],[name(1:end-4),'_9_1.jpg']};
        cc = randperm(length(name_t));
        cc = cc(1);
        name_t1 = name_t{cc};
    elseif index == 5
        name_t = {[name(1:end-4),'_10_1.jpg'],[name(1:end-4),'_11_1.jpg']};
        cc = randperm(length(name_t));
        cc = cc(1);
        name_t1 = name_t{cc};
    else
        error('wrong choose');
    end
    if imdb.images.set(batch(i)) == 1 %1ÎªÑµï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Ä¼ï¿½ï¿½ï¿½
        images{nn} = strcat([imdb.imageDir.train filesep] , name_t1);
    else
        images{nn} = strcat([imdb.imageDir.test filesep] , name_t1);
    end
    nn = nn+1;
end;
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
function [averageImage_view1,averageImage_view2,averageImage_view3,...
       averageImage_view4,averageImage_view5] = getImageStats(opts, meta, imdb)
% -------------------------------------------------------------------------
train = find(imdb.images.set == 1) ;
batch = 1:length(train);
fn = getBatchFn(opts, meta) ;
train = train(1:100:end);
aveimg = [];

for index = 1:5
    avg = {};
    for i = 1:length(train)-2
        disp(['calculate averageImage...',num2str(i/length(train)*100),'%']);
        temp = fn(imdb,index,batch(train(i):train(i)+99),aveimg) ;
        temp = temp{2};
        %     temp1 = mean(temp,4);
        avg{end+1} = mean(temp,4);
    end
    averageImage{index} = mean(cat(4,avg{:}),4) ;
end
averageImage_view1 = gather(averageImage{1});
averageImage_view2 = gather(averageImage{2});
averageImage_view3 = gather(averageImage{3});
averageImage_view4 = gather(averageImage{4});
averageImage_view5 = gather(averageImage{5});

