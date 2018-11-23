 
clc
clear
net1 = dagnn.DagNN.loadobj(load('D:\myself\matlab\matlab_documents\dynamic_image\exp\ntu\net-deployed.mat')) ;
net1.mode = 'test' ;

imdb = load('D:\myself\matlab\matlab_documents\dynamic_image\exp\ntu\imdb.mat') ;
% imdb = imdb.imdb;
b = 1;

opts.dataDir = fullfile('data','ntu') ;
opts.expDir  = fullfile('exp', 'ntu') ;
opts.train.train = find(imdb.images.sets==1) ;
opts.train.val = find(imdb.images.sets==3) ;
opts.train.gpus = [];
opts.numFetchThreads = 12 ;
imdb.images.set = imdb.images.sets;

% Set the class names in the network
net.meta = net1.meta;
net.meta.classes.name = imdb.classes.name ;
net.meta.classes.description = imdb.classes.name ;

% Compute image statistics (mean, RGB covariances, etc.)
imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
if exist(imageStatsPath)
  load(imageStatsPath, 'averageImage') ;
else
    averageImage = getImageStats(opts, net.meta, imdb) ;
    save(imageStatsPath, 'averageImage') ;
end
net.meta.normalization.averageImage = averageImage;

for i = 1:length(opts.train.train)
    ind = opts.train.train(i);
    input =  getBatchFn_ntu(opts, net.meta, imdb, ind);
%     input{4} = 1;
    % input = state.getBatch(imdb, 3) ;
    net1.eval({'input',input{2}}) ;
    scores = net1.vars(net1.getVarIndex('prob')).value ;
    scores = squeeze(gather(scores)) ;

    [bestScore, best] = max(scores) ;
    truth(b) = input{4}+1;
    pre(b) = best;
    plot(1:length(truth),truth,'*');
    hold on;
    plot(1:length(pre),pre,'o');
    pause(0.01);
    b = b+1;
%     b
end
% net.meta.classes.description{best}


% 
% scores = net.vars(net.getVarIndex('prob')).value ;
% 
% 
% 
% a = inputs{2};
% for i = 1:200
% imshow(a(:,:,:,i));
% pause(0.05);
% end
% 
% 
% input{1} = inputs{1};
% input{2} = a;
% input{3} = inputs{5};
% input{4} = inputs{6};
% input{5} = inputs{7};
% input{6} = inputs{8};

