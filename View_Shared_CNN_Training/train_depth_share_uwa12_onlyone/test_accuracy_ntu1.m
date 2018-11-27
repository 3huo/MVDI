%%
% 整体性的测试准确率
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

length(opts.train.val)
for i = 1:length(opts.train.val)
    index = opts.train.val(i);
    label = imdb.images.label(index);
    im_ =  imread(fullfile(imdb.imageDir.test,imdb.images.name{index}));
    im_ = single(im_);
    im_ = imresize(im_, net1.meta.normalization.imageSize(1:2)) ;
    im_ = bsxfun(@minus, im_, net1.meta.normalization.averageImage) ;
    
    net1.eval({'input',im_}) ;
    scores = net1.vars(net1.getVarIndex('prob')).value ;
    scores = squeeze(gather(scores)) ;

    [bestScore, best] = max(scores) ;
    truth(b) = label;
    pre(b) = best;
%     plot(1:length(truth),truth,'*');
%     hold on;
%     plot(1:length(pre),pre,'o');
%     pause(0.01);
    b = b+1;
    accurcy = length(find(pre==truth))/length(truth);
    disp(['i = ',num2str(i),' accurcy = ',num2str(accurcy*100),'%']);
%     b
end 
plot(1:length(truth),truth,'*');
hold on;
plot(1:length(pre),pre,'o');
pause(0.01);

% 计算准确率
accurcy = length(find(pre==truth))/length(truth);
disp(['accurcy = ',num2str(accurcy*100),'%']);

% net.meta.classes.description{best}


