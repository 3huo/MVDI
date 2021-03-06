% -------------------------------------------------------------------------
function net = prepareDINet(net,opts)
% -------------------------------------------------------------------------

drop6p = find(cellfun(@(a) strcmp(a.name, 'dropout6'), net.layers)==1);
drop7p = find(cellfun(@(a) strcmp(a.name, 'dropout7'), net.layers)==1);

if ~isempty(drop6p)
  assert(~isempty(drop7p));
  net.layers{drop6p}.rate = 0.5;
  net.layers{drop7p}.rate = 0.5;
else
  relu6p = find(cellfun(@(a) strcmp(a.name, 'relu6'), net.layers)==1);
  relu7p = find(cellfun(@(a) strcmp(a.name, 'relu7'), net.layers)==1);

  drop6 = struct('type','dropout','rate', 0.5,'name','dropout6') ;
  drop7 = struct('type','dropout','rate', 0.5,'name','dropout7') ;
  net.layers = [net.layers(1:relu6p) drop6 net.layers(relu6p+1:relu7p) drop7 net.layers(relu7p+1:end)];
end

% % replace fc8
fc8l = cellfun(@(a) strcmp(a.name, 'fc8'), net.layers)==1;

%%  note:ntu:60

nCls = 120;
sizeW = size(net.layers{fc8l}.weights{1});

if sizeW(4)~=nCls
  net.layers{fc8l}.weights = {zeros(sizeW(1),sizeW(2),sizeW(3),nCls,'single'), ...
    zeros(1, nCls, 'single')};
end

%%  change rgb 2 one single input
conv1 = find(cellfun(@(a) strcmp(a.name, 'conv1'), net.layers)==1);
weight_f = net.layers{conv1}.weights{1};

%----------------single
weights = mean(weight_f,3);
net.layers{conv1}.weights{1} = weights;
net.layers{conv1}.size = size(weights);

%----------------mutil
% weight = single(zeros(size(weight_f,1),size(weight_f,2),5,size(weight_f,4)));
% weight(:,:,1:3,:) = weight_f;weight(:,:,4:5,:) = weight_f(:,:,1:2,:);
% net.layers{conv1}.weights{1} = weight;
% net.layers{conv1}.size = size(weight);



% net.layers{conv1}.size(4) = [] ;
% net.layers{conv1}.pad(3) = [] ;

% net.layers{conv1}.weights{1} = squeeze(net.layers{conv1}.weights{1}) ;
% size(net.layers{conv1}.weights{1})


% [averageImage, rgbMean, rgbCovariance] = getImageStats(opts, net.meta, imdb) ;
% % �ı��С
net.meta.normalization.averageImage = [];%mean(net.meta.normalization.averageImage,3);
net.meta.normalization.imageSize = [net.meta.normalization.imageSize(1:2),1,net.meta.normalization.imageSize(end)];
% net.meta.normalization.imageSize(4) = 8;%[net.meta.normalization.imageSize(1:3)];


% change loss
net.layers{end} = struct('name','loss', 'type','softmaxloss') ;

% convert to dagnn
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;

net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
    {'prediction','label'}, 'top1err') ;
net.addLayer('top5err', dagnn.Loss('loss', 'topkerror', ...
    'opts', {'topK',5}), ...
    {'prediction','label'}, 'top5err') ;


% % configure appr-rank-pool
% if strcmp(opts.ARPoolLayer,'conv0')
%   net.addLayer('arpool',AppRankPooling(),{'input','VideoId1'},'DynImg');
% else
%   poolLyr1 = find(arrayfun(@(a) strcmp(a.name, opts.ARPoolLayer), net.layers)==1);
%   assert(~isempty(poolLyr1));
%   net.addLayer('arpool',AppRankPooling(),{net.layers(poolLyr1).inputs{1},'VideoId1'},'DynImg');
% end

% 
% if strcmp(opts.ARPoolLayer,'conv0')
%   l2param = [6e3 -128 128 0];
% elseif strcmp(opts.ARPoolLayer,'conv1') || strcmp(opts.ARPoolLayer,'relu1')
%   l2param = [8e4 -inf inf 0];
%   maxUpdateL2Norm = .02;
%   p = find(arrayfun(@(a) strcmp(a.params, 'conv1f'), net.layers)==1);
%   net.params(p).max_update_norm = maxUpdateL2Norm;
% elseif strcmp(opts.ARPoolLayer,'conv2') || strcmp(opts.ARPoolLayer,'relu2')
%   l2param =  [3e4 -inf inf 0] ;
%   maxUpdateL2Norm = .02;
%   p = find(arrayfun(@(a) strcmp(a.name, 'conv2f'), net.layers)==1);
%   net.params(p).max_update_norm = maxUpdateL2Norm;
% elseif strcmp(opts.ARPoolLayer,'conv3') || strcmp(opts.ARPoolLayer,'relu3')
%   l2param = [700 -inf inf 0];
%   maxUpdateL2Norm = .05;
%   p = find(arrayfun(@(a) strcmp(a.name, 'conv3f'), net.layers)==1);
%   net.params(p).max_update_norm = maxUpdateL2Norm;
% elseif strcmp(opts.ARPoolLayer,'conv4') || strcmp(opts.ARPoolLayer,'relu4')
%   l2param = [7e3 -inf inf 0];
%   maxUpdateL2Norm = .03;
%   p = find(arrayfun(@(a) strcmp(a.name, 'conv4f'), net.layers)==1);
%   net.params(p).max_update_norm = maxUpdateL2Norm;
% elseif strcmp(opts.ARPoolLayer,'conv5')
%   l2param = [6000 -inf inf 0] ;
%   maxUpdateL2Norm = .05;
%   p = find(arrayfun(@(a) strcmp(a.name, 'conv5f'), net.layers)==1);
%   net.params(p).max_update_norm = maxUpdateL2Norm;
% else
%    error('l2 normalization scale needs to be tuned for approximate rank');
% end
% 
% net.addLayer('l2norm',L2Normalize('scale',l2param(1),'clip',l2param(2:3),...
%   'offset',l2param(4)),'DynImg','DynImgN');
% 
% net.layers(1).inputs{1} = 'DynImgN';
% 
% 
% % second pool layer (max pooling)
% poolLyr2 = find(arrayfun(@(a) strcmp(a.name, 'pool5'), net.layers)==1);
% net.addLayer('tempPoolMax',TemporalPooling('method','max'),...
%   {net.layers(poolLyr2(1)).outputs{1},'VideoId2'},'tempPoolMax');
% 
% fc6 = find(arrayfun(@(a) strcmp(a.name, 'fc6'), net.layers)==1);
% net.layers(fc6(1)).inputs{1} = 'tempPoolMax';

% add multi-class error
% net.addLayer('errMC',ErrorMultiClass(),{'prediction','label'},'mcerr');

% net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
%   {'prediction','label'}, 'top1err') ;
