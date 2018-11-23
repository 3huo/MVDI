% -------------------------------------------------------------------------
function net = prepareDINet(net,nCls,opts)
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
%% nCls note:  ntu:60   UWA3D：30  UCLA:12
sizeW = size(net.layers{fc8l}.weights{1});
% 修改输出类别书
if sizeW(4)~=nCls
  net.layers{fc8l}.weights = {zeros(sizeW(1),sizeW(2),sizeW(3),nCls,'single'), ...
    zeros(1, nCls, 'single')};
end
%%  change rgb 2 one single input
conv1 = find(cellfun(@(a) strcmp(a.name, 'conv1'), net.layers)==1);
weight_f = net.layers{conv1}.weights{1};
% 修改三通道卷积变为单通道
%----------------single
weights = mean(weight_f,3);
net.layers{conv1}.weights{1} = weights;
net.layers{conv1}.size = size(weights);
% 修改图像的大小
net.meta.normalization.imageSize = [net.meta.normalization.imageSize(1:2),1,net.meta.normalization.imageSize(end)];
% change loss
net.layers{end} = struct('name','loss', 'type','softmaxloss') ;
% convert to dagnn
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;

net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
    {'prediction','label'}, 'top1err') ;
net.addLayer('top5err', dagnn.Loss('loss', 'topkerror', ...
    'opts', {'topK',5}), ...
    {'prediction','label'}, 'top5err') ;
