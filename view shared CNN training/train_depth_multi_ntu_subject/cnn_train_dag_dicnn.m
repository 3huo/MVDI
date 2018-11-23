% function [net_f,net_s,stats_f, stats_s] = cnn_train_dag_dicnn(net_f, net_s, imdb_f, imdb_s,getBatch,varargin)

function  [net_view1,net_view2,net_view3,net_view4,net_view5,...
    stats_view1, stats_view2,stats_view3,stats_view4,stats_view5]...
    = cnn_train_dag_dicnn(net_view1,net_view2,net_view3,net_view4,net_view5,...
    imdb_view1,imdb_view2,imdb_view3,imdb_view4,imdb_view5,getBatch,varargin)
                  
opts.expDir_view1  = fullfile('exp', 'ntu_multisubject_view1') ;
opts.expDir_view2  = fullfile('exp', 'ntu_multisubject_view2') ;
opts.expDir_view3  = fullfile('exp', 'ntu_multisubject_view3') ;
opts.expDir_view4  = fullfile('exp', 'ntu_multisubject_view4') ;
opts.expDir_view5  = fullfile('exp', 'ntu_multisubject_view5') ;

opts.continue = true ;
opts.batchSize = 256 ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.gpus = [1] ;
opts.prefetch = false ;
opts.numEpochs = 300 ;
opts.learningRate = 0.001 ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
opts.profile = false ;

opts.derOutputs = {'objective', 1} ;
opts.extractStatsFn = @extractStats ;
opts.plotStatistics = true;
opts = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir_view1, 'dir'), mkdir(opts.expDir_view1) ; end
if ~exist(opts.expDir_view2, 'dir'), mkdir(opts.expDir_view2) ; end
if ~exist(opts.expDir_view3, 'dir'), mkdir(opts.expDir_view3) ; end
if ~exist(opts.expDir_view4, 'dir'), mkdir(opts.expDir_view4) ; end
if ~exist(opts.expDir_view5, 'dir'), mkdir(opts.expDir_view5) ; end

if isempty(opts.train), opts.train = find(imdb_view1.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb_view1.images.set==3) ; end
if isnan(opts.train), opts.train = [] ; end

% -------------------------------------------------------------------------
%                                                            Initialization
% -------------------------------------------------------------------------

state.getBatch = getBatch ;
evaluateMode = isempty(opts.train) ;
if ~evaluateMode
  if isempty(opts.derOutputs)
    error('DEROUTPUTS must be specified when training.\n') ;
  end
end
stats = [] ;

% setup GPUs
numGpus = numel(opts.gpus) ;
if numGpus > 1
  if isempty(gcp('nocreate')),
    parpool('local',numGpus) ;
    spmd, gpuDevice(opts.gpus(labindex)), end
  end
  if exist(opts.memoryMapFile)
    delete(opts.memoryMapFile) ;
  end
elseif numGpus == 1
  gpuDevice(opts.gpus)
end

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------
modelPath_view1 = @(ep) fullfile(opts.expDir_view1, sprintf('net-epoch-%d.mat', ep));
modelPath_view2 = @(ep) fullfile(opts.expDir_view2, sprintf('net-epoch-%d.mat', ep));
modelPath_view3 = @(ep) fullfile(opts.expDir_view3, sprintf('net-epoch-%d.mat', ep));
modelPath_view4 = @(ep) fullfile(opts.expDir_view4, sprintf('net-epoch-%d.mat', ep));
modelPath_view5 = @(ep) fullfile(opts.expDir_view5, sprintf('net-epoch-%d.mat', ep));

% modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;

start = opts.continue * findLastCheckpoint(opts.expDir_view3) ;
if start >= 1
  fprintf('%s: resuming by loading epoch %d\n', mfilename, start) ;
  [net_view1, stats_view1] = loadState(modelPath_view1(start)) ;
  [net_view2, stats_view2] = loadState(modelPath_view2(start)) ;
  [net_view3, stats_view3] = loadState(modelPath_view3(start)) ;
  [net_view4, stats_view4] = loadState(modelPath_view4(start)) ;
  [net_view5, stats_view5] = loadState(modelPath_view5(start)) ;
end

for epoch=start+1:opts.numEpochs

  % train one epoch
  state.epoch = epoch ;
  state.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
  state.train = opts.train(randperm(numel(opts.train))) ; % shuffle
  state.val = opts.val ;
%   state.imdb = imdb ;

  if numGpus <= 1
    % change net
    % conv1-5 fc6f fc6b fc7f fc7b fc8f fc8b
    %   1-10   11  12   13   14   15   16
    %---------------------view1----------------------
%     for layer = 1:10
%         net_view1.params(layer).value = net_view5.params(layer).value;
%     end   
    state.imdb = imdb_view1 ;
    state.view_index = 1;
    stats_view1.train(epoch) = process_epoch(net_view1, state, opts, 'train') ;
    %---------------------view2----------------------
%     for layer = 1:10
%         net_view2.params(layer).value = net_view1.params(layer).value;
%     end      
    state.imdb = imdb_view2 ;
    state.view_index = 2;
    stats_view2.train(epoch) = process_epoch(net_view2, state, opts, 'train') ;
    %---------------------view3----------------------
%     for layer = 1:10
%         net_view3.params(layer).value = net_view2.params(layer).value;
%     end
    state.imdb = imdb_view3 ;
    state.view_index = 3;
    stats_view3.train(epoch) = process_epoch(net_view3, state, opts, 'train') ;
    %---------------------view4----------------------
%     for layer = 1:10
%         net_view4.params(layer).value = net_view3.params(layer).value;
%     end
    state.imdb = imdb_view4 ;
    state.view_index = 4;
    stats_view4.train(epoch) = process_epoch(net_view4, state, opts, 'train') ;
    %---------------------view5----------------------
%     for layer = 1:10
%         net_view5.params(layer).value = net_view4.params(layer).value;
%     end
    state.imdb = imdb_view5 ;
    state.view_index = 5;
    stats_view5.train(epoch) = process_epoch(net_view5, state, opts, 'train') ;
    %------------------- over train-----------------
    % test ------
    %-----------------------------------------------
    state.view_index = 1 ;
    state.imdb = imdb_view1 ;
    stats_view1.val(epoch) = process_epoch(net_view1, state, opts, 'val') ;
    %--------------------
    state.view_index = 2 ;
    state.imdb = imdb_view2 ;
    stats_view2.val(epoch) = process_epoch(net_view2, state, opts, 'val') ;
    %--------------------
    state.view_index = 3 ;
    state.imdb = imdb_view3 ;
    stats_view3.val(epoch) = process_epoch(net_view3, state, opts, 'val') ;
    %--------------------
    state.view_index = 4 ;
    state.imdb = imdb_view4 ;
    stats_view4.val(epoch) = process_epoch(net_view4, state, opts, 'val') ;
    %--------------------
    state.view_index = 5 ;
    state.imdb = imdb_view5 ;
    stats_view5.val(epoch) = process_epoch(net_view5, state, opts, 'val') ;
     %------------------- over test-----------------
  else
    error('something wrong in training');
  end

  if ~evaluateMode && (mod(epoch,10)==0)
    saveState(modelPath_view1(epoch), net_view1, stats_view1) ;
    saveState(modelPath_view2(epoch), net_view2, stats_view2) ;
    saveState(modelPath_view3(epoch), net_view3, stats_view3) ;
    saveState(modelPath_view4(epoch), net_view4, stats_view4) ;
    saveState(modelPath_view5(epoch), net_view5, stats_view5) ;
  end

  if opts.plotStatistics
    figure(1) ; clf ;
    plots = setdiff(...
      cat(2,...
      fieldnames(stats_view3.train)', ...
      fieldnames(stats_view3.val)'), {'num', 'time'}) ;
    for p = plots
      p = char(p) ;
      values = zeros(0, epoch) ;
      leg = {} ;
      for f = {'train', 'val'}
        f = char(f) ;
        if isfield(stats_view3.(f), p)
          tmp = [stats_view3.(f).(p)] ;
          values(end+1,:) = tmp(1,:)' ;
          leg{end+1} = f ;
        end
      end
      subplot(1,numel(plots),find(strcmp(p,plots))) ;
      plot(1:epoch, values','o-') ;
      xlabel('epoch') ;
      title(p) ;
      legend(leg{:}) ;
      grid on ;
    end
    drawnow ;
  %  print(1, modelFigPath, '-dpdf') ;
  end
end

% -------------------------------------------------------------------------
function stats = process_epoch(net, state, opts, mode)
% -------------------------------------------------------------------------

if strcmp(mode,'train')
  state.momentum = num2cell(zeros(1, numel(net.params))) ;
end

numGpus = numel(opts.gpus) ;
if numGpus >= 1
  net.move('gpu') ;
  if strcmp(mode,'train')
    state.momentum = cellfun(@gpuArray,state.momentum,'UniformOutput',false) ;
  end
end
if numGpus > 1
  mmap = map_gradients(opts.memoryMapFile, net, numGpus) ;
else
  mmap = [] ;
end

stats.time = 0 ;
stats.num = 0 ;
subset = state.(mode) ;
start = tic ;
num = 0 ;

for t=1:opts.batchSize:numel(subset)
  batchSize = min(opts.batchSize, numel(subset) - t + 1) ;

  for s=1:opts.numSubBatches
    % get this image batch and prefetch the next
    batchStart = t + (labindex-1) + (s-1) * numlabs ;
    batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
    batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
    num = num + numel(batch) ;
    if numel(batch) == 0, continue ; end
%��ȡ��Ƶ�Ĳ���
    inputs = state.getBatch(state.imdb,state.view_index,batch,net.meta.normalization.averageImage) ;

    if opts.prefetch
      if s == opts.numSubBatches
        batchStart = t + (labindex-1) + opts.batchSize ;
        batchEnd = min(t+2*opts.batchSize-1, numel(subset)) ;
      else
        batchStart = batchStart + numlabs ;
      end
      nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
      state.getBatch(state.imdb, state.view_index, nextBatch,net.meta.normalization.averageImage) ;
    end

    if strcmp(mode, 'train')
      net.mode = 'normal' ;
      net.accumulateParamDers = (s ~= 1) ;
      net.eval(inputs, opts.derOutputs) ;
    else
      net.mode = 'test' ;
      net.eval(inputs) ;
    end
  end

  % extract learning stats
  stats = opts.extractStatsFn(net) ;

  % accumulate gradient
  if strcmp(mode, 'train')
    if ~isempty(mmap)
      write_gradients(mmap, net) ;
      labBarrier() ;
    end
    state = accumulate_gradients(state, net, opts, batchSize, mmap) ;
  end

  % print learning statistics
  time = toc(start) ;
  stats.num = num ;
  stats.time = toc(start) ;

  fprintf('%s: epoch %02d: %3d/%3d: %.1f Hz', ...
    mode, ...
    state.epoch, ...
    fix(t/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize), ...
    stats.num/stats.time * max(numGpus, 1)) ;
  for f = setdiff(fieldnames(stats)', {'num', 'time'})
    f = char(f) ;
    fprintf(' %s:', f) ;
    fprintf(' %.3f', stats.(f)) ;
  end
  fprintf('\n') ;
end

net.reset() ;
net.move('cpu') ;

% -------------------------------------------------------------------------
function state = accumulate_gradients(state, net, opts, batchSize, mmap)
% -------------------------------------------------------------------------
for p=1:numel(net.params)

  % bring in gradients from other GPUs if any
  if ~isempty(mmap)
    numGpus = numel(mmap.Data) ;
    tmp = zeros(size(mmap.Data(labindex).(net.params(p).name)), 'single') ;
    for g = setdiff(1:numGpus, labindex)
      tmp = tmp + mmap.Data(g).(net.params(p).name) ;
    end
    net.params(p).der = net.params(p).der + tmp ;
  else
    numGpus = 1 ;
  end

  switch net.params(p).trainMethod

    case 'average' % mainly for batch normalization
      thisLR = net.params(p).learningRate ;
      net.params(p).value = ...
          (1 - thisLR) * net.params(p).value + ...
          (thisLR/batchSize/net.params(p).fanout) * net.params(p).der ;

    case 'gradient'
      thisDecay = opts.weightDecay * net.params(p).weightDecay ;
      thisLR = state.learningRate * net.params(p).learningRate ;
      state.momentum{p} = opts.momentum * state.momentum{p} ...
        - thisDecay * net.params(p).value ...
        - (1 / batchSize) * net.params(p).der ;
      net.params(p).value = net.params(p).value + thisLR * state.momentum{p} ;

    case 'otherwise'
      error('Unknown training method ''%s'' for parameter ''%s''.', ...
        net.params(p).trainMethod, ...
        net.params(p).name) ;
  end
end

% -------------------------------------------------------------------------
function mmap = map_gradients(fname, net, numGpus)
% -------------------------------------------------------------------------
format = {} ;
for i=1:numel(net.params)
  format(end+1,1:3) = {'single', size(net.params(i).value), net.params(i).name} ;
end
format(end+1,1:3) = {'double', [3 1], 'errors'} ;
if ~exist(fname) && (labindex == 1)
  f = fopen(fname,'wb') ;
  for g=1:numGpus
    for i=1:size(format,1)
      fwrite(f,zeros(format{i,2},format{i,1}),format{i,1}) ;
    end
  end
  fclose(f) ;
end
labBarrier() ;
mmap = memmapfile(fname, 'Format', format, 'Repeat', numGpus, 'Writable', true) ;

% -------------------------------------------------------------------------
function write_gradients(mmap, net)
% -------------------------------------------------------------------------
for i=1:numel(net.params)
  mmap.Data(labindex).(net.params(i).name) = gather(net.params(i).der) ;
end

% -------------------------------------------------------------------------
function stats = accumulateStats(stats_)
% -------------------------------------------------------------------------

stats = struct() ;

for s = {'train', 'val'}
  s = char(s) ;
  total = 0 ;

  for g = 1:numel(stats_)
    stats__ = stats_{g} ;
    num__ = stats__.(s).num ;
    total = total + num__ ;

    for f = setdiff(fieldnames(stats__.(s))', 'num')
      f = char(f) ;

      if g == 1
        stats.(s).(f) = 0 ;
      end
      stats.(s).(f) = stats.(s).(f) + stats__.(s).(f) * num__ ;

      if g == numel(stats_)
        stats.(s).(f) = stats.(s).(f) / total ;
      end
    end
  end
  stats.(s).num = total ;
end

% -------------------------------------------------------------------------
function stats = extractStats(net)
% -------------------------------------------------------------------------
sel = find(cellfun(@(x) isa(x,'dagnn.Loss') || isa(x,'ErrorMultiClass'), ...
  {net.layers.block})) ;
stats = struct() ;
for i = 1:numel(sel)
  stats.(net.layers(sel(i)).outputs{1}) = net.layers(sel(i)).block.average ;
end

% -------------------------------------------------------------------------
function saveState(fileName, net, stats)
% -------------------------------------------------------------------------
net_ = net ;
net = net_.saveobj() ;
save(fileName, 'net', 'stats') ;

% -------------------------------------------------------------------------
function [net, stats] = loadState(fileName)
% -------------------------------------------------------------------------
load(fileName, 'net', 'stats') ;
net = dagnn.DagNN.loadobj(net) ;

% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;
