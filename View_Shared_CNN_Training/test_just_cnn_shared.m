%  
% 模型为60类的训练模型--11个view-多multi
% 
% 保存特征到文件夹的版本
% 选择不同层的特征+PCA+进行liblinear实验
%
clc
clear

run(fullfile('F:\chenjun\dynamic','matconvnet-1.0-beta23\matconvnet-1.0-beta23', 'matlab', 'vl_setupnn.m')) ;

net_view1 = dagnn.DagNN.loadobj(load('F:\chenjun\dynamic\exp\uwa_share_view134\net-deployed-view1.mat')) ;
net_view1.mode = 'test' ;
net_view2 = dagnn.DagNN.loadobj(load('F:\chenjun\dynamic\exp\uwa_share_view234\net-deployed-view2.mat')) ;
net_view2.mode = 'test' ;
net_view3 = dagnn.DagNN.loadobj(load('F:\chenjun\dynamic\exp\uwa_share_view334\net-deployed-view3.mat')) ;
net_view3.mode = 'test' ;
net_view4 = dagnn.DagNN.loadobj(load('F:\chenjun\dynamic\exp\uwa_share_view434\net-deployed-view4.mat')) ;
net_view4.mode = 'test' ;
net_view5 = dagnn.DagNN.loadobj(load('F:\chenjun\dynamic\exp\uwa_share_view534\net-deployed-view5.mat')) ;
net_view5.mode = 'test' ;

%all view is the same imdb
imdb = load('F:\chenjun\dynamic\exp\uwa_share_view134\imdb.mat') ;

b = 1;

opts.train.train = find(imdb.images.sets==1) ;
opts.train.val = find(imdb.images.sets==3) ;

% net_view1.conserveMemory = 0;   %将特征都显示出来
% net_view2.conserveMemory = 0;   %将特征都显示出来
% net_view3.conserveMemory = 0;   %将特征都显示出来
% net_view4.conserveMemory = 0;   %将特征都显示出来
% net_view5.conserveMemory = 0;   %将特征都显示出来

length(opts.train.train)

% n_t = 1;
% for i = 1:length(opts.train.train)
%     i
%     name = imdb.images.name{opts.train.train(i)};
%     if strcmp(name(1:4),'S001')
%         index_train(n_t) = opts.train.train(i);
%         n_t = n_t + 1;
%     end
% end
% 
% n_t = 1;
% for i = 1:length(opts.train.val)
%     i
%     name = imdb.images.name{opts.train.val(i)};
%     if strcmp(name(1:4),'S001')
%         index_test(n_t) = opts.train.val(i);
%         n_t = n_t + 1;
%     end
% end

index_train = opts.train.train;
% 
index_test = opts.train.val;

% load('result_lab_shared.mat');
length(index_test)
nn = 1;
for i = 1:length(index_test)
    i
    index = index_test(i);
    pic_name = imdb.images.name{index};
    label = imdb.images.label(index);
    tic
    %     if ~exist(['feature_11view_mutil1\',pic_name(1:end-4),'.mat'],'file')
    n = 1;
    pre_tem = [];
    for view = 1:11
        data1 = [];
        data2 = [];
        for j = 1:5
            im_ =  imread(fullfile('F:\chenjun\dynamic\data\dynamic_multi_uwa3d_MBB',[pic_name(1:end-4),'_',num2str(view),'_',num2str(j),'.jpg']));
            im_ = single(im_);
            if view == 1 || view == 2
                net_temp = net_view1;
            elseif view == 3 || view == 4 || view == 5
                net_temp = net_view2;
            elseif view == 6
                net_temp = net_view3;
            elseif view == 7 || view == 8 || view == 9
                net_temp = net_view4;
            elseif view == 10 || view == 11
                net_temp = net_view5;
            end
            im_ = imresize(im_, net_temp.meta.normalization.imageSize(1:2)) ;
            im_ = bsxfun(@minus, im_, net_temp.meta.normalization.averageImage) ;
            
            net_temp.eval({'input',im_}) ;
            
            data11 = net_temp.vars(net_temp.getVarIndex('prob')).value ;
            data11 = squeeze(gather(data11));
            
            pre{i,n} = data11;
            n = n+1;
            
            clear data11
        end
        
    end
    pre_t = cell2mat(pre(i,1:55));
    [~,pre_f(i)] = max(sum(pre_t,2));   
%     end
    
    test_label(i) = label;
    nn = nn+1;
    
    acc = length(find(test_label==pre_f))/length(pre_f)*100
    
%     if ~mod(i,100)
%         save result_lab_shared pre test_label
%     end
end
acc = length(find(test_label==pre_f))/length(pre_f)*100

