%  
% 模型为60类的训练模型--11个view-多multi
% 
% 保存特征到文件夹的版本
% 选择不同层的特征+PCA+进行liblinear实验
%
clc
clear

% run(fullfile('G:\chenjun\ntu_rgbd\dynamic','matconvnet-1.0-beta23\matconvnet-1.0-beta23', 'matlab', 'vl_setupnn.m')) ;

imdb = load('exp\ntu_multisubject_view1\imdb.mat') ;

opts.train.train = find(imdb.images.sets==1) ;
opts.train.val = find(imdb.images.sets==3) ;

% n_t = 1;
% for i = 1:length(opts.train.val)
%     i
%     name = imdb.images.name{opts.train.val(i)};
%     if strcmp(name(1:4),'S001')
%         index_test(n_t) = opts.train.val(i);
%         n_t = n_t + 1;
%     end
% end

index_test = opts.train.val;

file = 'feature_2_1\feature_ntu_subject_multi_MBB';

viewchoose = 1:11;
nn = 1;
for i = 1:length(index_test)
    index = index_test(i);
    pic_name = imdb.images.name{index};
    label = imdb.images.label(index);
    tic
    
    temp1 = load([file,'\',pic_name(1:end-4),'_softmax.mat']);
    temp1 = temp1.datapre;
    
    pre = zeros(60,1);
    for view = 1:length(viewchoose)
        t = viewchoose(view);
        pre_t = cell2mat(temp1(t,:));
        pre = pre + sum(pre_t,2);
    end
    [~,pre_f(i)] = max(pre);
    
    test_label(i) = label;
    nn = nn+1;
    
    acc = length(find(test_label==pre_f))/length(pre_f)*100;
    disp(['i = ',num2str(i),'; acc = ', num2str(acc)]); 
end
    
    