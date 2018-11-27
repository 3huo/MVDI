%  
% 模型为60类的训练模型--11个view-多multi
% 
% 保存特征到文件夹的版本
% 选择不同层的特征+PCA+进行liblinear实验
%
clc
clear

run(fullfile('F:\chenjun\dynamic','matconvnet-1.0-beta23\matconvnet-1.0-beta23', 'matlab', 'vl_setupnn.m')) ;

net_view1 = dagnn.DagNN.loadobj(load('F:\chenjun\dynamic\exp\ntu_DMM_shareview_view1\net-deployed-view1.mat')) ;
net_view1.mode = 'test' ;
net_view2 = dagnn.DagNN.loadobj(load('F:\chenjun\dynamic\exp\ntu_DMM_shareview_view2\net-deployed-view2.mat')) ;
net_view2.mode = 'test' ;
net_view3 = dagnn.DagNN.loadobj(load('F:\chenjun\dynamic\exp\ntu_DMM_shareview_view3\net-deployed-view3.mat')) ;
net_view3.mode = 'test' ;
net_view4 = dagnn.DagNN.loadobj(load('F:\chenjun\dynamic\exp\ntu_DMM_shareview_view4\net-deployed-view4.mat')) ;
net_view4.mode = 'test' ;
net_view5 = dagnn.DagNN.loadobj(load('F:\chenjun\dynamic\exp\ntu_DMM_shareview_view5\net-deployed-view5.mat')) ;
net_view5.mode = 'test' ;

%all view is the same imdb
imdb = load('F:\chenjun\dynamic\exp\ntu_DMM_shareview_view1\imdb.mat') ;

b = 1;

opts.train.train = find(imdb.images.sets==1) ;
opts.train.val = find(imdb.images.sets==3) ;

net_view1.conserveMemory = 0;   %将特征都显示出来
net_view2.conserveMemory = 0;   %将特征都显示出来
net_view3.conserveMemory = 0;   %将特征都显示出来
net_view4.conserveMemory = 0;   %将特征都显示出来
net_view5.conserveMemory = 0;   %将特征都显示出来

% length(opts.train.train)
% 
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

index_test = opts.train.val;

nn = 1;
for i = 1:length(index_train)
    i
    index = index_train(i);
    pic_name = imdb.images.name{index};
    label = imdb.images.label(index);
    
    if ~exist(['feature_ntu_view_DMM_s001\',pic_name(1:end-4),'.mat'],'file')  
%         data1 = [];
%         data2 = [];
%         data3 = [];
%         data4 = []; 
        datapre = [];
        for view = 1:11
            data1 = [];
            data2 = [];
            for j = 1:5
                im_ =  imread(fullfile('F:\chenjun\dynamic\data\ntu_view_DMM_MBB_s001\train\',[pic_name(1:end-4),'_',num2str(view),'_',num2str(j),'.jpg']));
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
                
                data11 = net_temp.vars(net_temp.getVarIndex('x16')).value ;
                data11 = squeeze(gather(data11));
                data22 = net_temp.vars(net_temp.getVarIndex('x19')).value ;
                data22 = squeeze(gather(data22));
                
                temp = net_temp.vars(net_temp.getVarIndex('prob')).value ;
                temp = squeeze(gather(temp));
                datapre{view,j} = temp;
                
                data1 = [data1;data11(:)];
                data2 = [data2;data22(:)];
                clear data11 data22 temp
            end
            data{view,1} = data1;%fc6
            data{view,2} = data2;
            clear data1 data2
        end
        save(['feature_ntu_view_DMM_s001\',pic_name(1:end-4),'.mat'],'data');
        save(['feature_ntu_view_DMM_s001\',pic_name(1:end-4),'_softmax.mat'],'datapre');
        clear data datapre
    end
    
    temp1 = load(['feature_ntu_view_DMM_s001\',pic_name(1:end-4),'.mat']);
    temp1 = temp1.data;
    for view = 1:11
        data_train{nn,view} = [temp1{view,1}(:)];%fc6
    end
    train_label(nn) = label;
    nn = nn+1;
    clear temp1
end

nn=1;
for i = 1:length(index_test)
    i
    index = index_test(i);
    pic_name = imdb.images.name{index};
    label = imdb.images.label(index);
    
    if ~exist(['feature_ntu_view_DMM_s001\',pic_name(1:end-4),'.mat'],'file')  
%         data1 = [];
%         data2 = [];
%         data3 = [];
%         data4 = []; 
        for view = 1:11
            data1 = [];
            data2 = [];
            for j = 1:5
                im_ =  imread(fullfile('F:\chenjun\dynamic\data\ntu_view_DMM_MBB_s001\test\',[pic_name(1:end-4),'_',num2str(view),'_',num2str(j),'.jpg']));
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
                
                data11 = net_temp.vars(net_temp.getVarIndex('x16')).value ;
                data11 = squeeze(gather(data11));
                data22 = net_temp.vars(net_temp.getVarIndex('x19')).value ;
                data22 = squeeze(gather(data22));
                
                data1 = [data1;data11(:)];
                data2 = [data2;data22(:)];
                
                temp = net_temp.vars(net_temp.getVarIndex('prob')).value ;
                temp = squeeze(gather(temp));
                datapre{view,j} = temp;
                
                clear data11 data22 temp
            end
            data{view,1} = data1;%fc6
            data{view,2} = data2;
            clear data1 data2
        end
        save(['feature_ntu_view_DMM_s001\',pic_name(1:end-4),'.mat'],'data');
        save(['feature_ntu_view_DMM_s001\',pic_name(1:end-4),'_softmax.mat'],'datapre');

        clear data datapre
    end
    
    temp1 = load(['feature_ntu_view_DMM_s001\',pic_name(1:end-4),'.mat']);
    temp1 = temp1.data;
    for view = 1:11
        data_test{nn,view} = [temp1{view,1}(:)];%fc6
    end
    test_label(nn) = label;
    nn = nn+1;
    clear temp1
end
%%
Tr_all = [];
Te_all = [];
viewchoose = 1:11;
for view = 1:length(viewchoose)
    view
    Tr_f = double(cell2mat(data_train(:,view)'));
    Te_f = double(cell2mat(data_test(:,view)'));
    
    % //////////////////////// PCA //////////////////////%%%%%
    F_train = Tr_f;
    F_test = Te_f;
    
    Dim = 1000;
    disc_set = Eigenface_f(F_train,Dim);
    F_train = disc_set'*F_train;
    F_test  = disc_set'*F_test;
    F_train = F_train./(repmat(sqrt(sum(F_train.*F_train)), [Dim,1]));
    F_test  = F_test./(repmat(sqrt(sum(F_test.*F_test)), [Dim,1]));
    
    Tr_f = F_train;
    Te_f = F_test;
    
    Tr_all = [Tr_all;Tr_f];
    Te_all = [Te_all;Te_f];
    clear Tr_f Te_f
end

%clear data_train data_test
% % %------------------------------------
% F_train = Tr_f2;
% F_test = Te_f2;
% 
% Dim = 1000; 
% disc_set = Eigenface_f(F_train,Dim);
% F_train = disc_set'*F_train;
% F_test  = disc_set'*F_test;
% F_train = F_train./(repmat(sqrt(sum(F_train.*F_train)), [Dim,1]));
% F_test  = F_test./(repmat(sqrt(sum(F_test.*F_test)), [Dim,1]));
% 
% Tr_f2 = F_train;
% Te_f2 = F_test;
% %%
% Tr_f = [Tr_f1;Tr_f2];
% Te_f = [Te_f1;Te_f2];
% --------------------------------------------
%%% step3: Cross Validation for choosing parameter  
fprintf(1,'step3: Cross Validation for choosing parameter c...\n');  
% the larger c is, more time should be costed  
c = [2^0 2^1 2^2 2^3 2^4];  
max_acc = 0;  
tic;  
for i = 1 : size(c, 2)  
    option = ['-B 1 -c ' num2str(c(i)) ' -v 5 -q'];  
    fprintf(1,'Stage: %d/%d: c = %d, ', i, size(c, 2), c(i));  
    accuracy = train(train_label', sparse(Tr_all'), option);   
    if accuracy > max_acc  
        max_acc = accuracy;  
        best_c = i;  
    end  
end  
fprintf(1,'The best c is c = %d.\n', c(best_c));  
toc; 
%%%%%%
tic;  
fprintf(1,'step4: training...\n');  
option = ['-c ' num2str(c(best_c)) ' -B 1 -e 0.001'];  
model = train(train_label', sparse(Tr_all'), option);  
toc;  
  
%%% step5: test the model  
fprintf(1,'step5: Testing...\n');  
tic;  
[predict_label, accuracy, dec_values] = predict(test_label', sparse(Te_all'), model);  
toc;  

