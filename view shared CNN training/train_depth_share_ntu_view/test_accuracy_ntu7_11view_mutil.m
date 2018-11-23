% 模型为60类的训练模型--11个view-多multi
% 保存特征到文件夹的版本
% 选择不同层的特征+PCA+进行liblinear实验
clc,clear
% 加载工具箱
run(fullfile('F:\chenjun\dynamic','matconvnet-1.0-beta23\matconvnet-1.0-beta23', 'matlab', 'vl_setupnn.m')) ;
% 加载训练好的网络模型
net_view1 = dagnn.DagNN.loadobj(load('F:\chenjun\dynamic\exp\ntu_sharesubject_view1\net-deployed-view1.mat')) ;
net_view1.mode = 'test' ;  % 注意修改为测试模型
net_view2 = dagnn.DagNN.loadobj(load('F:\chenjun\dynamic\exp\ntu_sharesubject_view2\net-deployed-view2.mat')) ;
net_view2.mode = 'test' ;
net_view3 = dagnn.DagNN.loadobj(load('F:\chenjun\dynamic\exp\ntu_sharesubject_view3\net-deployed-view3.mat')) ;
net_view3.mode = 'test' ;
net_view4 = dagnn.DagNN.loadobj(load('F:\chenjun\dynamic\exp\ntu_sharesubject_view4\net-deployed-view4.mat')) ;
net_view4.mode = 'test' ;
net_view5 = dagnn.DagNN.loadobj(load('F:\chenjun\dynamic\exp\ntu_sharesubject_view5\net-deployed-view5.mat')) ;
net_view5.mode = 'test' ; 
% all view is the same imdb
imdb = load('F:\chenjun\dynamic\exp\ntu_all_view1\imdb.mat') ;

opts.train.train = find(imdb.images.sets==1) ;
opts.train.val = find(imdb.images.sets==3) ;
net_view1.conserveMemory = 0;   %将全连接层特征都显示出来 
net_view2.conserveMemory = 0;   %将特征都显示出来
net_view3.conserveMemory = 0;   %将特征都显示出来
net_view4.conserveMemory = 0;   %将特征都显示出来
net_view5.conserveMemory = 0;   %将特征都显示出来
% 找到对应的训练样本与测试样本
index_train = opts.train.train;
index_test = opts.train.val;
%---------------------train的所有样本---------------------------
for i = 1:length(index_train)
    i
    index = index_train(i);
    pic_name = imdb.images.name{index};
    label = imdb.images.label(index);
    % 设置保存特征的文件夹
    if ~exist(['feature_sharesubject\',pic_name(1:end-4),'.mat'],'file') 
        datapre = [];
        for view = 1:11
            data1 = [];data2 = [];
            for j = 1:5
            % 读取某个视角下的图像：设置好对应的数据库路径
                im_ =  imread(fullfile('F:\chenjun\dynamic\data\dynamic_mutil_ntu_MBB\',[pic_name(1:end-4),'_',num2str(view),'_',num2str(j),'.jpg']));
                im_ = single(im_);
                % 选择对应的模型
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
                % 图像调整大小并减去均值 
                im_ = imresize(im_, net_temp.meta.normalization.imageSize(1:2)) ;
                im_ = bsxfun(@minus, im_, net_temp.meta.normalization.averageImage) ;
                net_temp.eval({'input',im_}) ; % 送入网络中计算
                % 只提取全连接层特征
                data11 = net_temp.vars(net_temp.getVarIndex('x16')).value ; % fc6
                data11 = squeeze(gather(data11));
                data22 = net_temp.vars(net_temp.getVarIndex('x19')).value ; % fc7
                data22 = squeeze(gather(data22));
                % 直接softmax的特征
                temp = net_temp.vars(net_temp.getVarIndex('prob')).value ;
                temp = squeeze(gather(temp));
                datapre{view,j} = temp; 
                % 保存得到的结果
                data1 = [data1;data11(:)];
                data2 = [data2;data22(:)];
                clear data11 data22 temp
            end
            data{view,1} = data1;%fc6
            data{view,2} = data2;
            clear data1 data2
        end
        save(['feature_sharesubject\',pic_name(1:end-4),'.mat'],'data');
        save(['feature_sharesubject_softmax\',pic_name(1:end-4),'_softmax.mat'],'datapre');
        clear data datapre
    end
    % 加载特征
    temp1 = load(['feature_sharesubject\',pic_name(1:end-4),'.mat']);
    temp1 = temp1.data;
    for view = 1:11
        data_train{nn,view} = [temp1{view,1}(:)]; %  选择1：fc6  2：fc7
    end
    train_label(nn) = label;
    nn = nn+1;
    clear temp1
end
%---------------------test的所有样本---------------------------
for i = 1:length(index_test)
    i
    index = index_test(i);
    pic_name = imdb.images.name{index};
    label = imdb.images.label(index);
    if ~exist(['feature_sharesubject\',pic_name(1:end-4),'.mat'],'file')  
        for view = 1:11
            data1 = [];data2 = [];
            for j = 1:5
             % 读取某个视角下的图像：设置好对应的数据库路径
                im_ =  imread(fullfile('F:\chenjun\dynamic\data\dynamic_mutil_ntu_MBB\',[pic_name(1:end-4),'_',num2str(view),'_',num2str(j),'.jpg']));
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
                % 读取softmax概率输出
                temp = net_temp.vars(net_temp.getVarIndex('prob')).value ;
                temp = squeeze(gather(temp));
                datapre{view,j} = temp;
                clear data11 data22 temp
            end
            data{view,1} = data1;%fc6
            data{view,2} = data2;
            clear data1 data2
        end
        save(['feature_sharesubject\',pic_name(1:end-4),'.mat'],'data');
        save(['feature_sharesubject\',pic_name(1:end-4),'_softmax.mat'],'datapre');
        clear data datapre
    end
    % 加载特征
    temp1 = load(['feature_multi_ntu_shared300_viewtest\',pic_name(1:end-4),'.mat']);
    temp1 = temp1.data;
    for view = 1:11
        data_test{nn,view} = [temp1{view,1}(:)];%fc6
    end
    test_label(nn) = label;
    nn = nn+1;
    clear temp1
end

Tr_all = [];
Te_all = [];
viewchoose = 1:11;
% 每个视角单独降维
for view = 1:length(viewchoose)
    view
    Tr_f = double(cell2mat(data_train(:,view)'));
    Te_f = double(cell2mat(data_test(:,view)'));
    % //////////////////////// PCA //////////////////////%%%%%
    F_train = Tr_f;
    F_test = Te_f;
    Dim = 1000;   % 降维维度选择
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
%  交叉验证来选择svm训练的参数
c = [2^-2 2^-1 2^0 2^1 2^2 2^3 2^4];  
max_acc = 0;  
tic;  
for i = 1 : size(c, 2)  
    option = ['-B 1 -c ' num2str(c(i)) ' -v 5 -q'];  
    fprintf(1,'Stage: %d/%d: c = %d, ', i, size(c, 2), c(i));  
    accuracy = train(train_label', sparse(Tr_all'), option);   %liblinear工具箱函数，注意添加工具箱路径
    if accuracy > max_acc  
        max_acc = accuracy;  
        best_c = i;  
    end  
end  
fprintf(1,'The best c is c = %d.\n', c(best_c));  
toc; 
% ------模型训练---------
tic;  
fprintf(1,'training...\n');  
option = ['-c ' num2str(c(best_c)) ' -B 1 -e 0.001'];  
model = train(train_label', sparse(Tr_all'), option);  
toc;   
% ------模型测试--------
fprintf(1,'step5: Testing...\n');  
tic;  
[predict_label, accuracy, dec_values] = predict(test_label', sparse(Te_all'), model);  
toc;  