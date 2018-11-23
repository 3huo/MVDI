%  
% 模型为60类的训练模型---同一类组合串联起来进行分类
% 
% 保存特征到文件夹的版本
% 选择不同层的特征+PCA+进行liblinear实验
%
clc
clear
net1 = dagnn.DagNN.loadobj(load('E:\chenjun\dynamic\exp\ntu_1+2\net-deployed.mat')) ;
net1.mode = 'test' ;

imdb = load('E:\chenjun\dynamic\exp\ntu_1+2\imdb.mat') ;
% imdb = imdb.imdb;
b = 1;

opts.dataDir = fullfile('data','ntu_1+2') ;
opts.expDir  = fullfile('exp', 'ntu_1+2') ;
opts.train.train = find(imdb.images.sets==1) ;
opts.train.val = find(imdb.images.sets==3) ;

net1.conserveMemory = 0;   %将特征都显示出来

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

index_test = opts.train.val;


nn = 1;
for i = 1:2:length(index_train)
    i
    index = index_train(i);
    pic_name = imdb.images.name{index};
    label = imdb.images.label(index);
    
    if ~exist(['feature_all_5_f\',pic_name(1:end-6),'.mat'],'file')  
        data1 = [];
        data2 = [];
        data3 = [];
        data4 = [];    
        for j = 1:5
            im_ =  imread(fullfile('E:\chenjun\test_ntu\dynamic_mutil',[imdb.images.name{index}(1:end-6),'_',num2str(j),'.jpg']));
            im_ = single(im_);
            im_ = imresize(im_, net1.meta.normalization.imageSize(1:2)) ;
            im_ = bsxfun(@minus, im_, net1.meta.normalization.averageImage) ;
            
            net1.eval({'input',im_}) ;
            
            data11 = net1.vars(net1.getVarIndex('x16')).value ;
            data11 = squeeze(gather(data11));
            data22 = net1.vars(net1.getVarIndex('x19')).value ;
            data22 = squeeze(gather(data22));

            data1 = [data1;data11(:)];
            data2 = [data2;data22(:)];
            clear data11 data22
        end
        
        for j = 1:5
            im_ =  imread(fullfile('E:\chenjun\test_ntu\dynamic_mutil_s',[imdb.images.name{index}(1:end-6),'_',num2str(j),'.jpg']));
            im_ = single(im_);
            im_ = imresize(im_, net1.meta.normalization.imageSize(1:2)) ;
            im_ = bsxfun(@minus, im_, net1.meta.normalization.averageImage) ;
            
            net1.eval({'input',im_}) ;
            
            data33 = net1.vars(net1.getVarIndex('x16')).value ;
            data33 = squeeze(gather(data33));
            data44 = net1.vars(net1.getVarIndex('x19')).value ;
            data44 = squeeze(gather(data44));

            data3 = [data3;data33(:)];
            data4 = [data4;data44(:)];
            clear data33 data44
        end
        
        data = {data1,data2,data3,data4};
        save(['feature_all_5_f\',pic_name(1:end-6),'.mat'],'data');
        clear data1 data2 data3 data4 data
    end
    
    temp1 = load(['feature_all_5_f\',pic_name(1:end-6),'.mat']);
    temp1 = temp1.data;
  
    temp2 = load(['feature_all_5_s\',pic_name(1:end-6),'.mat']);
    temp2 = temp2.data;
    data_train1{nn} = [temp1{1}(:)];%fc7
    data_train2{nn} = [temp2{1}(:)];%fc6
    train_label(nn) = label;
    nn = nn+1;
    clear temp1 temp2
end
nn=1;
for i = 1:2:length(index_test)
    i
    index = index_test(i);
    pic_name = imdb.images.name{index};
    label = imdb.images.label(index);
    if ~exist(['feature_all_5_s\',pic_name(1:end-6),'.mat'],'file')  
        data1 = [];
        data2 = [];
        data3 = [];
        data4 = [];    
        for j = 1:5
            im_ =  imread(fullfile('E:\chenjun\test_ntu\dynamic_mutil',[imdb.images.name{index}(1:end-6),'_',num2str(j),'.jpg']));
            im_ = single(im_);
            im_ = imresize(im_, net1.meta.normalization.imageSize(1:2)) ;
            im_ = bsxfun(@minus, im_, net1.meta.normalization.averageImage) ;
            
            net1.eval({'input',im_}) ;
            
            data11 = net1.vars(net1.getVarIndex('x16')).value ;
            data11 = squeeze(gather(data11));
            data22 = net1.vars(net1.getVarIndex('x19')).value ;
            data22 = squeeze(gather(data22));

            data1 = [data1;data11(:)];
            data2 = [data2;data22(:)];
            clear data11 data22
        end
        
        for j = 1:5
            im_ =  imread(fullfile('E:\chenjun\test_ntu\dynamic_mutil_s',[imdb.images.name{index}(1:end-6),'_',num2str(j),'.jpg']));
            im_ = single(im_);
            im_ = imresize(im_, net1.meta.normalization.imageSize(1:2)) ;
            im_ = bsxfun(@minus, im_, net1.meta.normalization.averageImage) ;
            
            net1.eval({'input',im_}) ;
            
            data33 = net1.vars(net1.getVarIndex('x16')).value ;
            data33 = squeeze(gather(data33));
            data44 = net1.vars(net1.getVarIndex('x19')).value ;
            data44 = squeeze(gather(data44));

            data3 = [data3;data33(:)];
            data4 = [data4;data44(:)];
            clear data33 data44
        end
        
        data = {data1,data2,data3,data4};
        save(['feature_all_5_s\',pic_name(1:end-6),'.mat'],'data');
        clear data1 data2 data3 data4 data
     end
        
    temp1 = load(['feature_all_5_f\',pic_name(1:end-6),'.mat']);
    temp1 = temp1.data;
            
    temp2 = load(['feature_all_5_s\',pic_name(1:end-6),'.mat']);
    temp2 = temp2.data;
    
    data_test1{nn} = [temp1{1}(:)];%fc7
    data_test2{nn} = [temp2{1}(:)];%fc6
    test_label(nn) = label;
    nn = nn+1;
    clear temp1 temp2
        
end

Tr_f1 = double(cell2mat(data_train1));
Tr_f2 = double(cell2mat(data_train2));
Te_f1 = double(cell2mat(data_test1));
Te_f2 = double(cell2mat(data_test2));
%% //////////////////////// PCA //////////////////////%%%%%
F_train = Tr_f1;
F_test = Te_f1;

Dim = 2000; 
disc_set = Eigenface_f(F_train,Dim);
F_train = disc_set'*F_train;
F_test  = disc_set'*F_test;
F_train = F_train./(repmat(sqrt(sum(F_train.*F_train)), [Dim,1]));
F_test  = F_test./(repmat(sqrt(sum(F_test.*F_test)), [Dim,1]));

Tr_f1 = F_train;
Te_f1 = F_test;
% %------------------------------------
F_train = Tr_f2;
F_test = Te_f2;

Dim = 1000; 
disc_set = Eigenface_f(F_train,Dim);
F_train = disc_set'*F_train;
F_test  = disc_set'*F_test;
F_train = F_train./(repmat(sqrt(sum(F_train.*F_train)), [Dim,1]));
F_test  = F_test./(repmat(sqrt(sum(F_test.*F_test)), [Dim,1]));

Tr_f2 = F_train;
Te_f2 = F_test;
%%
Tr_f = [Tr_f1;Tr_f2];
Te_f = [Te_f1;Te_f2];
%% --------------------------------------------
%%% step3: Cross Validation for choosing parameter  
fprintf(1,'step3: Cross Validation for choosing parameter c...\n');  
% % the larger c is, more time should be costed  
% c = [2^-1 2^0 2^1 2^2 2^3 2^4];  
% max_acc = 0;  
% tic;  
% for i = 1 : size(c, 2)  
%     option = ['-B 1 -c ' num2str(c(i)) ' -v 5 -q'];  
%     fprintf(1,'Stage: %d/%d: c = %d, ', i, size(c, 2), c(i));  
%     accuracy = train(train_label', sparse(Tr_f'), option);   
%     if accuracy > max_acc  
%         max_acc = accuracy;  
%         best_c = i;  
%     end  
% end  
% fprintf(1,'The best c is c = %d.\n', c(best_c));  
% toc; 
%%%%%%
tic;  
fprintf(1,'step4: training...\n');  
option = ['-c ' num2str(8) ' -B 1 -e 0.001'];  
model = train(train_label', sparse(Tr_f'), option);  
toc;  
  
%%% step5: test the model  
fprintf(1,'step5: Testing...\n');  
tic;  
[predict_label, accuracy, dec_values] = predict(test_label', sparse(Te_f'), model);  
toc;  

