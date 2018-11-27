 
% 保存特征到文件夹的版本
% 选择不同层的特征+PCA+进行liblinear实验
%
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

net1.conserveMemory = 0;   %将特征都显示出来

length(opts.train.train)

n_t = 1;
for i = 1:length(opts.train.train)
    i
    name = imdb.images.name{opts.train.train(i)};
    if strcmp(name(1:4),'S001')
        index_train(n_t) = opts.train.train(i);
        n_t = n_t + 1;
    end
end

n_t = 1;
for i = 1:length(opts.train.val)
    i
    name = imdb.images.name{opts.train.val(i)};
    if strcmp(name(1:4),'S001')
        index_test(n_t) = opts.train.val(i);
        n_t = n_t + 1;
    end
end

for i = 1:length(index_train)
    i
    index = index_train(i);
    pic_name = imdb.images.name{index};
    label = imdb.images.label(index);
    
    if ~exist(['feature_5_s001\',pic_name(1:end-4),'.mat'],'file')  
        data3 = [];
        data4 = [];
        for j = 1:5
            im_ =  imread(fullfile('dynamic_5',[imdb.images.name{index}(1:end-4),'_',num2str(j),'.jpg']));
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
        data = {data3,data4};
        save(['feature_5_s001\',pic_name(1:end-4),'.mat'],'data');
        clear data3 data4 data
    end
    
    temp = load(['feature_5_s001\',pic_name(1:end-4),'.mat']);
    temp = temp.data;
%     data_train1{i} = [temp{1}(:)];%fc7
    data_train2{i} = [temp{1}(:);temp{2}(:)];%fc6
    train_label(i) = label;
    clear temp
end

for i = 1:length(index_test)
    i
    index = index_test(i);
    pic_name = imdb.images.name{index};
    label = imdb.images.label(index);
    if ~exist(['feature_5_s001\',pic_name(1:end-4),'.mat'],'file')  
        data3 = [];
        data4 = [];
        for j = 1:5
            im_ =  imread(fullfile('dynamic_5',[imdb.images.name{index}(1:end-4),'_',num2str(j),'.jpg']));
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
        data = {data3,data4};
        save(['feature_5_s001\',pic_name(1:end-4),'.mat'],'data');
        clear data3 data4 data
     end
        
    temp = load(['feature_5_s001\',pic_name(1:end-4),'.mat']);
    temp = temp.data;
%     data_test1{i} = [temp{1}(:)];%fc7
    data_test2{i} = [temp{1}(:);temp{2}(:)];%fc6
    test_label(i) = label;
    clear temp    
        
end

% Tr_f1 = double(cell2mat(data_train1));
Tr_f2 = double(cell2mat(data_train2));
% Te_f1 = double(cell2mat(data_test1));
Te_f2 = double(cell2mat(data_test2));
% %% //////////////////////// PCA //////////////////////%%%%%
% F_train = Tr_f1;
% F_test = Te_f1;
% 
% Dim = size(F_train,2); 
% disc_set = Eigenface_f(F_train,Dim);
% F_train = disc_set'*F_train;
% F_test  = disc_set'*F_test;
% F_train = F_train./(repmat(sqrt(sum(F_train.*F_train)), [Dim,1]));
% F_test  = F_test./(repmat(sqrt(sum(F_test.*F_test)), [Dim,1]));
% 
% Tr_f1 = F_train;
% Te_f1 = F_test;
%------------------------------------
F_train = Tr_f2;
F_test = Te_f2;

Dim = size(F_train,2); 
disc_set = Eigenface_f(F_train,Dim);
F_train = disc_set'*F_train;
F_test  = disc_set'*F_test;
F_train = F_train./(repmat(sqrt(sum(F_train.*F_train)), [Dim,1]));
F_test  = F_test./(repmat(sqrt(sum(F_test.*F_test)), [Dim,1]));

Tr_f2 = F_train;
Te_f2 = F_test;
%%
Tr_f = [Tr_f2];
Te_f = [Te_f2];
%% --------------------------------------------
%%% step3: Cross Validation for choosing parameter  
fprintf(1,'step3: Cross Validation for choosing parameter c...\n');  
% the larger c is, more time should be costed  
c = [0.1 2^-1 2^0 2^1 2^2 2^3 10 13 2^4];  
max_acc = 0;  
tic;  
for i = 1 : size(c, 2)  
    option = ['-B 1 -c ' num2str(c(i)) ' -v 5 -q'];  
    fprintf(1,'Stage: %d/%d: c = %d, ', i, size(c, 2), c(i));  
    accuracy = train(train_label', sparse(Tr_f'), option);   
    if accuracy > max_acc  
        max_acc = accuracy;  
        best_c = i;  
    end  
end  
fprintf(1,'The best c is c = %d.\n', c(best_c));  
toc; 
%%%%%%
tic;  
option = ['-c ' num2str(c(best_c)) ' -B 1 -e 0.001'];  
model = train(train_label', sparse(Tr_f'), option);  
toc;  
  
%%% step5: test the model  
fprintf(1,'step5: Testing...\n');  
tic;  
[predict_label, accuracy, dec_values] = predict(test_label', sparse(Te_f'), model);  
toc;  

