 
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
    label = imdb.images.label(index);
    im_ =  imread(fullfile(imdb.imageDir.train,imdb.images.name{index}));
    im_ = single(im_);
    im_ = imresize(im_, net1.meta.normalization.imageSize(1:2)) ;
    im_ = bsxfun(@minus, im_, net1.meta.normalization.averageImage) ;
    
    net1.eval({'input',im_}) ;
    
    data1 = net1.vars(net1.getVarIndex('x19')).value ;
    data1 = squeeze(gather(data1));
%     data2 = net1.vars(net1.getVarIndex('x17')).value ;
%     data2 = squeeze(gather(data2));
%     data3 = net1.vars(net1.getVarIndex('x19')).value ;
%     data3 = squeeze(gather(data3));
%     data4 = net1.vars(net1.getVarIndex('x20')).value ;
%     data4 = squeeze(gather(data4));
    
    data_train{i} = [data1];
    train_label(i) = label;
    
    clear data1
end

for i = 1:length(index_test)
    i
    index = index_test(i);
    label = imdb.images.label(index);
    im_ =  imread(fullfile(imdb.imageDir.test,imdb.images.name{index}));
    im_ = single(im_);
    im_ = imresize(im_, net1.meta.normalization.imageSize(1:2)) ;
    im_ = bsxfun(@minus, im_, net1.meta.normalization.averageImage) ;
    
    net1.eval({'input',im_}) ;
    
    data1 = net1.vars(net1.getVarIndex('x19')).value ;
    data1 = squeeze(gather(data1));
%     data2 = net1.vars(net1.getVarIndex('x17')).value ;
%     data2 = squeeze(gather(data2));
%     data3 = net1.vars(net1.getVarIndex('x19')).value ;
%     data3 = squeeze(gather(data3));
%     data4 = net1.vars(net1.getVarIndex('x20')).value ;
%     data4 = squeeze(gather(data4));
    
    data_test{i} = [data1];
    test_label(i) = label;
    
    clear data1
end

% %GMM
% run D:\myself\matlab\matlab_documents\dynamic_image\vlfeat-0.9.20\toolbox\vl_setup.m 
% %% fv
% Mat_train = cell2mat(data_train);
% 
% numClusters = 50; 
% 
% disp('gmm---------');
% [means, covariances, priors] = vl_gmm(Mat_train, numClusters);  
% %%
% fea_dim = size(Mat_train,1);
% Tr_f = zeros(fea_dim*2*numClusters, length(data_train));
% 
% for i = 1:length(data_train)
%     i
%     %%%% get Fisher vectors for the training data using the GMM parameters
%     FV = vl_fisher(data_train{i}, means, covariances, priors, 'Improved'); 
%     Tr_f(:,i) = FV;
% end
% 
% %%
% Te_f = zeros(fea_dim*2*numClusters, length(data_test));
% 
% for i = 1:length(data_test)
%     i
%     %%%% get Fisher vectors for the training data using the GMM parameters
%     FV = vl_fisher(data_test{i}, means, covariances, priors, 'Improved'); 
%     Te_f(:,i) = FV;
% end

Tr_f = double(cell2mat(data_train));
Te_f = double(cell2mat(data_test));

%% //////////////////////// PCA //////////////////////%%%%%
F_train = Tr_f;
F_test = Te_f;

Dim = size(F_train,2) - 10; 
disc_set = Eigenface_f(F_train,Dim);
F_train = disc_set'*F_train;
F_test  = disc_set'*F_test;
F_train = F_train./(repmat(sqrt(sum(F_train.*F_train)), [Dim,1]));
F_test  = F_test./(repmat(sqrt(sum(F_test.*F_test)), [Dim,1]));

Tr_f = F_train;
Te_f = F_test;

%% --------------------------------------------
%%% step3: Cross Validation for choosing parameter  
fprintf(1,'step3: Cross Validation for choosing parameter c...\n');  
% the larger c is, more time should be costed  
c = [2^-6 2^-5 2^-4 2^-3 2^-2 2^-1 2^0 2^1 2^2 2^3];  
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

