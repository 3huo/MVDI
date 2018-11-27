
% 对 UCF101的数据库进行前中后的dynamic image
clc
clear
% 读取第一个大序列 S001…………
%%
% 如果没有，需要添加Libsvm工具箱
addpath('E:\chenjun\dynamic\liblinear-1.96\matlab');

path_all = 'E:\chenjun\ucf101\UCF_101';
file_all = dir(path_all);
for i = 81:length(file_all)
    num = 1;
    path_file_i = [path_all,'\',file_all(i).name];
    file_all_i = dir([path_file_i,'\*.avi']);
    label = i-2;
    for n = 1:length(file_all_i)
        disp(['now == action:',num2str(label),' ;num = ',num2str(n)]);
        
        a1 = ['E:\chenjun\test_ntu\dynamic_ucf101_fmb\dynamic_f\', file_all_i(n).name(1:end-4),'_',num2str(label),'_f.jpg'];
        a2 = ['E:\chenjun\test_ntu\dynamic_ucf101_fmb\dynamic_m\', file_all_i(n).name(1:end-4),'_',num2str(label),'_m.jpg'];
        a3 = ['E:\chenjun\test_ntu\dynamic_ucf101_fmb\dynamic_b\', file_all_i(n).name(1:end-4),'_',num2str(label),'_b.jpg'];
        if exist(a1,'file') && exist(a2,'file') && exist(a3,'file') 
            continue
        end
        path_temp = [path_file_i,'\',file_all_i(n).name];
        obj = VideoReader(path_temp);
        numFrames = obj.NumberOfFrames;% 帧的总数
        n_ind = floor(numFrames/3);
        %---------------------------------
        ind = 1:numFrames;
        rgb = [];
        for k = 1:length(ind)
            img = read(obj,ind(k));
            rgb(:,:,:,k) = img;
        end
        %----------------------------------------
        %     rgb1 = cat(4,rgb(:,:,:,1:n_ind),rgb(:,:,:,2*n_ind+1:end));
        rgb1 = rgb(:,:,:,1:n_ind);
        zWF = GetDynamicImages_test_single(rgb1);
        zWF = imresize(zWF,[256,256]);
        imwrite(zWF,['E:\chenjun\test_ntu\dynamic_ucf101_fmb\dynamic_f\', file_all_i(n).name(1:end-4),'_',num2str(label),'_f.jpg']);
        %----------------------------------------
        rgb2 = rgb(:,:,:,n_ind+1:2*n_ind);
        zWF = GetDynamicImages_test_single(rgb2);
        zWF = imresize(zWF,[256,256]);
        imwrite(zWF,['E:\chenjun\test_ntu\dynamic_ucf101_fmb\dynamic_m\', file_all_i(n).name(1:end-4),'_',num2str(label),'_m.jpg']);
        %----------------------------------------
        rgb3 = rgb(:,:,:,2*n_ind+1:end);
        zWF = GetDynamicImages_test_single(rgb3);
        zWF = imresize(zWF,[256,256]);
        imwrite(zWF,['E:\chenjun\test_ntu\dynamic_ucf101_fmb\dynamic_b\', file_all_i(n).name(1:end-4),'_',num2str(label),'_b.jpg']);
    end
end

