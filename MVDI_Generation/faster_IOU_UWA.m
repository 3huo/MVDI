
clc
clear
% 读取第一个大序列 S001…………

% %---服务器路径
% ground_truth
path_sk = 'all_bounding_box_UWA_groundtruth';
% test
path_sk_out = 'all_bounding_box_UWA';

threh = 0.8;
nn = 0;
all = 0;
S = dir([path_sk_out,'\*.mat']);
for i = 1:length(S)
    
    if (exist([path_sk_out,'\',S(i).name],'file') && ...
            exist([path_sk,'\',S(i).name],'file'))
        box_find = load([path_sk_out,'\',S(i).name]);
        box_find = box_find.boxes_cell{1};
        % 读取该视频序列对应的骨架
        
        box_all = load([path_sk,'\',S(i).name]);
        box_all = box_all.box;
        
        o = boxoverlap(box_find, box_all);
        if o>threh
            nn = nn+1;
        end
        all = all + 1;
        nn/all
        
    end
end