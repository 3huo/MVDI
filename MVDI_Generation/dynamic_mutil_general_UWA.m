
%% test in dataset UWA3D
clc
clear
% 添加liblinear路径用于动态图生成
addpath(genpath('E:\NTU RGBD\test_ntu\liblinear-1.96'));

% %---服务器路径
path_depth = 'E:\NTU RGBD\UWA3DII_Depth\ActionsNew';

path_depth_mask = 'E:\NTU RGBD\test_ntu\all_bounding_box_UWA';
S_mask = dir([path_depth_mask,'\a*']);

S = dir([path_depth,'\a*']);
for i = 1:length(S)
    disp(['i=',num2str(i)]);
    if exist(['dynamic_multi_uwa3d_MBB\',S(i).name(1:end-4),'_',num2str(11),'_',num2str(5),'.jpg'],'file')
        continue;
    end
    load([path_depth,'\',S(i).name]);
    depth = double(curFrame);
    %-----------------------------------------------------
    depth_name_mask = [path_depth_mask,'\',S_mask(i).name];
    load(depth_name_mask);
    box = round(boxes_cell{1});
    %--------------------------------------------
    depth = depth(box(2):box(4),box(1):box(3),:);
    % -----------------------------
    view_angle = [-90 -40 -20 -10 -5 0 5 10 20 40 90];
    img = [];
    tic
    disp('生成多view：')
    img = change_view_all(view_angle,depth);
    toc
    %-------------------------------
    tic
    disp('dynamic image：')
    for n_d = 1:size(img,4)
        dynamic_depth = [];
        dynamic_depth = GetDynamicImages_mutil_view_mutil_t(img(:,:,:,n_d));
        for jj = 1:length(dynamic_depth)
            temp_d = dynamic_depth{jj};
            imwrite(temp_d,['dynamic_multi_uwa3d_MBB\', S(i).name(1:end-4),'_',...
                num2str(n_d),'_',num2str(jj),'.jpg']);
        end
    end
    toc
    
    clear depth
end

