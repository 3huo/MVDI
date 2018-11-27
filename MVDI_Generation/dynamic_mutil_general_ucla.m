
%% test in dataset UCLA
clc
clear
% 添加liblinear路径用于动态图生成
addpath(genpath('E:\NTU RGBD\test_ntu\liblinear-1.96'));

% %---服务器路径--------------
path_depth = 'E:\NUCLA3D\NUCLA3D';
S = dir([path_depth,'\a*']);

path_depth_mask = 'E:\NTU RGBD\test_ntu\all_bounding_box_ucla';
S_mask = dir([path_depth_mask,'\a*']);

for i = 1:length(S)
    disp(['i=',num2str(i)]);
    if exist(['dynamic_multi_ucla_MBB\',S(i).name,'_',num2str(11),'_',num2str(5),'.jpg'],'file')
        continue;
    end
    %--------------------------------------------
    path2 = dir([path_depth,'\',S(i).name,'\*.png']);
    for jj = 1:length(path2)
    	depth(:,:,jj) = imread([path_depth,'\',S(i).name,'\',path2(jj).name]);
    end    
    depth = double(depth);
    
    %-----------------------------------------------------
    depth_name_mask = [path_depth_mask,'\',S_mask(i).name];
    load(depth_name_mask);
    box = round(boxes_cell{1});
    %--------------------------------------------
    depth = depth(box(2):box(4),box(1):box(3),:);
    % %     imshow(depth(:,:,20),[]);pause(0.01);
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
            imwrite(temp_d,['dynamic_multi_ucla_MBB\', S(i).name,'_',...
                num2str(n_d),'_',num2str(jj),'.jpg']);
        end
    end

% --------------使用DMM生成生成动态图-----------------------
%     for n_d = 1:size(img,4)
%         dynamic_depth = [];
%         dynamic_depth = depth_projection_DMM(img(:,:,:,n_d));
%         for jj = 1:length(dynamic_depth)
%             temp_d = dynamic_depth{jj};
%             temp_d = uint8(temp_d./max(max(temp_d))*255);
%             imwrite(temp_d,['dynamic_multi_ucla_DMM\', S(i).name,'_',...
%                 num2str(n_d),'_',num2str(jj),'.jpg']);
%         end
%     end
    
    toc
    clear depth
end

