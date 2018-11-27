
clc
clear
% add liblinear path for dynamic images generation
addpath(genpath('E:\NTU RGBD\test_ntu\liblinear-1.96'));
% the human skeletons data path
path_sk = 'E:\NTU RGBD\nturgb+d_skeletons';

S = dir('E:\NTU RGBD\data_depth_w\n*');
for i = 1:length(S)  % 遍历读取所有场景下的数据集（s001-s017） 
    
    temp = ['E:\NTU RGBD\data_depth_w\',S(i).name,'\nturgb+d_depth'];
    path_s = [temp,'\S*'];
    video_n = dir(path_s);
    for j = 1:length(video_n) % 可改,遍历读取该S* 下的所有视频
        disp(['i=',num2str(i),';j=',num2str(j)]);
        % 判断该视频是否已经被转换
        if exist(['dynamic_mutil_general1\', video_n(j).name,'_',num2str(11),'_',num2str(5),'.jpg'],'file')
            continue;
        end
        %-------模拟生成运动边框/使用模型生成的边框----------
        depth_name = [temp,'\',video_n(j).name];
        sk_name = [path_sk,'\',video_n(j).name,'.skeleton'];
        % 读取一个视频序列深度图像
        depth_pic= dir([depth_name,'\*.png']);
        % 读取该视频序列对应的骨架
        box = get_bounding_box_skelen(sk_name);
        %-------判断模拟生成运动边框的存在性-----
        if isempty(box)
            continue;
        else
            % 根据骨架生成最终的bounding box与掩膜图
            [box_all,mask] = box_together(box);    
            %---------------生成带有运动边框的掩膜动态图------------------
            tic
            disp('生成掩膜动态图：')      
            depth = [];
            for k = 1:length(depth_pic)
                img_path = [temp,'\',video_n(j).name,'\',depth_pic(k).name];
                img = double(imread(img_path));     
                depth(:,:,k) = img(box_all(2):box_all(4),box_all(1):box_all(3)); 
%                 depth(:,:,k) = img;  % 直接使用原视频，不用运动边框
            end
            toc
            % -------------规定的多视角投影视角----------------
            view_angle = [-90 -40 -20 -10 -5 0 5 10 20 40 90];
            img = [];
            tic
            disp('生成多view：')
            % -----对所有帧进行多视角投影变换-------
            img = change_view_all(view_angle,depth);
            toc
             %------对多视角视频进行多时间段的动态图生成---
            tic
            disp('dynamic image：')
            for n_d = 1:size(img,4) % 遍历所有视角
                dynamic_depth = [];
                % 每个视角的视频进行5个时间段的动态图生成
                dynamic_depth = GetDynamicImages_mutil_view_mutil_t(img(:,:,:,n_d));
                % 对多视角多时间段动态图进行存储
                for jj = 1:length(dynamic_depth)
                    temp_d = dynamic_depth{jj};
                    imwrite(temp_d,['dynamic_mutil_general\', video_n(j).name,'_',...
                        num2str(n_d),'_',num2str(jj),'.jpg']);
                end
            end
            toc  
            clear depth 
        end
    end
end