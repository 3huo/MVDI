
clc
clear
% 读取第一个大序列 S001…………

% %---服务器路径
% S_name = dir('E:\NTU RGBD\data_depth_w\nturgbd_depth_s001\nturgb+d_depth\S*');
% path_depth = 'E:\NTU RGBD\data_depth_w\nturgbd_depth_s001\nturgb+d_depth';
path_sk = 'E:\NTU RGBD\nturgb+d_skeletons';

S = dir('E:\NTU RGBD\data_depth_w\n*');
for i = 1:17
    %1，2，9，10，11
    temp = ['E:\NTU RGBD\data_depth_w\',S(i).name,'\nturgb+d_depth'];
    path_s = [temp,'\S*'];
    video_n = dir(path_s);
    for j = 1:length(video_n) %%可改
        disp(['i=',num2str(i),';j=',num2str(j)]);
        if exist(['dynamic_multi_DMM\', video_n(j).name,'_',num2str(11),'_',num2str(5),'.jpg'],'file')
            continue;
        end
        depth_name = [temp,'\',video_n(j).name];
        sk_name = [path_sk,'\',video_n(j).name,'.skeleton'];
        % 读取一个视频序列深度图像
        depth_pic= dir([depth_name,'\*.png']);
        % 读取该视频序列对应的骨架
        box = get_bounding_box_skelen(sk_name);
        if isempty(box)
            continue;
        else
            % 根据骨架生成最终的bounding box与掩膜图
            [box_all,mask] = box_together(box);      
            tic
            disp('生成掩膜动态图：')
            % 生成掩膜动态图
            depth = [];
            for k = 1:length(depth_pic)
                img_path = [temp,'\',video_n(j).name,'\',depth_pic(k).name];
                img = double(imread(img_path));     
                depth(:,:,k) = img(box_all(2):box_all(4),box_all(1):box_all(3)); 
%                 depth(:,:,k) = img;
            end
            toc
            % -----------------------------
            view_angle = [-90 -40 -20 -10 -5 0 5 10 20 40 90];
            img = [];
            tic
            disp('生成多view：')
            img = change_view_all(view_angle,depth);
            toc
            %-------------------------------
            tic
            disp('DMM image：')
            for n_d = 1:size(img,4)
                dynamic_depth = [];
                dynamic_depth = depth_projection_DMM(img(:,:,:,n_d));
                for jj = 1:length(dynamic_depth)
                    temp_d = dynamic_depth{jj};
                    temp_d = uint8(temp_d./max(max(temp_d))*255);
                    imwrite(temp_d,['dynamic_multi_DMM\', video_n(j).name,'_',...
                        num2str(n_d),'_',num2str(jj),'.jpg']);
                end
            end
            
%             dynamic_depth = GetDynamicImages_mutil_5_view_2(depth,box_all)  ;
%             for jj = 1:length(dynamic_depth)
%                 temp_d = dynamic_depth{jj};
%                 imwrite(temp_d,['dynamic_s001_general\', video_n(j).name,'_',num2str(jj),'.jpg']);
%             end
            toc
        
            clear depth 
        end
    end
end