
clc
clear
% 读取第一个大序列 S001…………

path_box_rgb = 'E:\chenjun\faster_rcnn\all_bounding_box_rgb';
% path_box_depth = 'E:\chenjun\faster_rcnn\all_bounding_box';

%% -------------------- TESTING --------------------
num = 1;
path2 = 'E:\chenjun\data_rgb\nturgbd_rgb_s001\nturgb+d_rgb';
path2_name = dir([path2,'\S*']);
for ss = length(path2_name)/2+1:length(path2_name)
    ss
    num = num + 1;
    if ~exist([path_box_rgb,'\',path2_name(ss).name(1:end-8),'_rgb.mat'],'file')
        continue;
    end
    %------
    box = load([path_box_rgb,'\',path2_name(ss).name(1:end-4),'.mat']);
    box = box.boxes_cell;
    a = box{1}(1);b = box{1}(2);
    c = box{1}(3)-box{1}(1);d=box{1}(4)-box{1}(2);
    a = round(a);b = round(b);c = round(c);d = round(d);
    %--------
    path3 = [path2,'\',path2_name(ss).name];
    obj = VideoReader(path3);
    numFrames = obj.NumberOfFrames;% 帧的总数
    n_ind = floor(numFrames/3);
    %---------------------------------
    n = 1;
    ind = 1:numFrames;
    rgb = zeros(d+1,c+1,3,length(ind));
    for k = 1:length(ind)
        img = read(obj,ind(k));
        rgb(:,:,:,n) = img(b:b+d,a:a+c,:);
        n = n+1;
    end
    %----------------------------------------
    rgb1 = cat(4,rgb(:,:,:,1:n_ind),rgb(:,:,:,2*n_ind+1:end));
    zWF = GetDynamicImages_test_single(rgb1);
    zWF = imresize(zWF,[256,256]);
    imwrite(zWF,['E:\chenjun\test_ntu\dynamic_image_rgb\dynamic_fb\', path2_name(ss).name(1:end-4),'fb.jpg']);
    %----------------------------------------
    rgb2 = rgb(:,:,:,n_ind+1:2*n_ind);
    zWF = GetDynamicImages_test_single(rgb2);
    zWF = imresize(zWF,[256,256]);
    imwrite(zWF,['E:\chenjun\test_ntu\dynamic_image_rgb\dynamic_m\', path2_name(ss).name(1:end-4),'m.jpg']);
    
end

%         
% for i = 1
%     
%     temp = ['E:\chenjun\data_depth_w\',S(i).name,'\nturgb+d_depth'];
%     path_s = [temp,'\S*'];
%     video_n = dir(path_s);
%     for j = 1:length(video_n) %%可改
%         disp(['i=',num2str(i),';j=',num2str(j)]);
%         if exist(['dynamic_s003_1\', video_n(j).name,'_',num2str(11),'_',num2str(5),'.jpg'],'file')
%             continue;
%         end
%         depth_name = [temp,'\',video_n(j).name];
%         sk_name = [path_sk,'\',video_n(j).name,'.skeleton'];
%         % 读取一个视频序列深度图像
%         depth_pic= dir([depth_name,'\*.png']);
%         % 读取该视频序列对应的骨架
%         box = get_bounding_box_skelen(sk_name);
%         if isempty(box)
%             continue;
%         else
%             % 根据骨架生成最终的bounding box与掩膜图
% %             [box_all,mask] = box_together(box);      
%             tic
%             disp('生成掩膜动态图：')
%             % 生成掩膜动态图
%             depth = [];
%             for k = 1:length(depth_pic)
%                 img_path = [temp,'\',video_n(j).name,'\',depth_pic(k).name];
%                 img = double(imread(img_path));     
% %                 depth(:,:,k) = img(box_all(2):box_all(4),box_all(1):box_all(3)); 
%                 depth(:,:,k) = img;
%             end
%             toc
%             % -----------------------------
%             view_angle = [-90 -40 -20 -10 -5 0 5 10 20 40 90];
%             img = [];
%             tic
%             disp('生成多view：')
%             img = change_view_all(view_angle,depth);
%             toc
%             %-------------------------------
%             tic
%             disp('dynamic image：')
%             for n_d = 1:size(img,4)
%                 dynamic_depth = [];
%                 dynamic_depth = GetDynamicImages_mutil_view_mutil_t(img(:,:,:,n_d));
%                 for jj = 1:length(dynamic_depth)
%                     temp_d = dynamic_depth{jj};
%                     imwrite(temp_d,['dynamic_s003_1\', video_n(j).name,'_',...
%                         num2str(n_d),'_',num2str(jj),'.jpg']);
%                 end
%             end
%             
% %             dynamic_depth = GetDynamicImages_mutil_5_view_2(depth,box_all)  ;
% %             for jj = 1:length(dynamic_depth)
% %                 temp_d = dynamic_depth{jj};
% %                 imwrite(temp_d,['dynamic_s001_general\', video_n(j).name,'_',num2str(jj),'.jpg']);
% %             end
%             toc
%         
%             clear depth 
%         end
%     end
% end