
clc
clear
% 读取第一个大序列 S001…………

% %---服务器路径
path_sk = 'all_bounding_box_ntu_1';
path_sk_out = 'E:\NTU RGBD\faster_rcnn\all_bounding_box';

threh = 0.3;
nn = 0;
all = 0;
S = dir('E:\NTU RGBD\data_depth_w\n*');
for i = 1:17
    temp = ['E:\NTU RGBD\data_depth_w\',S(i).name,'\nturgb+d_depth'];
    path_s = [temp,'\S*'];
    video_n = dir(path_s);
    for j = 1:length(video_n) %%可改
        disp(['i=',num2str(i),';j=',num2str(j)]);
        depth_name = [temp,'\',video_n(j).name];
        if (exist([path_sk_out,'\',video_n(j).name,'.mat'],'file') && ...
             exist([path_sk,'\',video_n(j).name,'.mat'],'file'))
            box_find = load([path_sk_out,'\',video_n(j).name,'.mat']);
            box_find = box_find.boxes_cell{1};
            % 读取该视频序列对应的骨架
           
            box_all = load([path_sk,'\',video_n(j).name,'.mat']);
            box_all = box_all.box_all;
            
%             img = imread(['all_bounding_box_ntu\',video_n(j).name,'.jpg']);
%              figure(1);
%             imshow(img,[]);
%             hold on;
%             rectangle('Position',[box_find(1),box_find(2),box_find(3)-box_find(1),box_find(4)-box_find(2)],...
%                 'edgecolor','r');
%             hold on;
%             rectangle('Position',[box_all(1),box_all(2),box_all(3)-box_all(1),box_all(4)-box_all(2)],...
%                 'edgecolor','g');
%             pause(0.01);
%             
            o = boxoverlap(box_find, box_all);
            if o>threh
                nn = nn+1;
            end
            all = all + 1;
            nn/all
 
        end
    end
end