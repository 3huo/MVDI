
clc
clear
% 读取第一个大序列 S001…………

% %---服务器路径
path_sk = 'E:\NTU RGBD\nturgb+d_skeletons';
path_sk_out = 'E:\NTU RGBD\faster_rcnn\all_bounding_box';

threh = 0.7;
nn = 0;
all = 0;
S = dir('E:\NTU RGBD\data_depth_w\n*');
for i = 12:17
    temp = ['E:\NTU RGBD\data_depth_w\',S(i).name,'\nturgb+d_depth'];
    path_s = [temp,'\S*'];
    video_n = dir(path_s);
    for j = 1:length(video_n) %%可改
        disp(['i=',num2str(i),';j=',num2str(j)]);
        depth_name = [temp,'\',video_n(j).name];
        sk_name = [path_sk,'\',video_n(j).name,'.skeleton'];
        if exist([path_sk_out,'\',video_n(j).name,'.mat'],'file')
            box_find = load([path_sk_out,'\',video_n(j).name,'.mat']);
            box_find = box_find.boxes_cell{1};
            % 读取该视频序列对应的骨架
            box = get_bounding_box_skelen(sk_name);
            if isempty(box)
                continue;
            else
                [box_all,mask] = box_together(box); 
                save(['all_bounding_box_ntu_1\',video_n(j).name,'.mat'],'box_all')  % function form
%                 o = boxoverlap(box_find, box_all);
%                 if o>threh
%                     nn = nn+1;
%                 end
%                 all = all + 1;
%                 nn/all
            end
        end
    end
end