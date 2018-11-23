
clc
clear
% 读取第一个大序列 S001…………

% %---服务器路径
% S_name = dir('E:\chenjun\data_depth_w\nturgbd_depth_s001\nturgb+d_depth\S*');
% path_depth = 'E:\chenjun\data_depth_w\nturgbd_depth_s001\nturgb+d_depth';
path_sk = 'E:\chenjun\nturgb+d_skeletons';

S = dir('E:\chenjun\data_depth_w\n*');

path_box_rgb = 'E:\chenjun\faster_rcnn\all_bounding_box_rgb';
path_box_depth = 'E:\chenjun\faster_rcnn\all_bounding_box';

%% -------------------- TESTING --------------------
num = 1;
path1 = 'E:\chenjun\data_rgb\nturgbd_rgb_s017';
path1_name = dir([path1,'\n*']);
for s = 1:length(path1_name)
    path2 = [path1,'\',path1_name(s).name];
    path2_name = dir([path2,'\S*']);
    for ss = 1:length(path2_name)
        num
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
        frame = 1;
        box_all = [];
        n = 1;
        ind = 1:2:numFrames;
        rgb = zeros(d+1,c+1,3,length(ind));
        for k = 1:length(ind)  
            img = read(obj,ind(k)); 
            rgb(:,:,:,n) = img(b:b+d,a:a+c,:);
            n = n+1;
        end
        zWF = GetDynamicImages_test_single(rgb);
        zWF = imresize(zWF,[256,256]);
        imwrite(zWF,['dynamic_rgb_all\', path2_name(ss).name(1:end-4),'.jpg']);
    end
end
