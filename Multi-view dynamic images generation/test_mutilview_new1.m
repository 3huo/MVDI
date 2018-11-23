clc
clear
% 读取第一个大序列 S001…………

%---服务器路径
S_name = dir('E:\chenjun\data_depth_w\nturgbd_depth_s001\nturgb+d_depth\S*');
path_depth = 'E:\chenjun\data_depth_w\nturgbd_depth_s001\nturgb+d_depth';
path_sk = 'E:\chenjun\nturgbd_skeletons';

% % %---电脑路径
% S_name = dir('D:\myself\dataset\nturgb+d_depth\S*');
% path_depth = 'D:\myself\dataset\nturgb+d_depth';
% path_sk = 'D:\myself\dataset\nturgbd_skeletons\nturgb+d_skeletons';

i=50;

depth_name = [path_depth,'\',S_name(i).name];
sk_name = [path_sk,'\',S_name(i).name,'.skeleton'];
% 读取一个视频序列深度图像
depth_pic= dir([depth_name,'\*.png']);
% 读取该视频序列对应的骨架
box = get_bounding_box_skelen(sk_name);
% 根据骨架生成最终的bounding box与掩膜图
[box_all,mask] = box_together(box);

tic
% 生成掩膜动态图
tic
depth = [];

for j = 1:length(depth_pic)
    j
    img_path = [path_depth,'\',S_name(i).name,'\',depth_pic(j).name];
    img = double(imread(img_path));
    depth(:,:,j) = img(box_all(2):box_all(4),box_all(1):box_all(3)); 
end
toc
% -----------------------------
tic
% view_angle = [-90 -40 -20 -10 -5 0 5 10 20 40 90];
view_angle = [-10:0.1:10];
img = [];
% for i = 1:length(depth_pic)
%     i
%     im = depth(:,:,i);
%     img(:,:,:,i) = change_view(view_angle,im);
% end
% img = permute(img,[1 2 4 3]);
% 
% for i = 1:size(depth,3)
%      img1 =  depth(:,:,i);
%      img1 = (img1-min(min(img1)))/(max(max(img1))-min(min(img1)));
%      imwrite(double(img1),['C:\Users\chenjun\Desktop\image\depth\pro',num2str(i),'.png'])
% end
% %%
depth_t = depth(:,:,1);

img_rgb = imread('E:\chenjun\data_rgb_new\nturgbd_rgb_s001\nturgb+d_rgb\S001C001P001R001A050\0001.jpg');
img_rgb = imresize(img_rgb,[424,512]);
img_rgb = img_rgb(box_all(2):box_all(4),box_all(1):box_all(3),:);

img = change_view_all_1(view_angle,depth_t,img_rgb);
img = permute(img,[1 2 3 5 4]);

for i = 1:size(img,4)
    figure(1);
    imshow(uint8(imresize(img(:,:,:,i),3)),[]);
    pause(0.1);
end


% toc
% for i = 1:11
% %     figure;
%     imshow(img(:,:,i),[]);
%      img1 =  img(:,:,i);
%       img1 = (img1-min(min(img1)))/(max(max(img1))-min(min(img1)));
%       imwrite(double(img1),['C:\Users\chenjun\Desktop\image\1\',num2str(i),'.png'])
% end
% %% 
% depth_t = depth(:,:,15);
% img = change_view_all(view_angle,depth_t);
% toc
% for i = 1:11
% %     figure;
%     imshow(img(:,:,i),[]);
%      img1 =  img(:,:,i);
%       img1 = (img1-min(min(img1)))/(max(max(img1))-min(min(img1)));
%       imwrite(double(img1),['C:\Users\chenjun\Desktop\image\2\',num2str(i),'.png'])
% end
% %%
% depth_t = depth(:,:,30);
% img = change_view_all(view_angle,depth_t);
% toc
% for i = 1:11
% %     figure;
%     imshow(img(:,:,i),[]);
%      img1 =  img(:,:,i);
%       img1 = (img1-min(min(img1)))/(max(max(img1))-min(min(img1)));
%       imwrite(double(img1),['C:\Users\chenjun\Desktop\image\3\',num2str(i),'.png'])
% end
% %%
% depth_t = depth(:,:,45);
% img = change_view_all(view_angle,depth_t);
% toc
% for i = 1:11
% %     figure;
%     imshow(img(:,:,i),[]);
%      img1 =  img(:,:,i);
%       img1 = (img1-min(min(img1)))/(max(max(img1))-min(min(img1)));
%       imwrite(double(img1),['C:\Users\chenjun\Desktop\image\4\',num2str(i),'.png'])
% end
% %%
% depth_t = depth(:,:,60);
% img = change_view_all(view_angle,depth_t);
% toc
% for i = 1:11
% %     figure;
%     imshow(img(:,:,i),[]);
%      img1 =  img(:,:,i);
%       img1 = (img1-min(min(img1)))/(max(max(img1))-min(min(img1)));
%       imwrite(double(img1),['C:\Users\chenjun\Desktop\image\5\',num2str(i),'.png'])
% end
% %%
% depth_t = depth(:,:,75);
% img = change_view_all(view_angle,depth_t);
% toc
% for i = 1:11
% %     figure;
%     imshow(img(:,:,i),[]);
%      img1 =  img(:,:,i);
%       img1 = (img1-min(min(img1)))/(max(max(img1))-min(min(img1)));
%       imwrite(double(img1),['C:\Users\chenjun\Desktop\image\6\',num2str(i),'.png'])
% end
% %%
% depth_t = depth(:,:,90);
% img = change_view_all(view_angle,depth_t);
% toc
% for i = 1:11
% %     figure;
%     imshow(img(:,:,i),[]);
%      img1 =  img(:,:,i);
%       img1 = (img1-min(min(img1)))/(max(max(img1))-min(min(img1)));
%       imwrite(double(img1),['C:\Users\chenjun\Desktop\image\7\',num2str(i),'.png'])
% end
% 
% 
% 
% %%
% % img = permute(img,[1 2 4 3]);
% % 
% % video_path = 'test1.avi';
% % writerObj = VideoWriter(video_path);
% % writerObj.FrameRate = 15;
% % open(writerObj);
% % for i = 1:size(img,3)
% %     im2 = img(:,:,i);
% %     im2 = (im2-min(min(im2)))/((max(max(im2))-min(min(im2))));
% % %     figure(1);
% % %     imshow(im2,[]);
% %     writeVideo(writerObj,im2);
% % %     pause(0.1);
% % end
% % close(writerObj);    
% % 
% % %%
% % tic
% % for i = 1:size(img,4)
% %     zss = GetDynamicImages_test_gray(img(:,:,:,i));
% %     figure;
% %     imshow(zss,[]);
% % end
% % toc
% 
