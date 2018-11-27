
% path = 'E:\chenjun\data_depth_w\nturgbd_depth_s001\nturgb+d_depth\S001C001P001R001A001';
% 
% file = dir([path,'\*.png']);
% 
% choose = 1:floor(length(file)/5):length(file);
% 
% pathout = 'E:\chenjun\out1';
% for i = 1:length(choose)
%     img = imread([path,'\',file(choose(i)).name]);
%     out = gray_color(img);
%     imwrite(out,[pathout,'\',file(choose(i)).name]);
% end

path1 = 'E:\chenjun\data_depth_w\nturgbd_depth_s008\nturgb+d_depth\S008C001P008R001A059';
path2 = 'E:\chenjun\data_depth_w\nturgbd_depth_s008\nturgb+d_depth\S008C002P008R001A059';
path3 = 'E:\chenjun\data_depth_w\nturgbd_depth_s008\nturgb+d_depth\S008C003P008R001A059';
pathout = 'E:\chenjun\out1\4';

num = 70;
img = imread([path1,'\Depth-000000',num2str(num),'.png']);
% out = gray_color(img);
img = double(img);
img = img/max(max(img));
imwrite(img,[pathout,'\11.png']);

img = imread([path2,'\Depth-000000',num2str(num),'.png']);
% out = gray_color(img);
img = double(img);
img = img/max(max(img));
imwrite(img,[pathout,'\22.png']);

img = imread([path3,'\Depth-000000',num2str(num),'.png']);
% out = gray_color(img);
img = double(img);
img = img/max(max(img));
imwrite(img,[pathout,'\33.png']);

