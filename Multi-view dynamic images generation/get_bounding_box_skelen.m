% 根据某个骨架文件夹，得到这个序列的所有深度图的boundingbox
% 此过程模拟faster-rcnn得到bounding，假设已经得到

function box = get_bounding_box_skelen(filename)

bodyinfo = read_skeleton_file(filename);
if isempty(bodyinfo)
    disp('没有该序列的骨架没有bounding box');
    box = [];
else
    
    % filename = ['D:\myself\dataset\nturgbd_skeletons\nturgb+d_skeletons\S001C001P001R001A001.skeleton'];
    for num_depth_img = 1:length(bodyinfo)
        img_depth = bodyinfo(num_depth_img).bodies;
        num_box = length(bodyinfo(num_depth_img).bodies);
        for people = 1:num_box
            box_temp = img_depth(people).joints;
            xx_all = [];
            yy_all = [];
            for i = 1:25 %25 个关键点  固定
                xx = box_temp(i).depthX;
                yy = box_temp(i).depthY;
                xx_all = [xx_all,xx];
                yy_all = [yy_all,yy];
            end
            mean_x = mean(xx_all);
            mean_y = mean(yy_all);
            length_x = round((abs(mean_x - min(xx_all)) + abs(mean_x - max(xx_all)))/2 + 15);%15
            length_y = round((abs(mean_y - min(yy_all)) + abs(mean_y - max(yy_all)))/2 + 15);
            box_temp = zeros(1,4);
            box_temp(1) = round(mean_x - length_x);
            box_temp(2) = round(mean_y - length_y);
            box_temp(3) = round(mean_x + length_x);
            box_temp(4) = round(mean_y + length_y);
            if box_temp(1) < 0
                box_temp(1) = 0;
            end
            if box_temp(2) < 0
                box_temp(2) = 0;
            end
            if box_temp(3) > 512
                box_temp(3) = 512;
            end
            if box_temp(4) >424
                box_temp(4) = 424;
            end
            % 存储到最终的box中
            box{num_depth_img,people} = box_temp;
        end
    end
end
