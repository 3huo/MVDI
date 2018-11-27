function img_all = change_view_all(view_angle,image_depth)
frame = size(image_depth,3);

ind = 0;
nn = 0;
point3d = [];
for n_f = 1:frame
%     n_f
    temp_image_depth = image_depth(:,:,n_f);
    % 索引3维坐标点
    i = [];j = []; 
    [i ,j] = find(temp_image_depth>0|temp_image_depth<0);
    ind_p = sub2ind(size(temp_image_depth),i,j) ;
    point3d = [point3d;i,j,temp_image_depth(ind_p)];%原始坐标点
    nn = nn + length(i);
    ind = [ind,nn];
end
%
for angle_num = 1:length(view_angle)
    img = [];
    view_angle_temp = view_angle(angle_num);
    % 旋转轴为xy，正视图投影
%     data_3d = view3D_all(view_angle_temp,0,0,point3d,'view_xy');
    [data_3d_r,data_3d_p] = view3D_all_rp(view_angle_temp,0,0,point3d,'view_xy');
    %----将投影后坐标全都变成正数
    min_x = min(data_3d_p(:,1));
    if min_x < 0
        data_3d_p(:,1) = data_3d_p(:,1) + abs(min_x) + 1;
    end
    min_y = min(data_3d_p(:,2));
    if min_y < 0
        data_3d_p(:,2) = data_3d_p(:,2) + abs(min_y) + 1;
    end
    min_z = min(data_3d_p(:,3));
    if min_z < 0
        data_3d_p(:,3) = data_3d_p(:,3) + abs(min_z) + 1;
    end
    %----将旋转后坐标全都变成正数
    min_x = min(data_3d_r(:,1));
    if min_x < 0
        data_3d_r(:,1) = data_3d_r(:,1) + abs(min_x) + 1;
    end
    min_y = min(data_3d_r(:,2));
    if min_y < 0
        data_3d_r(:,2) = data_3d_r(:,2) + abs(min_y) + 1;
    end
    min_z = min(data_3d_r(:,3));
    if min_z < 0
        data_3d_r(:,3) = data_3d_r(:,3) + abs(min_z) + 1;
    end
    %-------------------------------------------------
    data_3d_p = ceil(data_3d_p);
    row1 = (max(data_3d_p(:,1)));
    col1 = (max(data_3d_p(:,2)));
    max_depth1 = (max(data_3d_p(:,3)));
    %-------------------------------------------------     
    img_temp = [];
    for n_f = 1:frame
        data_3d_temp = data_3d_p(ind(n_f)+1:ind(n_f+1),:);%投影后的坐标点
        point3d_temp = data_3d_r(ind(n_f)+1:ind(n_f+1),:); %旋转后的坐标点
        %----------------------------------------------------------
        if row1 == 0
            img = zeros(col1,max_depth1);
            for i = 1:size(point3d_temp,1)
                img(data_3d_temp(i,2),data_3d_temp(i,3)) = point3d_temp(i,1);
            end
        elseif col1 == 0
            img = zeros(row1,max_depth1);
            for i = 1:size(point3d_temp,1)
                img(data_3d_temp(i,1),data_3d_temp(i,3)) = point3d_temp(i,2);
            end
        elseif max_depth1==0
            img = zeros(row1,col1);
            for i = 1:size(point3d_temp,1)
                img(data_3d_temp(i,1),data_3d_temp(i,2)) = point3d_temp(i,3);
            end
        end
        img_temp(:,:,n_f) = img;
    end
    %%
    top = inf;
    bottom = 0;
    left = inf;
    right = 0;
    
    for n_f = 1:frame
        [top1,bottom1,left1,right1] = bounding_box1(img_temp(:,:,n_f));
        if top > top1
            top = top1;
        end
        if bottom < bottom1
            bottom = bottom1;
        end
        if left > left1
            left = left1;
        end
        if right < right1
            right = right1;
        end
    end
    for n_f = 1:frame
        temp = img_temp(:,:,n_f);
        % 归一化大小--改大小为cnn输入大小，为了方便
        img_all(:,:,n_f,angle_num) = imresize(temp(top:bottom,left:right),[224,224]);
    end
end

