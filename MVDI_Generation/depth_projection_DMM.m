function zWF = depth_projection_DMM(depth)

width = floor(size(depth,3)/5);% 5 dynamic×éÍ¼
s = [1,width,2*width,3*width,4*width,5*width,6*width] ;
start = [1,s(1),s(2),s(3),s(4)];
stop = [size(depth,3),s(3),s(4),s(5),s(6)];

num_mutil = length(start);

for i = 1:num_mutil
    temp_s = start(i);
    temp_t = stop(i);
    %²àÃæ
    depth_temp = depth(:,:,temp_s:temp_t);
    
    F = zeros(size(depth_temp,1), size(depth_temp,2));
    front = depth_temp(:,:,1);
    for D = 2:size(depth_temp,3)
        front_pre = front;
        front = depth_temp(:,:,D);
        F = F + abs(front - front_pre);
%         tmp = abs(front - front_pre);
%         tmp(tmp>50)=1;
%         tmp(tmp<=50)=0;
%         F = F + tmp;
    end
    zWF{i} = F;
end
