
function zWF = GetDynamicImages_mutil_view_mutil_t(depth) 

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
    zWF{i} = GetDynamicImages_test_gray(depth_temp);
end
