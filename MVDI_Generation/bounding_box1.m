function [top,bottom,left,right] = bounding_box1(img)
% [top,bottom,left,right]
[row,col] = size(img);
top = 1;
bottom = row;
left = 1;
right = col;
% nn对于不同帧会有影响，选择是否该使用
nn = 1;
for i = 1:row
    if length(find(img(i,:)~=0)) > nn
        top = i;
        break
    end
end

for i = row:(-1):1
    if length(find(img(i,:)~=0)) > nn
        bottom = i;
        break
    end
end

for i = 1:col
    if length(find(img(:,i)~=0)) > nn
        left = i;
        break
    end
end

for i = col:(-1):1
    if length(find(img(:,i)~=0)) > nn
        right = i;
        break
    end
end

y = img(top:bottom, left:right);


