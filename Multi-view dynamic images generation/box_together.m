
% 根据一个视频序列的所有单个的boundingbox  合并成一个完整的boundingbox
%
function [box,mask] = box_together(box)

% 定义一个完整深度图像的大小：424 * 512
mask = zeros(424,512);
box_num = size(box,2);
start_frame = 1;
for frame = 1:size(box,1)% 是从第二帧开始合并的
     if frame == start_frame
        box_new = [];
         for i = 1:box_num % 多数时候为 1
             box_temp = box{frame,i};
             if ~isempty(box_temp)
                 box_new = box_temp;
             end
         end
         if isempty(box_new)
             start_frame = start_frame + 1;
             continue;
         end 
     end

     for i = 1:box_num % 多数时候为 1    
         box_old = box_new;          
         box_new = box{frame,i};
         if ~isempty(box_new) % 去除有的时候第二个框在后面的帧才出现
             box_new(1) = max(min(box_new(1),box_old(1)),1);%上顶点取小
             box_new(2) = max(min(box_new(2),box_old(2)),1);
             box_new(3) = min(max(box_new(3),box_old(3)),512);%下顶点取大
             box_new(4) = min(max(box_new(4),box_old(4)),424);
         else
             box_new = box_old; % 保持不变
         end  
     end
end

box = box_new;
mask(box(2):box(4),box(1):box(3)) = 1;        
