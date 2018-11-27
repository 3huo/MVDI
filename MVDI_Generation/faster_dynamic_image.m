function Y = faster_dynamic_image(X)
% author:  
% approximate rank pooling
% X : 单通道视频序列：m * n * numberframe
% 输出 Y : 单张灰度图（动态图）: m * n

sz = size(X);
% pool among frames
N = sz(3);
% magic numbers
fw = zeros(1,N);
for i=1:N
    fw(i) = sum((2*(i:N)-N-1) ./ (i:N));
end
Y =  sum(bsxfun(@times,X,reshape(single(fw),[1 1 N])),3);

