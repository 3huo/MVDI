function Y = faster_dynamic_image_rgb(X)
% approximate rank pooling
% X : 三通道的RGB视频序列：m * n * 3 * numberframe
% 输出 Y : 单张RGB图（动态图）: m * n * 3

sz = size(X);
N = sz(4);
fw = zeros(1,N);
for i=1:N
    fw(i) = sum((2*(i:N)-N-1) ./ (i:N));
end
Y =  sum(bsxfun(@times,X,...
    reshape(single(fw),[1 1 1 N])),4);