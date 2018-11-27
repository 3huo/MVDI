function Y = faster_dynamic_image(X)
% author:  
% approximate rank pooling
% X : ��ͨ����Ƶ���У�m * n * numberframe
% ��� Y : ���ŻҶ�ͼ����̬ͼ��: m * n

sz = size(X);
% pool among frames
N = sz(3);
% magic numbers
fw = zeros(1,N);
for i=1:N
    fw(i) = sum((2*(i:N)-N-1) ./ (i:N));
end
Y =  sum(bsxfun(@times,X,reshape(single(fw),[1 1 N])),3);

