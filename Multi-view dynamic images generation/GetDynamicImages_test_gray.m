%  ���ԣ�����Ϊ��ά�������ݣ�m*n*k
%          �Ҷȵ���Ƶ����  ����  ���ͼ�����Ƶ
%
function zWF = GetDynamicImages_test_gray(data) 
   % zWF is the forward dynamic sequence
    x = data;
    [w,h,len] = size(x);
    x = reshape(x,h*w,len);x =x'; %��ͨ��   
    im_WF = processVideo(x,w,h);
    zWF = linearMapping(im_WF);
end

function im_WF = processVideo(x,w,h)
     WF = genRankPoolImageRepresentation(single(x),200);   %10 
     im_WF = reshape(WF,w,h);  
end

function W_fow = genRankPoolImageRepresentation(data,CVAL)
    OneToN = [1:size(data,1)]';    
    Data = cumsum(data);
    Data = Data ./ repmat(OneToN,1,size(Data,2));
    %% ע����Եķ�������  ������ط�
    W_fow = liblinearsvr(getNonLinearity(Data,'ssr'),CVAL,2); 
%     plot(getNonLinearity(Data,'ssr')*W_fow); % �������
    clear Data; 			             
end

function w = liblinearsvr(Data,C,normD)
    if normD == 2
        Data = normalizeL2(Data);
    end    
    if normD == 1
        Data = normalizeL1(Data);
    end    
    N = size(Data,1);
    Labels = [1:N]';
    model = train(double(Labels), sparse(double(Data)),sprintf('-c %1.6f -s 11 -q',C) );
    w = model.w';    
end

function Data = getNonLinearity(Data,nonLin)    
    switch nonLin            
        case 'ssr'
            Data = sign(Data).*sqrt(abs(Data));       
    end
end

function x = normalizeL2(x)
    v = sqrt(sum(x.*conj(x),2));
    v(find(v==0))=1;
    x=x./repmat(v,1,size(x,2));
end

% ��x ӳ�䵽0-255��uint8������
function x = linearMapping(x)
    minV = min(x(:));
    maxV = max(x(:));
    x = x - minV;
    x = x ./ (maxV - minV);
    x = x .* 255;
    x = uint8(x);
end
