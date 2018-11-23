%  测试：输入为四维矩阵数据：m*n*3*k
%
function zWF = GetDynamicImages_test_single(data)
  opts.window = size(data,4);%取帧长度
  
%    opts.window = 50;%取帧长度
   opts.stride = 30;%每隔多少帧取一次    
   % zWF is the forward dynamic sequence
   % zWR is the reverse dynamic sequence
  zWF = getVideoImageSegmentsFull(data,opts);        
    
end
 

function zWF = getVideoImageSegmentsFull(data,opts)    
    x = data;
    [w,h,~,len] = size(x);
    x = reshape(x,h*w*3,len);x =x';        
    Window_Size = opts.window; 
    stride = opts.stride;
    if len > Window_Size
        sStart = 1:stride:len-Window_Size+1;
        sEnd = Window_Size:stride:len;
        sEnd(end) = len;
    else
        sStart = 1;
        sEnd = len;
    end    
    segments = numel(sStart);    
    % pick only the middle segments
    if segments > 6
        sStart = sStart(round(segments/2)-3:round(segments/2)+2);
        sEnd = sEnd(round(segments/2)-3:round(segments/2)+2);
        segments = numel(sStart);    
    end
    zWF = zeros(w,h,3,segments,'uint8');
    for s = 1 : segments
        st = sStart(s);
        send = sEnd(s);
        im_WF = processVideo(x(st:send,:),w,h);
        im_WF = linearMapping(im_WF);
        zWF(:,:,:,s)  = im_WF;
    end    
end

function im_WF = processVideo(x,w,h)
     WF = genRankPoolImageRepresentation(single(x),10);    
     im_WF = reshape(WF,w,h,3);  
end

function W_fow = genRankPoolImageRepresentation(data,CVAL)
    OneToN = [1:size(data,1)]';    
    Data = cumsum(data);
    Data = Data ./ repmat(OneToN,1,size(Data,2));
    %% 注意测试的方向问题  在这个地方
    W_fow = liblinearsvr(getNonLinearity(Data,'ssr'),CVAL,2); 
%     plot(getNonLinearity(Data,'ssr')*W_fow); % 画出结果
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

% 将x 映射到0-255的uint8型数据
function x = linearMapping(x)
    minV = min(x(:));
    maxV = max(x(:));
    x = x - minV;
    x = x ./ (maxV - minV);
    x = x .* 255;
    x = uint8(x);
end
