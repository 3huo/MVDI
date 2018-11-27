
clc
clear
% add liblinear path for dynamic images generation
addpath(genpath('E:\NTU RGBD\test_ntu\liblinear-1.96'));
% the human skeletons data path
path_sk = 'E:\NTU RGBD\nturgb+d_skeletons';

S = dir('E:\NTU RGBD\data_depth_w\n*');
for i = 1:length(S)  % ������ȡ���г����µ����ݼ���s001-s017�� 
    
    temp = ['E:\NTU RGBD\data_depth_w\',S(i).name,'\nturgb+d_depth'];
    path_s = [temp,'\S*'];
    video_n = dir(path_s);
    for j = 1:length(video_n) % �ɸ�,������ȡ��S* �µ�������Ƶ
        disp(['i=',num2str(i),';j=',num2str(j)]);
        % �жϸ���Ƶ�Ƿ��Ѿ���ת��
        if exist(['dynamic_mutil_general1\', video_n(j).name,'_',num2str(11),'_',num2str(5),'.jpg'],'file')
            continue;
        end
        %-------ģ�������˶��߿�/ʹ��ģ�����ɵı߿�----------
        depth_name = [temp,'\',video_n(j).name];
        sk_name = [path_sk,'\',video_n(j).name,'.skeleton'];
        % ��ȡһ����Ƶ�������ͼ��
        depth_pic= dir([depth_name,'\*.png']);
        % ��ȡ����Ƶ���ж�Ӧ�ĹǼ�
        box = get_bounding_box_skelen(sk_name);
        %-------�ж�ģ�������˶��߿�Ĵ�����-----
        if isempty(box)
            continue;
        else
            % ���ݹǼ��������յ�bounding box����Ĥͼ
            [box_all,mask] = box_together(box);    
            %---------------���ɴ����˶��߿����Ĥ��̬ͼ------------------
            tic
            disp('������Ĥ��̬ͼ��')      
            depth = [];
            for k = 1:length(depth_pic)
                img_path = [temp,'\',video_n(j).name,'\',depth_pic(k).name];
                img = double(imread(img_path));     
                depth(:,:,k) = img(box_all(2):box_all(4),box_all(1):box_all(3)); 
%                 depth(:,:,k) = img;  % ֱ��ʹ��ԭ��Ƶ�������˶��߿�
            end
            toc
            % -------------�涨�Ķ��ӽ�ͶӰ�ӽ�----------------
            view_angle = [-90 -40 -20 -10 -5 0 5 10 20 40 90];
            img = [];
            tic
            disp('���ɶ�view��')
            % -----������֡���ж��ӽ�ͶӰ�任-------
            img = change_view_all(view_angle,depth);
            toc
             %------�Զ��ӽ���Ƶ���ж�ʱ��εĶ�̬ͼ����---
            tic
            disp('dynamic image��')
            for n_d = 1:size(img,4) % ���������ӽ�
                dynamic_depth = [];
                % ÿ���ӽǵ���Ƶ����5��ʱ��εĶ�̬ͼ����
                dynamic_depth = GetDynamicImages_mutil_view_mutil_t(img(:,:,:,n_d));
                % �Զ��ӽǶ�ʱ��ζ�̬ͼ���д洢
                for jj = 1:length(dynamic_depth)
                    temp_d = dynamic_depth{jj};
                    imwrite(temp_d,['dynamic_mutil_general\', video_n(j).name,'_',...
                        num2str(n_d),'_',num2str(jj),'.jpg']);
                end
            end
            toc  
            clear depth 
        end
    end
end