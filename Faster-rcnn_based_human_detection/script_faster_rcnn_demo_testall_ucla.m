function script_faster_rcnn_demo_testall_ucla()
close all;clc;clear mex;clear is_valid_handle; % to clear init_key
run(fullfile(fileparts((mfilename('fullpath'))), 'startup'));
%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe_faster_rcnn';
opts.gpu_id                 = 1;%auto_select_gpu;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

opts.per_nms_topN           = 6000;
opts.nms_overlap_thres      = 0.7;
opts.after_nms_topN         = 300;
opts.use_gpu                = true;
opts.test_scales            = 600;

%% -------------------- INIT_MODEL --------------------
model_dir                   = fullfile(pwd, 'output_ucla', 'faster_rcnn_final', 'faster_rcnn_VOC2007_ZF'); %% ZF

proposal_detection_model    = load_proposal_detection_model(model_dir);
proposal_detection_model.conf_proposal.test_scales = opts.test_scales;
proposal_detection_model.conf_detection.test_scales = opts.test_scales;
if opts.use_gpu
    proposal_detection_model.conf_proposal.image_means = gpuArray(proposal_detection_model.conf_proposal.image_means);
    proposal_detection_model.conf_detection.image_means = gpuArray(proposal_detection_model.conf_detection.image_means);
end
% proposal net
rpn_net = caffe.Net(proposal_detection_model.proposal_net_def, 'test');
rpn_net.copy_from(proposal_detection_model.proposal_net);
% fast rcnn net
fast_rcnn_net = caffe.Net(proposal_detection_model.detection_net_def, 'test');
fast_rcnn_net.copy_from(proposal_detection_model.detection_net);
% set gpu/cpu
if opts.use_gpu
    caffe.set_mode_gpu();
else
    caffe.set_mode_cpu();
end
%% -------------------- WARM UP --------------------
% the first run will be slower; use an empty image to warm up
for j = 1:2 % we warm up 2 times
    im = uint8(ones(375, 500, 3)*128);
    if opts.use_gpu
        im = gpuArray(im);
    end
    [boxes, scores]             = proposal_im_detect(proposal_detection_model.conf_proposal, rpn_net, im);
    aboxes                      = boxes_filter([boxes, scores], opts.per_nms_topN, opts.nms_overlap_thres, opts.after_nms_topN, opts.use_gpu);
    if proposal_detection_model.is_share_feature
        [boxes, scores,~,~]             = fast_rcnn_conv_feat_detect(proposal_detection_model.conf_detection, fast_rcnn_net, im, ...
            rpn_net.blobs(proposal_detection_model.last_shared_output_blob_name), ...
            aboxes(:, 1:4), opts.after_nms_topN);
    else
        [boxes, scores]             = fast_rcnn_im_detect(proposal_detection_model.conf_detection, fast_rcnn_net, im, ...
            aboxes(:, 1:4), opts.after_nms_topN);
    end
end
%% -------------------- TESTING --------------------
num = 1;
path1 = 'F:\chenjun\dynamic\data\NUCLA3D';
path1_name = dir([path1,'\a*']);
for s = 1:length(path1_name)
    path2 = dir([path1,'\',path1_name(s).name,'\*.png']);
    if exist(['all_bounding_box_ucla\',path1_name(s).name,'.mat'],'file')
        continue;
    end
    
    frame = 1;
    box_all = [];
    for n = 1:3:length(path2) % 对视频跳帧检测，加快检测速度
        num
        num = num + 1;
        clear im;
        im = imread([path1,'\',path1_name(s).name,'\',path2(n).name]);
        im = double(gray2rgb(im)); %单通道转化为三通道图像才能输入模型中
        if opts.use_gpu
            im = gpuArray(im);
        end
        % test proposal
        th = tic();
        [boxes, scores]             = proposal_im_detect(proposal_detection_model.conf_proposal, rpn_net, im);
        t_proposal = toc(th);
        th = tic();
        aboxes                      = boxes_filter([boxes, scores], opts.per_nms_topN, opts.nms_overlap_thres, opts.after_nms_topN, opts.use_gpu);
        t_nms = toc(th);
        % test detection
        th = tic();
        if proposal_detection_model.is_share_feature
            [boxes, scores, feat6,feat7]             = fast_rcnn_conv_feat_detect(proposal_detection_model.conf_detection, fast_rcnn_net, im, ...
                rpn_net.blobs(proposal_detection_model.last_shared_output_blob_name), ...
                aboxes(:, 1:4), opts.after_nms_topN);
        else
            [boxes, scores]             = fast_rcnn_im_detect(proposal_detection_model.conf_detection, fast_rcnn_net, im, ...
                aboxes(:, 1:4), opts.after_nms_topN);
        end
        t_detection = toc(th);
        % visualize
        classes = proposal_detection_model.classes;
        % 检测出来的所有目标框
        boxes_cell = cell(length(classes), 1);
        
        thres = 0.95; %设置检测为目标的置信度，大小决定了目标框的个数。
        for i = 1:length(boxes_cell)
            boxes_cell{i} = [boxes(:, (1+(i-1)*4):(i*4)), scores(:, i)];
            if isempty(boxes_cell{1})
                continue;
            end
            boxes_cell{i} = boxes_cell{i}(nms(boxes_cell{i}, 0.3), :);
            I = boxes_cell{i}(:, 5) >= thres;
            boxes_cell{i} = boxes_cell{i}(I, :);
        end
        if isempty(boxes_cell{1})
            continue;
        end
        
        for ii = 1:1%size(boxes_cell{1},1)
            box_all{frame} = boxes_cell{1}(ii,:);
            frame = frame + 1;
        end
        %             showboxes(im, boxes_cell, classes, 'default');%voc
    end
    % 对目标框进行范围限制与融合形成视频的运动范围边框
    if isempty(box_all)
        boxes_cell = {[1 1 320 240 1]};  % 如果没有框，默认整个图像大小为框
    else
        [box,~] = box_together(box_all);
        boxes_cell = {single(box)};
    end
    %         figure(1);
    %         showboxes(im, boxes_cell, classes, 'default');
    % % 保存视频的运动边框
    save(['all_bounding_box_ucla\',path1_name(s).name],'boxes_cell');
end
fprintf('mean time: %.3fs\n', mean(running_time));
caffe.reset_all();
clear mex;
end
function proposal_detection_model = load_proposal_detection_model(model_dir)
ld                          = load(fullfile(model_dir, 'model'));
proposal_detection_model    = ld.proposal_detection_model;
clear ld;
proposal_detection_model.proposal_net_def ...
    = fullfile(model_dir, proposal_detection_model.proposal_net_def);
proposal_detection_model.proposal_net ...
    = fullfile(model_dir, proposal_detection_model.proposal_net);
proposal_detection_model.detection_net_def ...
    = fullfile(model_dir, proposal_detection_model.detection_net_def);
proposal_detection_model.detection_net ...
    = fullfile(model_dir, proposal_detection_model.detection_net);
end

function aboxes = boxes_filter(aboxes, per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu)
% to speed up nms
if per_nms_topN > 0
    aboxes = aboxes(1:min(length(aboxes), per_nms_topN), :);
end
% do nms
if nms_overlap_thres > 0 && nms_overlap_thres < 1
    aboxes = aboxes(nms(aboxes, nms_overlap_thres, use_gpu), :);
end
if after_nms_topN > 0
    aboxes = aboxes(1:min(length(aboxes), after_nms_topN), :);
end
end
