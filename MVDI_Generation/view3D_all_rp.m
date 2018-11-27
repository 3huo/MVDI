function [data_3d_r,data_3d_p] = view3D_all_rp(row_angle,col_angle,depth_angle,data_3d,view)
% 默认向正面投影
if nargin < 5
    view = 'view_xy';
elseif nargin > 5
    error('input is more.');
end
%
theta_d = depth_angle;
theta_r = row_angle;
theta_c = col_angle;
% 角度制转换为弧度制
theta_d = theta_d/360*2*pi;
theta_r = theta_r/360*2*pi;
theta_c = theta_c/360*2*pi;
% 算出旋转矩阵
Rr = [1 0 0; 0 cos(theta_r) sin(theta_r); 0 -sin(theta_r) cos(theta_r)];
Rc = [cos(theta_c) 0 sin(theta_c);0 1 0; -sin(theta_c) 0 cos(theta_c) ];
Rd = [cos(theta_d) sin(theta_d) 0; -sin(theta_d) cos(theta_d) 0; 0 0 1];
% 旋转后的数据:data_3d_r  变成 N*3 的矩阵
if isequal(size(data_3d,2),3),
    data_3d_r = data_3d*Rr*Rc*Rd;
else
    data_3d = data_3d';
    if isequal(size(data_3d,2),3),
        data_3d_r = data_3d*Rr*Rc*Rd;
    else
        error('Rx: Input XYZ must be [N,3] or [3,N] matrix.\n');
    end
end
%
% 投影面规定
v_xy = [1 0 0; 0 1 0; 0 0 1];
v_xd = [1 0 0; 0 1 0; 0 0 1];
v_yd = [1 0 0; 0 1 0; 0 0 1];
% 选择投影面
if strcmp(view,'view_xd')
    v_xd = [1 0 0; 0 0 0; 0 0 1];
elseif strcmp(view,'view_xy')
    v_xy = [1 0 0; 0 1 0; 0 0 0];
elseif strcmp(view,'view_yd')
    v_yd = [0 0 0; 0 1 0; 0 0 1];
else
    error('view choose is wrong,view should be:[view_xd,view_xy,view_yd]');
end
% 投影数据
data_3d_p = data_3d_r*v_xy*v_xd*v_yd;

