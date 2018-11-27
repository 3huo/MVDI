function o = boxoverlap(b1, b2)
% Compute the symmetric intersection over union overlap between a set of
% bounding boxes in a and a single bounding box in b.
%
% a  a matrix where each row specifies a bounding box
% b  a matrix where each row specifies a bounding box

% a(1) = b1(1);a(2) = b1(2);a(3) = b1(1)+b1(3);a(4) = b1(2)+b1(4);
% b(1) = b2(1);b(2) = b2(2);b(3) = b2(1)+b2(3);b(4) = b2(2)+b2(4);
a = b1;b=b2;
x1 = max(a(1), b(1));
y1 = max(a(2), b(2));
x2 = min(a(3), b(3));
y2 = min(a(4), b(4));

w = x2-x1+1;
h = y2-y1+1;
inter = w.*h;
aarea = (a(3)-a(1)+1) .* (a(4)-a(2)+1);
barea = (b(3)-b(1)+1) * (b(4)-b(2)+1);
% intersection over union overlap
o = inter ./ (aarea+barea-inter);
% set invalid entries to 0 overlap
o(w <= 0) = 0;
o(h <= 0) = 0;
