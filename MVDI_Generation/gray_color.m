function out = gray_color(img)
I = img;
I=double(I);
[m,n]=size(I);
c=max(max(I));
for i=1:m
    for j=1:n
if I(i,j)<=c/4
    R(i,j)=0;
    G(i,j)=4*I(i,j);
    B(i,j)=c;
else if I(i,j)<=c/2
        R(i,j)=0;
        G(i,j)=c;
        B(i,j)=-4*I(i,j)+2*c;
    else if I(i,j)<=3*c/4
            R(i,j)=4*I(i,j)-2*c;
            G(i,j)=c;
            B(i,j)=0;
        else
            R(i,j)=c;
            G(i,j)=-4*I(i,j)+4*c;
            B(i,j)=0;
        end
    end
end
    end
end
for i=1:m
    for j=1:n
        out(i,j,1)=R(i,j);
        out(i,j,2)=G(i,j);
        out(i,j,3)=B(i,j);
    end
end
out=out/c;
% figure(1),imshow(out)