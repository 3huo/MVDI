
a = 'a28_'; s = 's01_'; 
v1 = 'v01';
v2 = 'v02';
v3 = 'v03';
v4 = 'v04';
name1 = ['E:\chenjun\dynamic\data\UWA3DII_Depth\ActionsNew\',a,s,v1,'.mat'];
name2 = ['E:\chenjun\dynamic\data\UWA3DII_Depth\ActionsNew\',a,s,v2,'.mat'];
name3 = ['E:\chenjun\dynamic\data\UWA3DII_Depth\ActionsNew\',a,s,v3,'.mat'];
name4 = ['E:\chenjun\dynamic\data\UWA3DII_Depth\ActionsNew\',a,s,v4,'.mat'];

choose = 20;
pathout = 'E:\chenjun\out2\4';
%%
img = load(name1);
img = img.curFrame;

img = double(img(:,:,choose));
img = img/max(max(img));
imwrite(img,[pathout,'\1.png']);
%%
img = load(name2);
img = img.curFrame;
img = double(img(:,:,choose));
img = img/max(max(img));
imwrite(img,[pathout,'\2.png']);
%%
img = load(name3);
img = img.curFrame;
img = double(img(:,:,choose));
img = img/max(max(img));
imwrite(img,[pathout,'\3.png']);
%%
img = load(name4);
img = img.curFrame;
img = double(img(:,:,choose));
img = img/max(max(img));
imwrite(img,[pathout,'\4.png']);
 
