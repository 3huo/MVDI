clc
clear

data_file = 'E:\chenjun\test_ntu\dynamic_mutil_general';
data_file_name = dir([data_file,'\*.jpg']);

test_file = 'E:\chenjun\dynamic\data\ntu_all\test';
train_file = 'E:\chenjun\dynamic\data\ntu_all\train';

file_train = 'train_label.txt';
f_train = fopen(file_train,'wt');

file_test = 'test_label.txt';
f_test = fopen(file_test,'wt');

for i = 1:length(data_file_name)
    temp_name = data_file_name(i).name;
    index = str2num(temp_name(8));
    if index == 1 %test
        copyfile([data_file,'\',temp_name], [test_file,'\',temp_name]);
%         label = temp_name(19:20);
%         fprintf(f_test,'%s\t%s\n',temp_name,label);
    else
        copyfile([data_file,'\',temp_name], [train_file,'\',temp_name]);
%         label = temp_name(19:20);
%         fprintf(f_train,'%s\t%s\n',temp_name,label);
    end
end
    
fclose(f_test);    %关闭txt文件   
fclose(f_train);    %关闭txt文件
       