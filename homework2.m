clear all;
clc;
image=[1 3 9 9 8;2 1 3 7 3;3 6 0 6 4;6 8 2 0 5;2 9 2 6 0];
% reshape(image,5,5); %如果矩阵不符合图像标准，则需要reshape

image2 = mat2gray(image,[0,9]);  % 将矩阵转化为灰度图像 灰度范围0-9 共10级

disp('矩阵如下:');
disp(image); % 输出原图的数组

% 图像分布状况
[count,x]=imhist(image2,10);% x是灰度级的起始值 count是灰度级中的元素个数
disp('矩阵归一化之后:');
disp(image2)
disp('分布情况');
reshape(count,1,10)
disp('灰度等级');
reshape(x,1,10)

%进行图像均衡 均衡灰度为5
disp('进行图像均衡')
level=5;
[count1,x1]=imhist(image2,level);
disp('分布情况');
reshape(count1,1,level)
disp('灰度等级');
reshape(x1,1,level)

image3 = fix(image2/0.25);
image3 = mat2gray(image3,[0,4]);
disp('图像均衡后矩阵分布情况')
disp(image3*9)
figure
subplot(211),imshow(image2),title('befor');
figure
subplot(222),imshow(image3),title('after');