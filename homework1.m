list = [0,1,1,2,3,1,3,2;7,6,5,0,6,2,5,7;6,6,0,1,1,6,4,3;2,7,6,5,5,3,6,5;...
    3,2,2,7,2,6,6,1;2,6,5,0,2,7,5,0;1,2,3,2,1,2,1,2;3,2,1,3,1,1,2,2];
disp(numel(list))
a=[];
for i =0:7
    a(end+1) = length(find(list==i));
end
disp(a)
h = [0,1,2,3,4,5,6,7];
bar(h,a)
axis([-1,8,0,20])
title('灰度直方图')
xlabel('样本数据');
ylabel('频数');
% 林啸涛1795131028