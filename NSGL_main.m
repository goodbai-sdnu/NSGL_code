clc
clear all;
warning off;

% load('Caltech101-7.mat');d = 'Caltech101-7';v = size(X,2);c = 7;n = size(Y,1);
% load('handwritten.mat');data = 'handwritten';v = size(X,2);c = 10;n = size(Y,1);
load('MSRC_V1_5views.mat');data = 'MSRC_v1_5v';v = size(X,2);c = 7;n = size(Y,1);
% load('youtube.mat');data = 'youtube';v = size(X,2);c = 11;n = size(Y,1);

for i = 1 :v
    for  j = 1:n
        X{i}(j,:) = ( X{i}(j,:) - mean( X{i}(j,:) ) ) / std( X{i}(j,:) ) ;
    end
end
XX = DataConcatenate(X);
[n,d] = size(XX);
XX = XX';

param.alpha = 1e+4;
param.beta = 1e+4;
param.gamma = 1e+4; 
param.lambda = 1e-4;
param.v = v;%view num
param.t = 2;
param.k = 5;% 
param.n = n;%data num
param.d = d;%feature dimension
param.c = c;%cluster num
param.NITER = 10;% iter num
l = 500;% the dimension of selected features; range[100,200,300,400,500]

[F,W,S,Wv] = NSGL(XX,X,param);

d = size(W,1);
[Wi_des, Wi_index] = sort(W,'descend');
Wi_idx = Wi_index(1:l,:);
Wi_l = Wi_des(1:l,:);

Wi_identify = zeros(l,d);
for i = 1:l
    Wi_identify(i,Wi_idx(i)) = 1;
end
Xw = Wi_identify*XX;
for m = 1:10
    [y] = litekmeans(Xw', param.k);
    result = ClusteringMeasure(Y,y);
    
    Fin_result(m,(1:3)) = result;
end
result1 = sum(Fin_result);
result2 = result1/10;

fprintf('dataset = %s, k = %d, alpha = %d, beta = %d,gamma = %d, lambda = %d, ',...
    data,param.k,param.alpha,param.beta,param.gamma,param.lambda);
fprintf('\n');
disp(['mean. ACC: ',num2str(result2(1))]);
disp(['mean. NMI: ',num2str(result2(2))]);
disp(['mean. Purity: ',num2str(result2(3))]);
fprintf('\n');
    