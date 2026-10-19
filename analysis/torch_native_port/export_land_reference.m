addpath('D:\MyProjects\AI细胞\Data\7fb\7fb-959277c48267\Land_model_matlab\release');
rng(17); n=128; Y=zeros(n,7); C=zeros(n,1); L=zeros(n,1); DL=zeros(n,1); O=zeros(n,10);
for i=1:n
 Y(i,:)=[0.05*rand,0.2*rand,0.05+0.9*rand,0.1*rand,0.2*randn,0.2*randn,0.2*rand]; C(i)=0.1+1.5*rand; L(i)=1+0.15*rand; DL(i)=0.02*randn;
 [d,T,Ta,Tp]=Land2016(0,Y(i,:),C(i),'intact',L(i),DL(i),[2.3,-2.4]); O(i,:)=[d,T,Ta,Tp];
end
save('D:\MyProjects\AI细胞\analysis\torch_native_port\land_reference.mat','Y','C','L','DL','O','-v7');
