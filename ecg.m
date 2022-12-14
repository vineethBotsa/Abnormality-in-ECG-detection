clc;
k = 1;
featurevector = zeros(119, 30);
groupvector = zeros(119,1);

%%%%%%% data extraction from normal ecg signal %%%%%%%%%%%%%

for n=1:7
filename = sprintf('%d.mat',n);
load(filename);
for i=1:2
x = val(i,:);

[c, l] = wavedec(x, 5, 'db10');
[d1, d2, d3, d4, d5] = detcoef(c,l, [1,2,3,4,5]); 

a1 = appcoef(c,l,'db10', 1);
a2 = appcoef(c,l,'db10', 2);
a3 = appcoef(c,l,'db10', 3);
a4 = appcoef(c,l,'db10', 4);
a5 = appcoef(c,l,'db10', 5);

%%%%%%-------------------------------------------------%%%%%

%%%%%%%%% feature extraction  for normal ecg signal%%%%%%%%

skd1 = skewness(d1);skd2 = skewness(d2);skd3 = skewness(d3);skd4 = skewness(d4);skd5 = skewness(d5);
ska1 = skewness(a1);ska2 = skewness(a2);ska3 = skewness(a3);ska4 = skewness(a4);ska5 = skewness(a5);

kud1 = kurtosis(d1);kud2 = kurtosis(d2);kud3 = kurtosis(d3);kud4 = kurtosis(d4);kud5 = kurtosis(d5);
kua1 = kurtosis(a1);kua2 = kurtosis(a2);kua3 = kurtosis(a3);kua4 = kurtosis(a4);kua5 = kurtosis(a5);

sdd1 = std(d1);sdd2 = std(d2);sdd3 = std(d3);sdd4 = std(d4);sdd5 = std(d5);
sda1 = std(a1);sda2 = std(a2);sda3 = std(a3);sda4 = std(a4);sda5 = std(a5);

vector = [skd1 skd2 skd3 skd4 skd5 ska1 ska2 ska3 ska4 ska5 kud1 kud2 kud3 kud4 kud5 kua1 kua2 kua3 kua4 kua5 sdd1 sdd2 sdd3 sdd4 sdd5 sda1 sda2 sda3 sda4 sda5];
featurevector(k, :) = vector;
gvector = [1];
groupvector(k, :) = gvector;
k = k+1;

%%%%%%%%%%%---------------------------------------%%%%%%%%%%
end

end

%%%%%%%% data extraction from abnormal ecg signals %%%%%%%%%
for j=19:25

filename = sprintf('%d.mat',j);
load(filename);
for i=1:15
x = val(i,:);

[c, l] = wavedec(x, 5, 'db10');
[d1, d2, d3, d4, d5] = detcoef(c,l, [1,2,3,4,5]); 

a1 = appcoef(c,l,'db10', 1);
a2 = appcoef(c,l,'db10', 2);
a3 = appcoef(c,l,'db10', 3);
a4 = appcoef(c,l,'db10', 4);
a5 = appcoef(c,l,'db10', 5);

%%%%%%%%%------------------------------------------%%%%%%%%%


%%%%%%%%% feature extraction from abnormal ecg signals %%%%%%%%

skd1 = skewness(d1);skd2 = skewness(d2);skd3 = skewness(d3);skd4 = skewness(d4);skd5 = skewness(d5);
ska1 = skewness(a1);ska2 = skewness(a2);ska3 = skewness(a3);ska4 = skewness(a4);ska5 = skewness(a5);

kud1 = kurtosis(d1);kud2 = kurtosis(d2);kud3 = kurtosis(d3);kud4 = kurtosis(d4);kud5 = kurtosis(d5);
kua1 = kurtosis(a1);kua2 = kurtosis(a2);kua3 = kurtosis(a3);kua4 = kurtosis(a4);kua5 = kurtosis(a5);

sdd1 = std(d1);sdd2 = std(d2);sdd3 = std(d3);sdd4 = std(d4);sdd5 = std(d5);
sda1 = std(a1);sda2 = std(a2);sda3 = std(a3);sda4 = std(a4);sda5 = std(a5);

vector = [skd1 skd2 skd3 skd4 skd5 ska1 ska2 ska3 ska4 ska5 kud1 kud2 kud3 kud4 kud5 kua1 kua2 kua3 kua4 kua5 sdd1 sdd2 sdd3 sdd4 sdd5 sda1 sda2 sda3 sda4 sda5];
featurevector(k, :) = vector;
gvector = [0];
groupvector(k, :) = gvector;
k = k+1;

%%%%%%%%----------------------------------------------%%%%%%%%%

end

end    

%%%%% Support Vector Machine Model making %%%%%

SVMModel = fitcsvm(featurevector,groupvector);

%%%%%-------------------------------------%%%%%


%%%%% Using 45 abnormal ecg signals from the testing set %%%%%    
test_abnormal_vector = zeros(45,1);
h=1;
for i=26:28
filename = sprintf('%d.mat',i);
load(filename);
for j=1:15
x = val(j,:);

[c, l] = wavedec(x, 5, 'db10');
[d1, d2, d3, d4, d5] = detcoef(c,l, [1,2,3,4,5]); 

a1 = appcoef(c,l,'db10', 1);
a2 = appcoef(c,l,'db10', 2);
a3 = appcoef(c,l,'db10', 3);
a4 = appcoef(c,l,'db10', 4);
a5 = appcoef(c,l,'db10', 5);

skd1 = skewness(d1);skd2 = skewness(d2);skd3 = skewness(d3);skd4 = skewness(d4);skd5 = skewness(d5);
ska1 = skewness(a1);ska2 = skewness(a2);ska3 = skewness(a3);ska4 = skewness(a4);ska5 = skewness(a5);

kud1 = kurtosis(d1);kud2 = kurtosis(d2);kud3 = kurtosis(d3);kud4 = kurtosis(d4);kud5 = kurtosis(d5);
kua1 = kurtosis(a1);kua2 = kurtosis(a2);kua3 = kurtosis(a3);kua4 = kurtosis(a4);kua5 = kurtosis(a5);

sdd1 = std(d1);sdd2 = std(d2);sdd3 = std(d3);sdd4 = std(d4);sdd5 = std(d5);
sda1 = std(a1);sda2 = std(a2);sda3 = std(a3);sda4 = std(a4);sda5 = std(a5);

vector = [skd1 skd2 skd3 skd4 skd5 ska1 ska2 ska3 ska4 ska5 kud1 kud2 kud3 kud4 kud5 kua1 kua2 kua3 kua4 kua5 sdd1 sdd2 sdd3 sdd4 sdd5 sda1 sda2 sda3 sda4 sda5];

[label,score] = predict(SVMModel,vector);
test_abnormal_vector(h,:) = label;
h = h+1;

end
end

%%%%%%--------------------------------------------------%%%%%%


%%%%%% Using 22 normal ecg signals from the testing set %%%%% 
test_normal_vector = zeros(22,1);
h=1;
for i=8:18
filename = sprintf('%d.mat',i);
load(filename);
for j=1:2
x = val(j,:);

[c, l] = wavedec(x, 5, 'db10');
[d1, d2, d3, d4, d5] = detcoef(c,l, [1,2,3,4,5]); 

a1 = appcoef(c,l,'db10', 1);
a2 = appcoef(c,l,'db10', 2);
a3 = appcoef(c,l,'db10', 3);
a4 = appcoef(c,l,'db10', 4);
a5 = appcoef(c,l,'db10', 5);

skd1 = skewness(d1);skd2 = skewness(d2);skd3 = skewness(d3);skd4 = skewness(d4);skd5 = skewness(d5);
ska1 = skewness(a1);ska2 = skewness(a2);ska3 = skewness(a3);ska4 = skewness(a4);ska5 = skewness(a5);

kud1 = kurtosis(d1);kud2 = kurtosis(d2);kud3 = kurtosis(d3);kud4 = kurtosis(d4);kud5 = kurtosis(d5);
kua1 = kurtosis(a1);kua2 = kurtosis(a2);kua3 = kurtosis(a3);kua4 = kurtosis(a4);kua5 = kurtosis(a5);

sdd1 = std(d1);sdd2 = std(d2);sdd3 = std(d3);sdd4 = std(d4);sdd5 = std(d5);
sda1 = std(a1);sda2 = std(a2);sda3 = std(a3);sda4 = std(a4);sda5 = std(a5);

vector = [skd1 skd2 skd3 skd4 skd5 ska1 ska2 ska3 ska4 ska5 kud1 kud2 kud3 kud4 kud5 kua1 kua2 kua3 kua4 kua5 sdd1 sdd2 sdd3 sdd4 sdd5 sda1 sda2 sda3 sda4 sda5];

[label,score] = predict(SVMModel,vector);
test_normal_vector(h,:) = label;
h = h+1;

end
end

%%%%%%--------------------------------------------------%%%%%%
