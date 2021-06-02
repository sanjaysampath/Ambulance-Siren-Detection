clc; close all; clear all;
 
folder = 'C:\Users\Sanjay Sampath\Desktop\Programming\Matlab\ambulance_dataset\ambulance\';
audio_files = dir(fullfile(folder,'*.wav'));
[audio1,fs] = audioread('C:\Users\Sanjay Sampath\Desktop\Programming\Matlab\ambulance_dataset\ambulance\sound_1.wav');
   audio_mfcc = mfcc(audio1(:,1),fs);
   rms_audio = rms(audio_mfcc);
for k=1:numel(audio_files)-1
   [audio1,fs] = audioread(strcat(folder,audio_files(k+1).name));
   audio_mfcc=mfcc(audio1(:,1),fs);

    rms_audio_1 = (rms(audio_mfcc));
    rms_audio = [rms_audio;rms_audio_1];
end

folder = 'C:\Users\Sanjay Sampath\Desktop\Programming\Matlab\ambulance_dataset\firetruck\';
audio_files = dir(fullfile(folder,'*.wav'));
[audio1,fs] = audioread('C:\Users\Sanjay Sampath\Desktop\Programming\Matlab\ambulance_dataset\firetruck\dog1.wav');
   audio_mfcc = mfcc(audio1(:,1),fs);
   rms_audio_not = rms(audio_mfcc);
for k=1:numel(audio_files)-1
   [audio1,fs] = audioread(strcat(folder,audio_files(k+1).name));
   audio_mfcc=mfcc(audio1(:,1),fs);

    rms_audio_1 = (rms(audio_mfcc));
    rms_audio_not = [rms_audio_not;rms_audio_1];
end
x = [rms_audio ; rms_audio_not];
y = [ones(200,1) ; zeros(401,1)];

rand_num =randperm(601);

x_train = x(rand_num(1:500),:);
y_train = y(rand_num(1:500));

x_test = x(rand_num(501:end),:);
y_test = y(rand_num(501:end));

c = cvpartition(y_train,'k',5);
opts = statset('display','iter');

fun =@(train_data, train_labels, test_data, test_labels)... 
 sum(predict(fitcsvm(train_data, train_labels,'KernelFunction','rbf'), test_data) ~= test_labels);

[f1, history] = sequentialfs(fun,x_train,y_train,'cv',c, 'options' ,opts);

x_train_with_best = x_train(:,f1);
Md1 = fitcsvm(x_train_with_best,y_train,'KernelFunction','rbf','OptimizeHyperparameters','auto',...
      'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
      'expected-improvement-plus','ShowPlots',true));
  
x_test_with_best = x_test(:,f1);
sum(predict(Md1,x_test_with_best) == y_test)/length(y_test) * 100



[audio1,fs] = audioread('C:\Users\Sanjay Sampath\Desktop\Programming\Matlab\Test\test_audio7.wav');
[coeffs1] = mfcc(audio1(:,1),fs);
rms_test= rms(coeffs1);


% [audio1,fs] = audioread('C:\Users\Sanjay Sampath\Desktop\Programming\Matlab\Test\dog2.wav');
% [coeffs1] = mfcc(audio1(:,1),fs);
% real_test = rms(coeffs1);
% 
% x_real_testing_final = real_test(:,f1);
% predict(Md1,x_real_testing_final)







