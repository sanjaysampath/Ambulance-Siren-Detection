% training model
clc; close all; clear all;

%............1)Reading all Audio Signals and finding their MFCC............


folder = 'C:\Users\Sanjay Sampath\Desktop\Programming\Matlab\Final_Datasets\Ambulance\';
audio_files = dir(fullfile(folder,'*.wav'));
var_audio = [];

for k=1:numel(audio_files)      % reads all the ambulance audio files
   [audio1,fs] = audioread(strcat(folder,audio_files(k).name));
   audio_mfcc=mfcc(audio1(:,1),fs);
   var_audio_1 = (rms(audio_mfcc));        %finds rms of ambulance signals
   var_audio = [var_audio;var_audio_1];
end

folder = 'C:\Users\Sanjay Sampath\Desktop\Programming\Matlab\Final_Datasets\Not_ambulance\';
audio_files = dir(fullfile(folder,'*.wav'));
var_audio_not = [];

for k=1:numel(audio_files)       % reads all the non-ambulance audio files
   [audio1,fs] = audioread(strcat(folder,audio_files(k).name));
   audio_mfcc=mfcc(audio1(:,1),fs);
   var_audio_1 = (rms(audio_mfcc));        %finds rms of non-ambulance signals
   var_audio_not = [var_audio_not;var_audio_1];
end


%...........................2)Training the Model...........................


x = [var_audio ; var_audio_not];
y = [ones(800,1) ; zeros(1000,1)];

rand_num =randperm(1800);

x_train = x(rand_num(1:1440),:);
y_train = y(rand_num(1:1440));

x_test = x(rand_num(1441:end),:);
y_test = y(rand_num(1441:end));

c = cvpartition(y_train,'k',5);
opts = statset('display','iter');

fun =@(train_data, train_labels, test_data, test_labels)... 
 sum(predict(fitcsvm(train_data, train_labels,'KernelFunction','rbf'), test_data) ~= test_labels);

[f1, history] = sequentialfs(fun,x_train,y_train,'cv',c, 'options' ,opts, 'nfeatures', 14);

x_train_with_best = x_train(:,f1);
Md1 = fitcsvm(x_train_with_best,y_train,'KernelFunction','rbf','OptimizeHyperparameters','auto',...
      'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
      'expected-improvement-plus','ShowPlots',true));
  
x_test_with_best = x_test(:,f1);
accuracy = sum(predict(Md1,x_test_with_best) == y_test)/length(y_test) * 100

save Md1;


%...........................3)Real time testing............................



clc; close all; clear all;

load Md1;

instrhwinfo('Bluetooth','HC-05');
bt = Bluetooth('HC-05', 1);
fopen(bt);
count_t = 0;
count_f = 0;

while true
    recObj = audiorecorder(44100,16,1);
    % disp("recording");
    recordblocking(recObj, 3);
    test_audio = getaudiodata(recObj);
    % disp("end of recording");
    filename = 'C:\Users\Sanjay Sampath\Desktop\Programming\Matlab\test_audio_record.wav';
    audiowrite(filename, test_audio, 44100);
    
    [audioIn_test,fsTest] = audioread(filename);
    [test_coeffs] = mfcc(audioIn_test(:,1),fsTest);
    testAudio_rms = rms(test_coeffs);
    
    x_real_testing_final = testAudio_rms(:,f1);
    b = predict(Md1,x_real_testing_final);
    
    if(b == 1)
        
        count_t = count_t + 1;
        if ( count_t  < 3)
            disp("Calibrating ambulance");
        end
        if ( count_t > 2)
            fprintf(bt,1);
            disp("ambulance");
            count_f = 0;
        end
    else
        
        count_f = count_f + 1;
        if ( count_f  < 3)
            disp("Calibrating not an ambulance");
        end
        
        if (count_f > 2 )
            count_t = 0;
            disp("not an ambulance");
            fprintf(bt,0);
        end
        
    end
    
end






