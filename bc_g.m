%% Clear everything 
clearvars -global;
clear all; close all; clc;

addpath('~/ANNLib/');
addpath('~/BookClub/BC2EClGen/');


%% Dataset root folder template and suffix
%trDataFolder = '~/data/GTS/GTSRB/Final_Training/Images';

dataFolderTmpl = '~/data/BC2E_Sfx';
dataFolderSfx = '1072x712';

[trImageDS, eTrainLab, mTrainLab, sTrainLab, trainDataSetFolders] = createBCFuzzTrain(dataFolderTmpl, dataFolderSfx, @readFunctionGTS);

%
n = 80;
m = 80;
[l,~] = size(trImageDS.Files);

XTrain = zeros([n,m,3,l]);
XTrainF = zeros([n*m*3,l]);
XTrainMean = zeros([l,1]);
XTrainStd = zeros([l,1]);

for i=1:l
    XTrain(:,:,:,i) = double(readimage(trImageDS,i));% / 255.;
    %image(int8(XTrain(:,:,:,i)));

    XTrainF(:,i) = reshape(XTrain(:,:,:,i), [n*m*3,1] );
    XTrainMean(i) = mean(XTrainF(:,i), 1);
    XTrainStd(i) = std(XTrainF(:,i), 0, 1);
end
XTrainF = generic_mean_std_scale2D(XTrainF, XTrainMean, XTrainStd);
YTraine = trImageDS.Labels;


%% Emotion Injection one-hot
[inje,~] = size(unique(YTraine));

YTrainFe = zeros(inje,l);

YTrainDe = unique(YTraine);
YTrainIe = grp2idx(YTrainDe);

for j = 1:inje
    YTrainFe(j, grp2idx(YTraine)==(j)) = 1;
end

% one-hot injection
XYTrainF = vertcat(XTrainF,YTrainFe);


% Makeup Injection one-hot
nmIdx = contains(mTrainLab, 'NM');
mTrainLab(nmIdx) = 'NM';
YTrainm = categorical(mTrainLab);
[injm,~] = size(unique(YTrainm));

YTrainFm = zeros(injm,l);

YTrainDm = unique(YTrainm);
YTrainIm = grp2idx(YTrainDm);

for j = 1:injm
    YTrainFm(j, grp2idx(YTrainm)==(j)) = 1;
end

% one-hot injection
XYTrainF = vertcat(XYTrainF,YTrainFm);


% Subject Injection one-hot
YTrains = categorical(sTrainLab);
[injs,~] = size(unique(YTrains));

YTrainFs = zeros(injs,l);

YTrainDs = unique(YTrains);
YTrainIs = grp2idx(YTrainDs);

for j = 1:injs
    YTrainFs(j, grp2idx(YTrains)==(j)) = 1;
end

% one-hot injection
XYTrainF = vertcat(XYTrainF,YTrainFs);


inj = inje + injm + injs;

%% Test images
[tsImageDS, eTestLab, mTestLab, sTestLab, testDataSetFolders] = createBCFuzzTest(dataFolderTmpl, dataFolderSfx, @readFunctionGTS);

%
[lts,~] = size(tsImageDS.Files);

XTest = zeros([n,m,3,lts]);
XTestF = zeros([n*m*3,lts]);
XTestMean = zeros([lts,1]);
XTestStd = zeros([lts,1]);

for i=1:lts
    XTest(:,:,:,i) = double(readimage(tsImageDS,i));

    XTestF(:,i) = reshape(XTest(:,:,:,i), [n*m*3,1] );
    XTestMean(i) = mean(XTestF(:,i), 1);
    XTestStd(i) = std(XTestF(:,i), 0, 1);
end
XTestF = generic_mean_std_scale2D(XTestF, XTestMean, XTestStd);
YTeste = tsImageDS.Labels;


%% Emotion Injection one-hot in train terms
%[inje,~] = size(unique(YTeste));

YTestFe = zeros([inje,lts]);
YTestIe = zeros([lts,1]);

for j = 1:lts
    YTestIe(j) = strmatch(char(YTeste(j)), char(YTrainDe), 'exact');
end

for j = 1:inje
    YTestFe(j, YTestIe==(j)) = 1;
end

% one-hot injection
XYTestF = vertcat(XTestF,YTestFe);


% Makeup Injection one-hot
nmIdx = contains(mTestLab, 'NM');
mTestLab(nmIdx) = 'NM';
YTestm = categorical(mTestLab);
%[injm,~] = size(unique(YTestm));

YTestFm = zeros(injm,lts);
YTestIm = zeros([lts,1]);

for j = 1:lts
    YTestIm(j) = strmatch(char(YTestm(j)), char(YTrainDm), 'exact');
end

for j = 1:injm
    YTestFm(j, YTestIm==(j)) = 1;
end

% one-hot injection
XYTestF = vertcat(XYTestF,YTestFm);


% Subject Injection one-hot
YTests = categorical(sTestLab);
%[injs,~] = size(unique(YTests));

YTestFs = zeros(injs,lts);
YTestIs = zeros([lts,1]);

for j = 1:lts
    YTestIs(j) = strmatch(char(YTests(j)), char(YTrainDs), 'exact');
end

for j = 1:injs
    YTestFs(j, YTestIs==(j)) = 1;
end

% one-hot injection
XYTestF = vertcat(XYTestF,YTestFs);


%%
x_off=0;
x_in=n*m*3+inj;
t_in=1;

y_off=0;
%y_out=1;
y_out=n*m*3;
t_out=t_in;

ini_rate = 0.0002; 
max_epoch = 100;

modelName = 'bc_vis4x4ae';

modelFile = strcat(modelName, '.mat');


%%
reTrainFl = 1;

loadedFl = 0;
if isfile(modelFile)
    fprintf('Loading %s %d\n', modelFile, i);
    load(modelFile, 'regNet');
    loadedFl = 1;
else


        regNet = residualVis4x4x3BTransAEBaseNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch, 1/9, 1/18, 20, 20, 2);
        %regNet = visBTransAEBaseNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch, 1/3);

        %
        regNet.mb_size = 256; 

        regNet = Create(regNet);

end

%%
if (loadedFl == 0) || (reTrainFl == 1)

    fprintf('Training %s %d\n', modelFile, i);
%
%i=1;
        % GPU on
        gpuDevice(1);
        reset(gpuDevice(1));
    
        regNet = regNet.Train(1, XYTrainF, XTrainF);%YTrain3C);

        % GPU off
        delete(gcp('nocreate'));
        gpuDevice([]); 

    %
    fprintf('Saving %s %d\n', modelFile, i);
    save(modelFile, 'regNet');

% end no file - train
end

% end of contionous learning
%end

%% LrReLU weights
%histogram(regNet.lGraph.Layers(6,1).A) %28
%histogram(regNet.lGraph.Layers(8,1).A,'BinLimits',[0.45,1], Normalization="percentage") %43905
%ytickformat("percentage") 

%% activations
        % GPU on
%        gpuDevice(1);
%        reset(gpuDevice(1));
        
%act1 = activations(regNet.trainedNet, XYTestF(:,:)', 'b_k_hid1');
%ma = max(act1,[],'all');
%mi = min(act1,[],'all');
%actn = (act1 - mi)/(ma - mi);

        % GPU off
%        delete(gcp('nocreate'));
%        gpuDevice([]); 

%%
%i = 1 + floor(rand()*l);

%subplot(2,2,1);
%If = XTestF(:,i);
%I2 = reshape(If, [n, m]);

%image(I2 .* 255);

%subplot(2,2,2);
%Ifp = actn(:,i);
%I2p = reshape(Ifp, [n, m]);

%image(I2p .* 255);
%% test

% GPU on
gpuDevice(1);
reset(gpuDevice(1));

predictedScores = predict(regNet.trainedNet, XYTestF');
XTestF2 = predictedScores';

% GPU off
delete(gcp('nocreate'));
gpuDevice([]);

%[M, I]=max(XTestF2,[],1);
%acc = sum(I == (YTestD + 1))/lts

%% synthetic duplication

% GPU on
gpuDevice(1);
reset(gpuDevice(1));

predictedScores = predict(regNet.trainedNet, XYTrainF');
XTrainF2 = predictedScores';

% GPU off
delete(gcp('nocreate'));
gpuDevice([]);

%XTrainF23C = reshape(XTrainF2, [n, m, 3, l]);
%XTrain3C(:,:,:,l+1:2*l) = XTrainF23C;
%YTrain3C(:,l+1:2*l) = YTrain3C;



%% Generate

%% Emotion variants
batch_szE = floor(regNet.mb_size);%/inje);
    
XGenFSB = zeros([n*m*3,batch_szE*inje]);
XGenMeanSB = zeros([batch_szE*inje, 1]);
XGenStdSB = zeros([batch_szE*inje, 1]);
XYGenFB = zeros([n*m*3+inj,batch_szE*inje]);

for k=1:batch_szE

    % random train results
    i = 1 + floor(rand()*lts);
    le = inje;

    XGenF = XTestF(:,i);
    XGenMean = XTestMean(i);
    XGenStd = XTestStd(i);

    YGenFm = YTestFm(:,i);
    YGenFs = YTestFs(:,i);

    XGenFS = repmat(XGenF,1,le);
    XGenMeanS = repmat(XGenMean,le,1);
    XGenStdS = repmat(XGenStd,le,1);

    YGenFmS = repmat(YGenFm,1,le);
    YGenFsS = repmat(YGenFs,1,le);

    %XYGenFO = XYTrainF(:,i);

    % Emotion Injection one-hot
    YGenFeS = zeros(inje,le);

    for j = 1:inje
        YGenFeS(j, j) = 20;
    end

    % one-hot injection
    XYGenF = vertcat(XGenFS,YGenFeS);
    XYGenF = vertcat(XYGenF,YGenFmS);
    XYGenF = vertcat(XYGenF,YGenFsS);

    % batch building
    XGenFSB(:,1+(k-1)*inje:k*inje) = XGenFS;
    XGenMeanSB(1+(k-1)*inje:k*inje) = XGenMeanS;
    XGenStdSB(1+(k-1)*inje:k*inje) = XGenStdS;
    XYGenFB(:,1+(k-1)*inje:k*inje) = XYGenF;

end

%% Makeup variants
batch_szM = floor(regNet.mb_size);%/injm);
    
XGenFSB = zeros([n*m*3,batch_szM*injm]);
XGenMeanSB = zeros([batch_szM*injm, 1]);
XGenStdSB = zeros([batch_szM*injm, 1]);
XYGenFB = zeros([n*m*3+inj,batch_szM*injm]);

for k=1:batch_szM

    % random train results
    i = 1 + floor(rand()*lts);
    ls = injm;

    XGenF = XTestF(:,i);
    XGenMean = XTestMean(i);
    XGenStd = XTestStd(i);

    YGenFe = YTestFe(:,i);
    YGenFm = YTestFm(:,i);
    YGenFs = YTestFs(:,i);

    XGenFS = repmat(XGenF,1,ls);
    XGenMeanS = repmat(XGenMean,ls,1);
    XGenStdS = repmat(XGenStd,ls,1);

    YGenFeS = repmat(YGenFe,1,ls);
    %YGenFmS = repmat(YGenFm,1,ls);
    YGenFsS = repmat(YGenFs,1,ls);

    %XYGenFO = XYTrainF(:,i);

    % Subject Injection one-hot
    YGenFmS = zeros(injm,ls);

    for j = 1:injm
        YGenFmS(j, j) = 20;
    end

    % one-hot injection
    XYGenF = vertcat(XGenFS,YGenFeS);
    XYGenF = vertcat(XYGenF,YGenFmS);
    XYGenF = vertcat(XYGenF,YGenFsS);

    % batch building
    XGenFSB(:,1+(k-1)*injm:k*injm) = XGenFS;
    XGenMeanSB(1+(k-1)*injm:k*injm) = XGenMeanS;
    XGenStdSB(1+(k-1)*injm:k*injm) = XGenStdS;
    XYGenFB(:,1+(k-1)*injm:k*injm) = XYGenF;

end
%% Subject variants
batch_szS = floor(regNet.mb_size);%/injs);
    
XGenFSB = zeros([n*m*3,batch_szS*injs]);
XGenMeanSB = zeros([batch_szS*injs, 1]);
XGenStdSB = zeros([batch_szS*injs, 1]);
XYGenFB = zeros([n*m*3+inj,batch_szS*injs]);

for k=1:batch_szS

    % random train results
    i = 1 + floor(rand()*lts);
    ls = injs;

    XGenF = XTestF(:,i);
    XGenMean = XTestMean(i);
    XGenStd = XTestStd(i);

    YGenFe = YTestFe(:,i);
    YGenFm = YTestFm(:,i);
    YGenFs = YTestFs(:,i);

    XGenFS = repmat(XGenF,1,ls);
    XGenMeanS = repmat(XGenMean,ls,1);
    XGenStdS = repmat(XGenStd,ls,1);

    YGenFeS = repmat(YGenFe,1,ls);
    YGenFmS = repmat(YGenFm,1,ls);
    %YGenFsS = repmat(YGenFs,1,ls);

    %XYGenFO = XYTrainF(:,i);

    % Subject Injection one-hot
    YGenFsS = zeros(injs,ls);

    for j = 1:injs
        YGenFsS(j, j) = 20;
    end

    % one-hot injection
    XYGenF = vertcat(XGenFS,YGenFeS);
    XYGenF = vertcat(XYGenF,YGenFmS);
    XYGenF = vertcat(XYGenF,YGenFsS);

    % batch building
    XGenFSB(:,1+(k-1)*injs:k*injs) = XGenFS;
    XGenMeanSB(1+(k-1)*injs:k*injs) = XGenMeanS;
    XGenStdSB(1+(k-1)*injs:k*injs) = XGenStdS;
    XYGenFB(:,1+(k-1)*injs:k*injs) = XYGenF;

end

%%
% GPU on
gpuDevice(1);
reset(gpuDevice(1));

%for j = 1:inje
    predictedScores = predict(regNet.trainedNet, XYGenFB');
    XGenFB2 = predictedScores';
%end

% GPU off
delete(gcp('nocreate'));
gpuDevice([]);

% Rescale train results
XGenFR = generic_mean_std_rescale2D(XGenFSB, XGenMeanSB, XGenStdSB);
XGenFR2 = generic_mean_std_rescale2D(XGenFB2, XGenMeanSB, XGenStdSB);


%% Display Emotion Gen
k = 1 + floor(rand()*batch_szE);

subplot(3,3,1);
If = int8(XGenFR(:,1+(k-1)*inje) * .5 );
I2 = reshape(If, [n, m, 3]);

image(I2);
%title(string(YTest(i)));
%title(strcat(string(label_names(YTest(i)+1)) ));

for j=1:inje

    subplot(3,3,1+j);
    Ift = int8(XGenFR2(:,j+(k-1)*inje) * .5);
    I2t = reshape(Ift, [n, m, 3]);

    image(I2t);
%title(string(I(i)));

end

%% Display Makeup Gen
k = 1 + floor(rand()*batch_szM);

subplot(8,8,1);
If = int8(XGenFR(:,1+(k-1)*injm) * .5 );
I2 = reshape(If, [n, m, 3]);

image(I2);
%title(string(YTest(i)));
%title(strcat(string(label_names(YTest(i)+1)) ));

for j=1:injm

    subplot(8,8,1+j);
    Ift = int8(XGenFR2(:,j+(k-1)*injm) * .5);
    I2t = reshape(Ift, [n, m, 3]);

    image(I2t);
%title(string(I(i)));

end

%% Display Subject Gen
k = 1 + floor(rand()*batch_szS);

subplot(5,5,1);
If = int8(XGenFR(:,1+(k-1)*injs) * .5 );
I2 = reshape(If, [n, m, 3]);

image(I2);
%title(string(YTest(i)));
%title(strcat(string(label_names(YTest(i)+1)) ));

for j=1:injs

    subplot(5,5,1+j);
    Ift = int8(XGenFR2(:,j+(k-1)*injs) * .5);
    I2t = reshape(Ift, [n, m, 3]);

    image(I2t);
%title(string(I(i)));

end






%% Rescale train results
XTrainFR = generic_mean_std_rescale2D(XTrainF, XTrainMean, XTrainStd);
XTrainFR2 = generic_mean_std_rescale2D(XTrainF2, XTrainMean, XTrainStd);
%% random train results
i = 1 + floor(rand()*l);


subplot(2,2,1);
If = int8(XTrainFR(:,i) * .5);
I2 = reshape(If, [n, m, 3]);

image(I2);
%title(string(YTest(i)));
%title(strcat(string(label_names(YTest(i)+1)) ));

subplot(2,2,2);
Ift = int8(XTrainFR2(:,i) * .5);
I2t = reshape(Ift, [n, m, 3]);

image(I2t);
%title(string(I(i)));


%% Rescale test results
XTestFR = generic_mean_std_rescale2D(XTestF, XTestMean, XTestStd);
XTestFR2 = generic_mean_std_rescale2D(XTestF2, XTestMean, XTestStd);
%% random test results
i = 1 + floor(rand()*lts);


subplot(2,2,1);
If = int8(XTestFR(:,i) * .5);
I2 = reshape(If, [n, m, 3]);

image(I2);
%image(I2 .* 255);
%title(strcat(string(i),' c:',string(YTest(i))));
%title(strcat(string(label_names(YTest(i)+1)) ));

subplot(2,2,2);
Ift = int8(XTestFR2(:,i) * .5);
I2t = reshape(Ift, [n, m, 3]);

image(I2t);
%title(string(I(i)));


%% difficult results
%idx = [6149 9386 6131 3518 8163 12187 1991 1157 2163 9532 4300 6172];
%[~,ni] = size(idx);

%cidx = [12 13 5 38 33 11 7 17 9 16 26 14];

%colormap(gray)

%for i = 1:ni
%    subplot(4,10,i);
%    If = XTestF(1:n*m,idx(i));
%    I2 = reshape(If, [n, m]);

%    image(I2 .* 255);   
%end

%for i = 1:ni
%    subplot(4,10,ni+i);
%    Ift = XTestF2(1:n*m,idx(i));
%    I2t = reshape(Ift, [n, m]);

%    image(I2t .* 255);
%end

%% mutated results
idx = [6149 9386 6131 3518 8163 12187 1991 1157 2163 9532 4300 6172];
[~,ni] = size(idx);

cidx = [12 13 5 38 33 11 7 17 9 16 26 14];


XTestM = XTestF(:,idx);
XTestMM = repmat(XTestM', ni, 1)';

YTestM = zeros([inj, ni*ni]);


for i = 1:ni %mutatuon
    for j = 1:ni %seed
        YTestM(cidx(i),(i-1)*ni+j) = 1;
    end
end

XYTestM = vertcat(XTestMM,YTestM);

% GPU on
gpuDevice(1);
reset(gpuDevice(1));

predictedScores = predict(regNet.trainedNet, XYTestM');
XTestM2 = predictedScores';

% GPU off
delete(gcp('nocreate'));
gpuDevice([]);

%% mutated display
%colormap(gray)
If = int8(XTestF(:,i).* 255);
I2 = reshape(If, [n, m, 3]);

for i = 0:ni %mutation
    for j = 1:ni %seed
        subplot(ni+1,ni,i*ni+j);

        if i==0
            Im = int8(XTestF(:,idx(j)) .* 255);
        else
            Im = int8(XTestM2(:,(i-1)*ni+j) .* 255);
        end

        I2m = reshape(Im, [n, m, 3]);

        image(I2m);

        if i == 0
            title(strcat('c:', string(cidx(j)) ));
        %elseif i == 11
        %    title(strcat('m:-'));
        else
            title(strcat('m:', string(cidx(i))));
        end

    end
end


%% SOTA comparison

%% Load Pre-trained Network (AlexNet)
% AlexNet is a pre-trained network trained on 1000 object categories. 
alex = alexnet('Weights','none');
layers = alex;

% Review Network Architecture 
%layers = alex.Layer;

% Modify Pre-trained Network 
% AlexNet was trained to recognize 1000 classes, we need to modify it to
% recognize just nClasses classes. 
n_ll = 25;
n_sml = n_ll - 2;
layers(1) = imageInputLayer([32, 32, 3]);
layers(9) = maxPooling2dLayer(2, 'Name', 'pool2');
layers(16) = maxPooling2dLayer(1, 'Name', 'pool5');
layers(n_sml) = fullyConnectedLayer(inj); % change this based on # of classes
layers(n_ll) = classificationLayer;

% Perform Transfer Learning
% For transfer learning we want to change the weights of the network ever so slightly. How
% much a network is changed during training is controlled by the learning
% rates. 

max_epoch = 200;
mb_size = 512;
opts = trainingOptions('adam', ...
                'ExecutionEnvironment','auto',...
                'Shuffle', 'every-epoch',...
                'MiniBatchSize', mb_size, ...
                'InitialLearnRate', ini_rate, ...
                'MaxEpochs', max_epoch);

                        
                      %'Plots', 'training-progress',...

%% Train the Network  

        % GPU on
        gpuDevice(1);
        reset(gpuDevice(1));

myNet = trainNetwork(XTrain3C, YTrain3C, layers, opts);

        % GPU off
        delete(gcp('nocreate'));
        gpuDevice([]); 

%% test

% GPU on
gpuDevice(1);
reset(gpuDevice(1));

predictedScores = predict(myNet, XTest3C);
X2Test3C = predictedScores';

% GPU off
delete(gcp('nocreate'));
gpuDevice([]);

%%
[M, I]=max(X2Test3C,[],1);
acc = sum(I == (YTestD + 1))/lts       