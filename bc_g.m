%% Clear everything 
clearvars -global;
clear all; close all; clc;

addpath('~/ANNLib/');
addpath('~/BookClub/BC2EClGen/');

saveFolder = '~/';


%% Dataset root folder template and suffix
%trDataFolder = '~/data/GTS/GTSRB/Final_Training/Images';

dataFolderTmpl = '~/data/BC2EF_Sfx';
dataFolderSfx = '1072x712';

%%[trImageDS, eTrainLab, mTrainLab, sTrainLab, trainDataSetFolders] = createBCFuzzTrain(dataFolderTmpl, dataFolderSfx, @readFunctionGTS);

%nFold = 1;
%[trImageDS, eTrainLab, mTrainLab, sTrainLab, trainDataSetFolders] = createBCCleanTrain(dataFolderTmpl, dataFolderSfx, @readFunction100x100); %@readFunctionGTS);
%nFold = 2;
%[trImageDS, eTrainLab, mTrainLab, sTrainLab, trainDataSetFolders] = createBCCleanTrain2(dataFolderTmpl, dataFolderSfx, @readFunction100x100); %@readFunctionGTS);
nFold = 3;
[trImageDS, eTrainLab, mTrainLab, sTrainLab, trainDataSetFolders] = createBCCleanTrain3(dataFolderTmpl, dataFolderSfx, @readFunction100x100); %@readFunctionGTS);
%nFold = 4;
%[trImageDS, eTrainLab, mTrainLab, sTrainLab, trainDataSetFolders] = createBCCleanTrain4(dataFolderTmpl, dataFolderSfx, @readFunction100x100); %@readFunctionGTS);
%nFold = 5;
%[trImageDS, eTrainLab, mTrainLab, sTrainLab, trainDataSetFolders] = createBCCleanTrain5(dataFolderTmpl, dataFolderSfx, @readFunction100x100); %@readFunctionGTS);

% Init
inje = 0;
injs = 0;
injm = 0;
eFl = 1;
sFl = 0;
mFl = 0;

n = 100; %80;
m = 100; %80;
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


XYTrainF = XTrainF;
%% Emotion Injection one-hot
if eFl
    [inje,~] = size(unique(YTraine));

    YTrainFe = zeros(inje,l);

    YTrainDe = unique(YTraine);
    YTrainIe = grp2idx(YTrainDe);

    for j = 1:inje
        YTrainFe(j, grp2idx(YTraine)==(j)) = 1;
    end

    % one-hot injection
    XYTrainF = vertcat(XYTrainF,YTrainFe);
end

%% Makeup Injection one-hot
%nmIdx = contains(mTrainLab, 'NM');
%mTrainLab(nmIdx) = 'NM';
%YTrainm = categorical(mTrainLab);
%[injm,~] = size(unique(YTrainm));

%YTrainFm = zeros(injm,l);

%YTrainDm = unique(YTrainm);
%YTrainIm = grp2idx(YTrainDm);

%for j = 1:injm
%    YTrainFm(j, grp2idx(YTrainm)==(j)) = 1;
%end

% one-hot injection
%XYTrainF = vertcat(XYTrainF,YTrainFm);


%% Subject Injection one-hot
if sFl
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
end

%%
inj = inje + injm + injs;

%% Test images
if nFold == 1
    [tsImageDS, eTestLab, mTestLab, sTestLab, testDataSetFolders] = createBCFuzzTest(dataFolderTmpl, dataFolderSfx, @readFunction100x100); %@readFunctionGTS);
elseif nFold == 2
    [tsImageDS, eTestLab, mTestLab, sTestLab, testDataSetFolders] = createBCFuzzTest2(dataFolderTmpl, dataFolderSfx, @readFunction100x100); %@readFunctionGTS);
elseif nFold == 3
    [tsImageDS, eTestLab, mTestLab, sTestLab, testDataSetFolders] = createBCFuzzTest3(dataFolderTmpl, dataFolderSfx, @readFunction100x100); %@readFunctionGTS);
elseif nFold == 4
    [tsImageDS, eTestLab, mTestLab, sTestLab, testDataSetFolders] = createBCFuzzTest4(dataFolderTmpl, dataFolderSfx, @readFunction100x100); %@readFunctionGTS);
else
    [tsImageDS, eTestLab, mTestLab, sTestLab, testDataSetFolders] = createBCFuzzTest5(dataFolderTmpl, dataFolderSfx, @readFunction100x100); %@readFunctionGTS);
end


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


XYTestF = XTestF;
%% Emotion Injection one-hot in test terms
if eFl
    YTestFe = zeros([inje,lts]);
    YTestIe = zeros([lts,1]);

    for j = 1:lts
        YTestIe(j) = strmatch(char(YTeste(j)), char(YTrainDe), 'exact');
    end

    for j = 1:inje
        YTestFe(j, YTestIe==(j)) = 1;
    end

    % one-hot injection
    XYTestF = vertcat(XYTestF,YTestFe);
end

%% Makeup Injection one-hot
%nmIdx = contains(mTestLab, 'NM');
%mTestLab(nmIdx) = 'NM';
%YTestm = categorical(mTestLab);

%YTestFm = zeros(injm,lts);
%YTestIm = zeros([lts,1]);

%for j = 1:lts
%    YTestIm(j) = strmatch(char(YTestm(j)), char(YTrainDm), 'exact');
%end

%for j = 1:injm
%    YTestFm(j, YTestIm==(j)) = 1;
%end

% one-hot injection
%XYTestF = vertcat(XYTestF,YTestFm);


%% Subject Injection one-hot
if sFl
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
end


%%
x_off=0;
x_in=n*m*3+inj;
t_in=1;

y_off=0;
%y_out=1;
y_out=n*m*3;
t_out=t_in;

ini_rate = 0.0002; 
max_epoch = 500; %2000; <15 rmse

%modelName = 'bc_rvis5x5ae';
modelName = 'bc_rvis5x5mixae';

modelFile = strcat(saveFolder, modelName, '.', string(nFold), '.mat');


%%
reTrainFl = 1; %1;

loadedFl = 0;
if isfile(modelFile)
    fprintf('Loading %s %d\n', modelFile, i);
    load(modelFile, 'regNet');
    loadedFl = 1;
else

        regNet = residualVis5x5x3BTransAEBaseNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch, 1/3, 1/3, 1/3, 1/3, 20, 20, inj, 1/20); %1/20 for mix model, or 1/30
        %regNet = residualVis4x4x3BTransAEBaseNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch, 1/3, 1/3, 1/3, 1/3, 20, 20, inj, 1/10);
        %regNet = vis4x4x3BTransAEBaseNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch, 1/9, 1/18, 20, 20, inj);


        %
        regNet.mb_size = 64; %256; 

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

%% Not for augmentation check
AugmentFl = 1;
if ~AugmentFl

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

% Rescale test results
XTestFR = generic_mean_std_rescale2D(XTestF, XTestMean, XTestStd);
XTestFR2 = generic_mean_std_rescale2D(XTestF2, XTestMean, XTestStd);
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



% Rescale train results
XTrainFR = generic_mean_std_rescale2D(XTrainF, XTrainMean, XTrainStd);
XTrainFR2 = generic_mean_std_rescale2D(XTrainF2, XTrainMean, XTrainStd);

end
%% Not for augmentation check
AugmentFl = 1;
if ~AugmentFl

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

end



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

    %YGenFm = YTestFm(:,i);
    %YGenFs = YTestFs(:,i);

    XGenFS = repmat(XGenF,1,le);
    XGenMeanS = repmat(XGenMean,le,1);
    XGenStdS = repmat(XGenStd,le,1);

    %YGenFmS = repmat(YGenFm,1,le);
    %YGenFsS = repmat(YGenFs,1,le);

    %%XYGenFO = XYTrainF(:,i);

    % Emotion Injection one-hot
    YGenFeS = zeros(inje,le);

    for j = 1:inje
        YGenFeS(j, j) = 1; %20;
    end

    % one-hot injection
    XYGenF = vertcat(XGenFS,YGenFeS);
    %XYGenF = vertcat(XYGenF,YGenFmS);
    %XYGenF = vertcat(XYGenF,YGenFsS);

    % batch building
    XGenFSB(:,1+(k-1)*inje:k*inje) = XGenFS;
    XGenMeanSB(1+(k-1)*inje:k*inje) = XGenMeanS;
    XGenStdSB(1+(k-1)*inje:k*inje) = XGenStdS;
    XYGenFB(:,1+(k-1)*inje:k*inje) = XYGenF;

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



%% Not for augmentation check
if ~AugmentFl
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
        YGenFmS(j, j) = 1; %20;
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

    %YGenFe = YTestFe(:,i);
    %YGenFm = YTestFm(:,i);
    YGenFs = YTestFs(:,i);

    XGenFS = repmat(XGenF,1,ls);
    XGenMeanS = repmat(XGenMean,ls,1);
    XGenStdS = repmat(XGenStd,ls,1);

    %YGenFeS = repmat(YGenFe,1,ls);
    %YGenFmS = repmat(YGenFm,1,ls);
    %%YGenFsS = repmat(YGenFs,1,ls);

    %%XYGenFO = XYTrainF(:,i);

    % Subject Injection one-hot
    YGenFsS = zeros(injs,ls);

    for j = 1:injs
        YGenFsS(j, j) = 1; %20;
    end

    XYGenF = XGenFS;
    % one-hot injection
    %XYGenF = vertcat(XGenF,YGenFeS);
    %XYGenF = vertcat(XYGenF,YGenFmS);
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


end










%% Classification model
%reTrainFl = 1;
%loadedFl = 0;

%modelName = 'bc_rvis5x5mixcl';
%modelFile = strcat(saveFolder, modelName, '.mat');

%x_in=n*m*3;
%y_out=inje;

%regNet = residualVis5x5x3BTransAEBaseNet2CL(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch, 1/3, 1/3, 1/3, 1/3, 20, 20, inj, 1/20);
        
%regNet.mb_size = 64; 
%regNet = Create(regNet);
%%
%if (loadedFl == 0) || (reTrainFl == 1)

%    fprintf('Training %s %d\n', modelFile, i);
%
%i=1;
        % GPU on
%        gpuDevice(1);
%        reset(gpuDevice(1));
    
%        regNet = regNet.Train(1, XTrainF, YTraine);

        % GPU off
%        delete(gcp('nocreate'));
%        gpuDevice([]); 

    %
%    fprintf('Saving %s %d\n', modelFile, i);
%    save(modelFile, 'regNet');

% end no file - train
%end








%% SOTA comparison

%% Generate All Train Emotion variants
%batch_szE = floor(regNet.mb_size);%/inje);
lgen = l*inje;

XGenFSB = zeros([n*m*3,l*inje]);
XGenMeanSB = zeros([l*inje, 1]);
XGenStdSB = zeros([l*inje, 1]);
XYGenFB = zeros([n*m*3+inj,l*inje]);
YGenFSt = string([l*inje,1]);


for k=1:l %batch_szE

    % random train results
    i = k; %1 + floor(rand()*lts);
    le = inje;

    XGenF = XTrainF(:,i);
    XGenMean = XTrainMean(i);
    XGenStd = XTrainStd(i);


    XGenFS = repmat(XGenF,1,le);
    XGenMeanS = repmat(XGenMean,le,1);
    XGenStdS = repmat(XGenStd,le,1);


    % Emotion Injection one-hot and categorical
    YGenFeS = zeros(inje,le);
    YGenFeSt = strings([inje,1]);

    for j = 1:inje
        YGenFeS(j, j) = 1; %20;
        YGenFeSt(j) = char(YTrainDe(j));
    end

    % one-hot injection
    XYGenF = vertcat(XGenFS,YGenFeS);

    % batch building
    XGenFSB(:,1+(k-1)*inje:k*inje) = XGenFS;
    XGenMeanSB(1+(k-1)*inje:k*inje) = XGenMeanS;
    XGenStdSB(1+(k-1)*inje:k*inje) = XGenStdS;
    XYGenFB(:,1+(k-1)*inje:k*inje) = XYGenF;

    YGenFSt(1+(k-1)*inje:k*inje) = YGenFeSt;
end
    YGenFC = categorical(YGenFSt');

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
k = 1 + floor(rand()*l);

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




%% Generate All Test Emotion variants
%%batch_szE = floor(regNet.mb_size);%/inje);
    
%XGenFSB = zeros([n*m*3,lts*inje]);
%XGenMeanSB = zeros([lts*inje, 1]);
%XGenStdSB = zeros([lts*inje, 1]);
%XYGenFB = zeros([n*m*3+inj,lts*inje]);

%for k=1:lts %batch_szE

    % random train results
%    i = k; %1 + floor(rand()*lts);
%    le = inje;

%    XGenF = XTestF(:,i);
%    XGenMean = XTestMean(i);
%    XGenStd = XTestStd(i);


%    XGenFS = repmat(XGenF,1,le);
%    XGenMeanS = repmat(XGenMean,le,1);
%    XGenStdS = repmat(XGenStd,le,1);


    % Emotion Injection one-hot
%    YGenFeS = zeros(inje,le);

%    for j = 1:inje
%        YGenFeS(j, j) = 1; %20;
%    end

    % one-hot injection
%    XYGenF = vertcat(XGenFS,YGenFeS);

    % batch building
%    XGenFSB(:,1+(k-1)*inje:k*inje) = XGenFS;
%    XGenMeanSB(1+(k-1)*inje:k*inje) = XGenMeanS;
%    XGenStdSB(1+(k-1)*inje:k*inje) = XGenStdS;
%    XYGenFB(:,1+(k-1)*inje:k*inje) = XYGenF;

%end


%% Generate All Test Emotion variants from neutral

%ITestN = YTeste == 'NE';
%lne = sum(ITestN);
%lgen = lne*inje;
    
%XGenFSB = zeros([n*m*3,lne*inje]);
%XGenMeanSB = zeros([lne*inje, 1]);
%XGenStdSB = zeros([lne*inje, 1]);
%XYGenFB = zeros([n*m*3+inj,lne*inje]);
%YGenFSt = string([lne*inje,1]);

%i = 0;
%for k=1:lts

%    if ITestN(k)

    % random train results
%    i = i + 1;
%    le = inje;

%    XGenF = XTestF(:,k);
%    XGenMean = XTestMean(k);
%    XGenStd = XTestStd(k);


%    XGenFS = repmat(XGenF,1,le);
%    XGenMeanS = repmat(XGenMean,le,1);
%    XGenStdS = repmat(XGenStd,le,1);


    % Emotion Injection one-hot
%    YGenFeS = zeros(inje,le);
%    YGenFeSt = strings([inje,1]);

%    for j = 1:inje
%        YGenFeS(j, j) = 1; %20;
%        YGenFeSt(j) = char(YTrainDe(j));
%    end

    % one-hot injection
%    XYGenF = vertcat(XGenFS,YGenFeS);

    % batch building
%    XGenFSB(:,1+(i-1)*inje:i*inje) = XGenFS;
%    XGenMeanSB(1+(i-1)*inje:i*inje) = XGenMeanS;
%    XGenStdSB(1+(i-1)*inje:i*inje) = XGenStdS;
%    XYGenFB(:,1+(i-1)*inje:i*inje) = XYGenF;

%    YGenFSt(1+(i-1)*inje:i*inje) = YGenFeSt;

%    end
%end

%YGenFC = categorical(YGenFSt');

%%
% GPU on
%gpuDevice(1);
%reset(gpuDevice(1));

%%for j = 1:inje
%    predictedScores = predict(regNet.trainedNet, XYGenFB');
%    XGenFB2 = predictedScores';
%%end

% GPU off
%delete(gcp('nocreate'));
%gpuDevice([]);

% Rescale train results
%XGenFR = generic_mean_std_rescale2D(XGenFSB, XGenMeanSB, XGenStdSB);
%XGenFR2 = generic_mean_std_rescale2D(XGenFB2, XGenMeanSB, XGenStdSB);


%% Display Emotion Gen
%k = 1 + floor(rand()*lne);

%subplot(3,3,1);
%If = int8(XGenFR(:,1+(k-1)*inje) * .5 );
%I2 = reshape(If, [n, m, 3]);

%image(I2);
%%title(string(YTest(i)));
%%title(strcat(string(label_names(YTest(i)+1)) ));

%for j=1:inje

%    subplot(3,3,1+j);
%    Ift = int8(XGenFR2(:,j+(k-1)*inje) * .5);
%    I2t = reshape(Ift, [n, m, 3]);

%    image(I2t);
%%title(string(I(i)));

%end






%% Load Pre-trained Network (AlexNet)
% AlexNet is a pre-trained network trained on 1000 object categories. 
%alex = alexnet('Weights','none');
%layers = alex;

% Review Network Architecture 
%%layers = alex.Layer;

% Modify Pre-trained Network 
% AlexNet was trained to recognize 1000 classes, we need to modify it to
% recognize just nClasses classes. 
%n_ll = 25;
%n_sml = n_ll - 2;
%layers(1) = imageInputLayer([100, 100, 3]);
%%layers(9) = maxPooling2dLayer(2, 'Name', 'pool2');
%%layers(16) = maxPooling2dLayer(1, 'Name', 'pool5');
%layers(n_sml) = fullyConnectedLayer(inje); % change this based on # of classes
%layers(n_ll) = classificationLayer;

%lgraph = layerGraph(layers);

% Perform Transfer Learning
% For transfer learning we want to change the weights of the network ever so slightly. How
% much a network is changed during training is controlled by the learning
% rates. 

%max_epoch = 100;
%mb_size = 64;
%opts = trainingOptions('adam', ...
%                'ExecutionEnvironment','auto',...
%                'Shuffle', 'every-epoch',...
%                'MiniBatchSize', mb_size, ...
%                'InitialLearnRate', ini_rate, ...
%                'MaxEpochs', max_epoch);

                        
                      %'Plots', 'training-progress',...
%% Load Pre-trained Network (Inceptions)
         % Load Pre-trained Network 
        incept3 = inceptionv3;

        max_epoch = 50;
        mb_size = 64;

        % Review Network Architecture 
        lgraph_r = layerGraph(incept3);

        % Modify Pre-trained Network 
        lgraph = replaceLayer(lgraph_r, 'input_1', imageInputLayer([100, 100, 3], 'Name', 'input_1'));
        lgraph = replaceLayer(lgraph, 'predictions', fullyConnectedLayer(inje, 'Name', 'predictions'));
        lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', classificationLayer('Name', 'ClassificationLayer_predictions'));

        % Perform Transfer Learning
        % For transfer learning we want to change the weights of the network 
        % ever so slightly. How much a network is changed during training is 
        % controlled by the learning rates. 
        opts = trainingOptions('adam',...
                            'ExecutionEnvironment','auto',...
                'Shuffle', 'every-epoch',...
                'MiniBatchSize', mb_size, ...
                'InitialLearnRate', ini_rate, ...
                'MaxEpochs', max_epoch);

                       %'ExecutionEnvironment','parallel',...
                       %'InitialLearnRate', 0.001,...
                       %'LearnRateSchedule', 'piecewise',...
                       %'LearnRateDropPeriod', 5,...
                       %'LearnRateDropFactor', 0.9,...
                       %'MiniBatchSize', mb_size,...
                       %'MaxEpochs', 10);


%% Original train data set
XTrain3C = zeros([n, m, 3, l]);
XTrain3C(:, :, 1, :) = reshape(XTrainF(1:n*m,:),[n,m,l]);
XTrain3C(:, :, 2, :) = reshape(XTrainF(1+n*m:2*n*m,:),[n,m,l]);
XTrain3C(:, :, 3, :) = reshape(XTrainF(1+2*n*m:3*n*m,:),[n,m,l]);
XTrain3Ct = XTrain3C;

YTrain3C = YTraine;
YTrain3Ct = YTrain3C;

lv = l;
%% Synthetic train data set
XTrain3C = zeros([n, m, 3, lgen]);
XTrain3C(:, :, 1, :) = reshape(XGenFB2(1:n*m,:),[n,m,lgen]);
XTrain3C(:, :, 2, :) = reshape(XGenFB2(1+n*m:2*n*m,:),[n,m,lgen]);
XTrain3C(:, :, 3, :) = reshape(XGenFB2(1+2*n*m:3*n*m,:),[n,m,lgen]);

YTrain3C = YGenFC;

lv = lgen;
%% Combined train data set
XTrain3C = cat(4, XTrain3Ct, XTrain3C);

YTrain3C = vertcat(YTrain3Ct, YTrain3C);

lv = l+lgen;

%% Train the Network  

        % GPU on
        gpuDevice(1);
        reset(gpuDevice(1));

%myNet = trainNetwork(XTrain3C, YTrain3C, layers, opts);
myNet = trainNetwork(XTrain3C, YTrain3C, lgraph, opts);

        % GPU off
        delete(gcp('nocreate'));
        gpuDevice([]); 



%% test

%% Original test data set
XTest3C = zeros([n, m, 3, lts]);
XTest3C(:, :, 1, :) = reshape(XTestF(1:n*m,:),[n,m,lts]);
XTest3C(:, :, 2, :) = reshape(XTestF(1+n*m:2*n*m,:),[n,m,lts]);
XTest3C(:, :, 3, :) = reshape(XTestF(1+2*n*m:3*n*m,:),[n,m,lts]);

% GPU on
gpuDevice(1);
reset(gpuDevice(1));

predictedScores = classify(myNet, XTest3C);
Y2Test3C = predictedScores;

% GPU off
delete(gcp('nocreate'));
gpuDevice([]);

%
acc = sum(YTeste == Y2Test3C)/lts      


% 1 0.3348 0.4141 0.4405
% 2 0.3860 0.4561 0.4649
% 3 0.3755 0.4192 0.3799 (0.3100 0.3974 0.3493)
% 4 0.3466 0.4701 0.4382
% 5 0.3581 0.4367 0.4410


%% train verification
% GPU on
%gpuDevice(1);
%reset(gpuDevice(1));

%predictedScores = classify(myNet, XTrain3C);
%Y2Train3C = predictedScores;

% GPU off
%delete(gcp('nocreate'));
%gpuDevice([]);

%
%acc = sum(YTrain3C == Y2Train3C)/lv