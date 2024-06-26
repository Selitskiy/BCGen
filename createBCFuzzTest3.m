function [mkImages, eLabels, mLabels, sLabels, mkDataSetFolders] = createBCFuzzTest3(dataFolderTmpl, dataFolderSfx, readFcn)

%% Create a real folder
dataFolder = strrep(dataFolderTmpl, 'Sfx', dataFolderSfx);


%% Create vectors of the makeup folder templates and category labels

% Empty vectors
mkDataSetFolders = strings(0);
mkLabels = strings(0);

% Let's populate vectors one by one, making labels from the top directory

mkDataSetFolders = [mkDataSetFolders, 'S1/S1NM2/S1NM2AN'];
%mkDataSetFolders = [mkDataSetFolders, 'S1/S1NM1/S1NM1AN'];
[~, n] = size(mkDataSetFolders);
[tmpStr, ~] = strsplit(mkDataSetFolders(n), '/');
mkLabelCur = tmpStr(1,3);
mkLabelCurN = strlength(mkLabelCur);
mkLabelCur = extractBetween(mkLabelCur, mkLabelCurN-1, mkLabelCurN);
mkLabels = [mkLabels, mkLabelCur];
%mkDataSetFolders = [mkDataSetFolders, 'S2/S2NM2/S2NM2AN'];
mkDataSetFolders = [mkDataSetFolders, 'S2/S2NM1/S2NM1AN'];

mkDataSetFolders = [mkDataSetFolders, 'S3/S3NM3/S3NM3AN'];
mkDataSetFolders = [mkDataSetFolders, 'S4/S4NM2/S4NM2AN'];
mkDataSetFolders = [mkDataSetFolders, 'S5/S5NM4/S5NM4AN'];
mkDataSetFolders = [mkDataSetFolders, 'S7/S7NM4/S7NM4AN'];
mkDataSetFolders = [mkDataSetFolders, 'S10/S10NM2/S10NM2AN'];
mkDataSetFolders = [mkDataSetFolders, 'S11/S11NM3/S11NM3AN'];
mkDataSetFolders = [mkDataSetFolders, 'S12/S12NM2/S12NM2AN'];
mkDataSetFolders = [mkDataSetFolders, 'S14/S14NM2/S14NM2AN'];

%
mkDataSetFolders = [mkDataSetFolders, 'S1/S1NM2/S1NM2CE'];
%mkDataSetFolders = [mkDataSetFolders, 'S1/S1NM1/S1NM1CE'];
[~, n] = size(mkDataSetFolders);
[tmpStr, ~] = strsplit(mkDataSetFolders(n), '/');
mkLabelCur = tmpStr(1,3);
mkLabelCurN = strlength(mkLabelCur);
mkLabelCur = extractBetween(mkLabelCur, mkLabelCurN-1, mkLabelCurN);
mkLabels = [mkLabels, mkLabelCur];
%mkDataSetFolders = [mkDataSetFolders, 'S2/S2NM2/S2NM2CE'];
mkDataSetFolders = [mkDataSetFolders, 'S2/S2NM1/S2NM1CE'];

mkDataSetFolders = [mkDataSetFolders, 'S3/S3NM3/S3NM3CE'];
mkDataSetFolders = [mkDataSetFolders, 'S4/S4NM2/S4NM2CE'];
mkDataSetFolders = [mkDataSetFolders, 'S5/S5NM4/S5NM4CE'];
mkDataSetFolders = [mkDataSetFolders, 'S7/S7NM4/S7NM4CE'];
mkDataSetFolders = [mkDataSetFolders, 'S10/S10NM2/S10NM2CE'];
mkDataSetFolders = [mkDataSetFolders, 'S11/S11NM3/S11NM3CE'];
mkDataSetFolders = [mkDataSetFolders, 'S12/S12NM2/S12NM2CE'];
mkDataSetFolders = [mkDataSetFolders, 'S14/S14NM2/S14NM2CE'];

%
mkDataSetFolders = [mkDataSetFolders, 'S1/S1NM2/S1NM2DS'];
%mkDataSetFolders = [mkDataSetFolders, 'S1/S1NM1/S1NM1DS'];
[~, n] = size(mkDataSetFolders);
[tmpStr, ~] = strsplit(mkDataSetFolders(n), '/');
mkLabelCur = tmpStr(1,3);
mkLabelCurN = strlength(mkLabelCur);
mkLabelCur = extractBetween(mkLabelCur, mkLabelCurN-1, mkLabelCurN);
mkLabels = [mkLabels, mkLabelCur];
%mkDataSetFolders = [mkDataSetFolders, 'S2/S2NM2/S2NM2DS'];
mkDataSetFolders = [mkDataSetFolders, 'S2/S2NM1/S2NM1DS'];

mkDataSetFolders = [mkDataSetFolders, 'S3/S3NM3/S3NM3DS'];
mkDataSetFolders = [mkDataSetFolders, 'S4/S4NM2/S4NM2DS'];
mkDataSetFolders = [mkDataSetFolders, 'S5/S5NM4/S5NM4DS'];
mkDataSetFolders = [mkDataSetFolders, 'S7/S7NM4/S7NM4DS'];
mkDataSetFolders = [mkDataSetFolders, 'S10/S10NM2/S10NM2DS'];
mkDataSetFolders = [mkDataSetFolders, 'S11/S11NM3/S11NM3DS'];
mkDataSetFolders = [mkDataSetFolders, 'S12/S12NM2/S12NM2DS'];
mkDataSetFolders = [mkDataSetFolders, 'S14/S14NM2/S14NM2DS'];

%
mkDataSetFolders = [mkDataSetFolders, 'S1/S1NM2/S1NM2HP'];
%mkDataSetFolders = [mkDataSetFolders, 'S1/S1NM1/S1NM1HP'];
[~, n] = size(mkDataSetFolders);
[tmpStr, ~] = strsplit(mkDataSetFolders(n), '/');
mkLabelCur = tmpStr(1,3);
mkLabelCurN = strlength(mkLabelCur);
mkLabelCur = extractBetween(mkLabelCur, mkLabelCurN-1, mkLabelCurN);
mkLabels = [mkLabels, mkLabelCur];
%mkDataSetFolders = [mkDataSetFolders, 'S2/S2NM2/S2NM2HP'];
mkDataSetFolders = [mkDataSetFolders, 'S2/S2NM1/S2NM1HP'];

mkDataSetFolders = [mkDataSetFolders, 'S3/S3NM3/S3NM3HP'];
mkDataSetFolders = [mkDataSetFolders, 'S4/S4NM2/S4NM2HP'];
mkDataSetFolders = [mkDataSetFolders, 'S5/S5NM4/S5NM4HP'];
mkDataSetFolders = [mkDataSetFolders, 'S7/S7NM4/S7NM4HP'];
mkDataSetFolders = [mkDataSetFolders, 'S10/S10NM2/S10NM2HP'];
mkDataSetFolders = [mkDataSetFolders, 'S11/S11NM3/S11NM3HP'];
mkDataSetFolders = [mkDataSetFolders, 'S12/S12NM2/S12NM2HP'];
mkDataSetFolders = [mkDataSetFolders, 'S14/S14NM2/S14NM2HP'];

%
mkDataSetFolders = [mkDataSetFolders, 'S1/S1NM2/S1NM2NE'];
%mkDataSetFolders = [mkDataSetFolders, 'S1/S1NM1/S1NM1NE'];
[~, n] = size(mkDataSetFolders);
[tmpStr, ~] = strsplit(mkDataSetFolders(n), '/');
mkLabelCur = tmpStr(1,3);
mkLabelCurN = strlength(mkLabelCur);
mkLabelCur = extractBetween(mkLabelCur, mkLabelCurN-1, mkLabelCurN);
mkLabels = [mkLabels, mkLabelCur];
%mkDataSetFolders = [mkDataSetFolders, 'S2/S2NM2/S2NM2NE'];
mkDataSetFolders = [mkDataSetFolders, 'S2/S2NM1/S2NM1NE'];

mkDataSetFolders = [mkDataSetFolders, 'S3/S3NM3/S3NM3NE'];
mkDataSetFolders = [mkDataSetFolders, 'S4/S4NM2/S4NM2NE'];
mkDataSetFolders = [mkDataSetFolders, 'S5/S5NM4/S5NM4NE'];
mkDataSetFolders = [mkDataSetFolders, 'S7/S7NM4/S7NM4NE'];
mkDataSetFolders = [mkDataSetFolders, 'S10/S10NM2/S10NM2NE'];
mkDataSetFolders = [mkDataSetFolders, 'S11/S11NM3/S11NM3NE'];
mkDataSetFolders = [mkDataSetFolders, 'S12/S12NM2/S12NM2NE'];
mkDataSetFolders = [mkDataSetFolders, 'S14/S14NM2/S14NM2NE'];

%
mkDataSetFolders = [mkDataSetFolders, 'S1/S1NM2/S1NM2SA'];
%mkDataSetFolders = [mkDataSetFolders, 'S1/S1NM1/S1NM1SA'];
[~, n] = size(mkDataSetFolders);
[tmpStr, ~] = strsplit(mkDataSetFolders(n), '/');
mkLabelCur = tmpStr(1,3);
mkLabelCurN = strlength(mkLabelCur);
mkLabelCur = extractBetween(mkLabelCur, mkLabelCurN-1, mkLabelCurN);
mkLabels = [mkLabels, mkLabelCur];
%mkDataSetFolders = [mkDataSetFolders, 'S2/S2NM2/S2NM2SA'];
mkDataSetFolders = [mkDataSetFolders, 'S2/S2NM1/S2NM1SA'];

mkDataSetFolders = [mkDataSetFolders, 'S3/S3NM3/S3NM3SA'];
mkDataSetFolders = [mkDataSetFolders, 'S4/S4NM2/S4NM2SA'];
mkDataSetFolders = [mkDataSetFolders, 'S5/S5NM4/S5NM4SA'];
mkDataSetFolders = [mkDataSetFolders, 'S7/S7NM4/S7NM4SA'];
mkDataSetFolders = [mkDataSetFolders, 'S10/S10NM2/S10NM2SA'];
mkDataSetFolders = [mkDataSetFolders, 'S11/S11NM3/S11NM3SA'];
mkDataSetFolders = [mkDataSetFolders, 'S12/S12NM2/S12NM2SA'];
mkDataSetFolders = [mkDataSetFolders, 'S14/S14NM2/S14NM2SA'];

%
mkDataSetFolders = [mkDataSetFolders, 'S1/S1NM2/S1NM2SC'];
%mkDataSetFolders = [mkDataSetFolders, 'S1/S1NM1/S1NM1SC'];
[~, n] = size(mkDataSetFolders);
[tmpStr, ~] = strsplit(mkDataSetFolders(n), '/');
mkLabelCur = tmpStr(1,3);
mkLabelCurN = strlength(mkLabelCur);
mkLabelCur = extractBetween(mkLabelCur, mkLabelCurN-1, mkLabelCurN);
mkLabels = [mkLabels, mkLabelCur];
%mkDataSetFolders = [mkDataSetFolders, 'S2/S2NM2/S2NM2SC'];
mkDataSetFolders = [mkDataSetFolders, 'S2/S2NM1/S2NM1SC'];

mkDataSetFolders = [mkDataSetFolders, 'S3/S3NM3/S3NM3SC'];
mkDataSetFolders = [mkDataSetFolders, 'S4/S4NM2/S4NM2SC'];
mkDataSetFolders = [mkDataSetFolders, 'S5/S5NM4/S5NM4SC'];
mkDataSetFolders = [mkDataSetFolders, 'S7/S7NM4/S7NM4SC'];
mkDataSetFolders = [mkDataSetFolders, 'S10/S10NM2/S10NM2SC'];
mkDataSetFolders = [mkDataSetFolders, 'S11/S11NM3/S11NM3SC'];
mkDataSetFolders = [mkDataSetFolders, 'S12/S12NM2/S12NM2SC'];
mkDataSetFolders = [mkDataSetFolders, 'S14/S14NM2/S14NM2SC'];

%
mkDataSetFolders = [mkDataSetFolders, 'S1/S1NM2/S1NM2SR'];
%mkDataSetFolders = [mkDataSetFolders, 'S1/S1NM1/S1NM1SR'];
[~, n] = size(mkDataSetFolders);
[tmpStr, ~] = strsplit(mkDataSetFolders(n), '/');
mkLabelCur = tmpStr(1,3);
mkLabelCurN = strlength(mkLabelCur);
mkLabelCur = extractBetween(mkLabelCur, mkLabelCurN-1, mkLabelCurN);
mkLabels = [mkLabels, mkLabelCur];
%mkDataSetFolders = [mkDataSetFolders, 'S2/S2NM2/S2NM2SR'];
mkDataSetFolders = [mkDataSetFolders, 'S2/S2NM1/S2NM1SR'];

mkDataSetFolders = [mkDataSetFolders, 'S3/S3NM3/S3NM3SR'];
mkDataSetFolders = [mkDataSetFolders, 'S4/S4NM2/S4NM2SR'];
mkDataSetFolders = [mkDataSetFolders, 'S5/S5NM4/S5NM4SR'];
mkDataSetFolders = [mkDataSetFolders, 'S7/S7NM4/S7NM4SR'];
mkDataSetFolders = [mkDataSetFolders, 'S10/S10NM2/S10NM2SR'];
mkDataSetFolders = [mkDataSetFolders, 'S11/S11NM3/S11NM3SR'];
mkDataSetFolders = [mkDataSetFolders, 'S12/S12NM2/S12NM2SR'];
mkDataSetFolders = [mkDataSetFolders, 'S14/S14NM2/S14NM2SR'];



%% Replace Sfx template with the actual value of the image dimensions
mkDataSetFolders = strrep(mkDataSetFolders, 'Sfx', dataFolderSfx);
mkLabels = strrep(mkLabels, 'Sfx', dataFolderSfx);

% Build a full path  
mkDataSetFullFolders = fullfile(dataFolder, mkDataSetFolders);


%% Create a vector of the makeup iamges Datastores with top folder lables

[~, nMakeups] = size(mkDataSetFolders);
%mkImages = cell(1,1);
[~, nLabels] = size(mkLabels);
%mkImages = cell(nLabels, 1);

fprintf("Labeling test set:\n");

%%
for i=1:nMakeups
    %fprintf(" %s,", mkDataSetFolders(i));
    
for j=1:nLabels
    
    %fprintf(" %s,", mkLabels(j));
    
    mkDataSetFullFoldersCur = mkDataSetFullFolders(i); 
    matches = contains(mkDataSetFullFoldersCur, mkLabels(j));
    
    if matches > 0
        mkDataSetFullFoldersLabel = mkDataSetFullFoldersCur(matches); 
    
        % Create Datastore for each label
        mkImage = imageDatastore(mkDataSetFullFoldersLabel, 'IncludeSubfolders', false,...
                                'LabelSource', 'none');
        mkImage.ReadFcn = readFcn;
       
        % Label all images in the Datastore with the top folder label                       
        [n, ~] = size(mkImage.Files);  
        tmpStr = strings(n,1);
        tmpStr(:) = mkLabels(j);
        mkImage.Labels = categorical(tmpStr); 

        eLabelsS = mkImage.Labels;

        [flStr, ~] = strsplit(mkDataSetFullFoldersLabel, '/');
        [~, nf] = size(flStr);
        
        mLabelsS = strings(n,1);
        mLabelsS(:) = flStr(nf-1);

        sLabelsS = strings(n,1);
        sLabelsS(:) = flStr(nf-2);

                            
        %countEachLabel(mkImage)    
    
        if (i == 1) && (j == 1)
            mkImages = mkImage;
            eLabels = eLabelsS;
            mLabels = mLabelsS;
            sLabels = sLabelsS;
        else
            mkCombined = imageDatastore(cat(1, mkImages.Files, mkImage.Files)); 
            mkCombined.Labels = cat(1, mkImages.Labels, mkImage.Labels);
            mkCombined.ReadFcn = readFcn;
            mkImages = mkCombined; 

            eLabels = cat(1, eLabels, eLabelsS);
            mLabels = cat(1, mLabels, mLabelsS);
            sLabels = cat(1, sLabels, sLabelsS);
        end
    end
    
end
%fprintf("\n"); 
end
%fprintf("\n");

mkImages = shuffle(mkImages);
    

end

