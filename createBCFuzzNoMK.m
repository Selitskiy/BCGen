function [mkImages, eLabels, mLabels, sLabels, mkDataSetFolders] = createBCFuzzTest(dataFolderTmpl, dataFolderSfx, readFcn)

%% Create a real folder
dataFolder = strrep(dataFolderTmpl, 'Sfx', dataFolderSfx);


%% Create vectors of the makeup folder templates and category labels

% Empty vectors
mkDataSetFolders = strings(0);
mkLabels = strings(0);

% Let's populate vectors one by one, making labels from the top directory

mkDataSetFolders = [mkDataSetFolders, 'S1/S1NM1/S1NM1AN'];
[~, n] = size(mkDataSetFolders);
[tmpStr, ~] = strsplit(mkDataSetFolders(n), '/');
mkLabelCur = tmpStr(1,3);
mkLabelCurN = strlength(mkLabelCur);
mkLabelCur = extractBetween(mkLabelCur, mkLabelCurN-1, mkLabelCurN);
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S2/S2NM1/S2NM1AN'];
mkDataSetFolders = [mkDataSetFolders, 'S3/S3NM1/S3NM1AN'];
mkDataSetFolders = [mkDataSetFolders, 'S3/S3NM2/S3NM2AN'];
mkDataSetFolders = [mkDataSetFolders, 'S4/S4NM1/S4NM1AN'];
mkDataSetFolders = [mkDataSetFolders, 'S5/S5NM1/S5NM1AN'];
mkDataSetFolders = [mkDataSetFolders, 'S5/S5NM2/S5NM2AN'];
mkDataSetFolders = [mkDataSetFolders, 'S5/S5NM3/S5NM3AN'];
mkDataSetFolders = [mkDataSetFolders, 'S6/S6NM1/S6NM1AN'];
mkDataSetFolders = [mkDataSetFolders, 'S7/S7NM1/S7NM1AN'];
mkDataSetFolders = [mkDataSetFolders, 'S7/S7NM2/S7NM2AN'];
mkDataSetFolders = [mkDataSetFolders, 'S7/S7NM3/S7NM3AN'];
mkDataSetFolders = [mkDataSetFolders, 'S8/S8NM1/S8NM1AN'];
mkDataSetFolders = [mkDataSetFolders, 'S9/S9NM1/S9NM1AN'];
mkDataSetFolders = [mkDataSetFolders, 'S10/S10NM1/S10NM1AN'];
mkDataSetFolders = [mkDataSetFolders, 'S11/S11NM1/S11NM1AN'];
mkDataSetFolders = [mkDataSetFolders, 'S11/S11NM2/S11NM2AN'];
mkDataSetFolders = [mkDataSetFolders, 'S12/S12NM1/S12NM1AN'];
mkDataSetFolders = [mkDataSetFolders, 'S13/S13NM1/S13NM1AN'];
mkDataSetFolders = [mkDataSetFolders, 'S14/S14NM1/S14NM1AN'];
mkDataSetFolders = [mkDataSetFolders, 'S15/S15NM1/S15NM1AN'];
mkDataSetFolders = [mkDataSetFolders, 'S16/S16NM1/S16NM1AN'];
mkDataSetFolders = [mkDataSetFolders, 'S17/S17NM1/S17NM1AN'];
mkDataSetFolders = [mkDataSetFolders, 'S18/S18NM1/S18NM1AN'];
mkDataSetFolders = [mkDataSetFolders, 'S19/S19NM1/S19NM1AN'];
mkDataSetFolders = [mkDataSetFolders, 'S20/S20NM1/S20NM1AN'];
mkDataSetFolders = [mkDataSetFolders, 'S21/S21NM1/S21NM1AN'];
%

mkDataSetFolders = [mkDataSetFolders, 'S1/S1NM1/S1NM1CE'];
[~, n] = size(mkDataSetFolders);
[tmpStr, ~] = strsplit(mkDataSetFolders(n), '/');
mkLabelCur = tmpStr(1,3);
mkLabelCurN = strlength(mkLabelCur);
mkLabelCur = extractBetween(mkLabelCur, mkLabelCurN-1, mkLabelCurN);
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S2/S2NM1/S2NM1CE'];
mkDataSetFolders = [mkDataSetFolders, 'S3/S3NM1/S3NM1CE'];
mkDataSetFolders = [mkDataSetFolders, 'S3/S3NM2/S3NM2CE'];
mkDataSetFolders = [mkDataSetFolders, 'S4/S4NM1/S4NM1CE'];
mkDataSetFolders = [mkDataSetFolders, 'S5/S5NM1/S5NM1CE'];
mkDataSetFolders = [mkDataSetFolders, 'S5/S5NM2/S5NM2CE'];
mkDataSetFolders = [mkDataSetFolders, 'S5/S5NM3/S5NM3CE'];
mkDataSetFolders = [mkDataSetFolders, 'S6/S6NM1/S6NM1CE'];
mkDataSetFolders = [mkDataSetFolders, 'S7/S7NM1/S7NM1CE'];
mkDataSetFolders = [mkDataSetFolders, 'S7/S7NM2/S7NM2CE'];
mkDataSetFolders = [mkDataSetFolders, 'S7/S7NM3/S7NM3CE'];
mkDataSetFolders = [mkDataSetFolders, 'S8/S8NM1/S8NM1CE'];
mkDataSetFolders = [mkDataSetFolders, 'S9/S9NM1/S9NM1CE'];
mkDataSetFolders = [mkDataSetFolders, 'S10/S10NM1/S10NM1CE'];
mkDataSetFolders = [mkDataSetFolders, 'S11/S11NM1/S11NM1CE'];
mkDataSetFolders = [mkDataSetFolders, 'S11/S11NM2/S11NM2CE'];
mkDataSetFolders = [mkDataSetFolders, 'S12/S12NM1/S12NM1CE'];
mkDataSetFolders = [mkDataSetFolders, 'S13/S13NM1/S13NM1CE'];
mkDataSetFolders = [mkDataSetFolders, 'S14/S14NM1/S14NM1CE'];
mkDataSetFolders = [mkDataSetFolders, 'S15/S15NM1/S15NM1CE'];
mkDataSetFolders = [mkDataSetFolders, 'S16/S16NM1/S16NM1CE'];
mkDataSetFolders = [mkDataSetFolders, 'S17/S17NM1/S17NM1CE'];
mkDataSetFolders = [mkDataSetFolders, 'S18/S18NM1/S18NM1CE'];
mkDataSetFolders = [mkDataSetFolders, 'S19/S19NM1/S19NM1CE'];
mkDataSetFolders = [mkDataSetFolders, 'S20/S20NM1/S20NM1CE'];
mkDataSetFolders = [mkDataSetFolders, 'S21/S21NM1/S21NM1CE'];
%

mkDataSetFolders = [mkDataSetFolders, 'S1/S1NM1/S1NM1DS'];
[~, n] = size(mkDataSetFolders);
[tmpStr, ~] = strsplit(mkDataSetFolders(n), '/');
mkLabelCur = tmpStr(1,3);
mkLabelCurN = strlength(mkLabelCur);
mkLabelCur = extractBetween(mkLabelCur, mkLabelCurN-1, mkLabelCurN);
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S2/S2NM1/S2NM1DS'];
mkDataSetFolders = [mkDataSetFolders, 'S3/S3NM1/S3NM1DS'];
mkDataSetFolders = [mkDataSetFolders, 'S3/S3NM2/S3NM2DS'];
mkDataSetFolders = [mkDataSetFolders, 'S4/S4NM1/S4NM1DS'];
mkDataSetFolders = [mkDataSetFolders, 'S5/S5NM1/S5NM1DS'];
mkDataSetFolders = [mkDataSetFolders, 'S5/S5NM2/S5NM2DS'];
mkDataSetFolders = [mkDataSetFolders, 'S5/S5NM3/S5NM3DS'];
mkDataSetFolders = [mkDataSetFolders, 'S6/S6NM1/S6NM1DS'];
mkDataSetFolders = [mkDataSetFolders, 'S7/S7NM1/S7NM1DS'];
mkDataSetFolders = [mkDataSetFolders, 'S7/S7NM2/S7NM2DS'];
mkDataSetFolders = [mkDataSetFolders, 'S7/S7NM3/S7NM3DS'];
mkDataSetFolders = [mkDataSetFolders, 'S8/S8NM1/S8NM1DS'];
mkDataSetFolders = [mkDataSetFolders, 'S9/S9NM1/S9NM1DS'];
mkDataSetFolders = [mkDataSetFolders, 'S10/S10NM1/S10NM1DS'];
mkDataSetFolders = [mkDataSetFolders, 'S11/S11NM1/S11NM1DS'];
mkDataSetFolders = [mkDataSetFolders, 'S11/S11NM2/S11NM2DS'];
mkDataSetFolders = [mkDataSetFolders, 'S12/S12NM1/S12NM1DS'];
mkDataSetFolders = [mkDataSetFolders, 'S13/S13NM1/S13NM1DS'];
mkDataSetFolders = [mkDataSetFolders, 'S14/S14NM1/S14NM1DS'];
mkDataSetFolders = [mkDataSetFolders, 'S15/S15NM1/S15NM1DS'];
mkDataSetFolders = [mkDataSetFolders, 'S16/S16NM1/S16NM1DS'];
mkDataSetFolders = [mkDataSetFolders, 'S17/S17NM1/S17NM1DS'];
mkDataSetFolders = [mkDataSetFolders, 'S18/S18NM1/S18NM1DS'];
mkDataSetFolders = [mkDataSetFolders, 'S19/S19NM1/S19NM1DS'];
mkDataSetFolders = [mkDataSetFolders, 'S20/S20NM1/S20NM1DS'];
mkDataSetFolders = [mkDataSetFolders, 'S21/S21NM1/S21NM1DS'];
%

mkDataSetFolders = [mkDataSetFolders, 'S1/S1NM1/S1NM1HP'];
[~, n] = size(mkDataSetFolders);
[tmpStr, ~] = strsplit(mkDataSetFolders(n), '/');
mkLabelCur = tmpStr(1,3);
mkLabelCurN = strlength(mkLabelCur);
mkLabelCur = extractBetween(mkLabelCur, mkLabelCurN-1, mkLabelCurN);
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S2/S2NM1/S2NM1HP'];
mkDataSetFolders = [mkDataSetFolders, 'S3/S3NM1/S3NM1HP'];
mkDataSetFolders = [mkDataSetFolders, 'S3/S3NM2/S3NM2HP'];
mkDataSetFolders = [mkDataSetFolders, 'S4/S4NM1/S4NM1HP'];
mkDataSetFolders = [mkDataSetFolders, 'S5/S5NM1/S5NM1HP'];
mkDataSetFolders = [mkDataSetFolders, 'S5/S5NM2/S5NM2HP'];
mkDataSetFolders = [mkDataSetFolders, 'S5/S5NM3/S5NM3HP'];
mkDataSetFolders = [mkDataSetFolders, 'S6/S6NM1/S6NM1HP'];
mkDataSetFolders = [mkDataSetFolders, 'S7/S7NM1/S7NM1HP'];
mkDataSetFolders = [mkDataSetFolders, 'S7/S7NM2/S7NM2HP'];
mkDataSetFolders = [mkDataSetFolders, 'S7/S7NM3/S7NM3HP'];
mkDataSetFolders = [mkDataSetFolders, 'S8/S8NM1/S8NM1HP'];
mkDataSetFolders = [mkDataSetFolders, 'S9/S9NM1/S9NM1HP'];
mkDataSetFolders = [mkDataSetFolders, 'S10/S10NM1/S10NM1HP'];
mkDataSetFolders = [mkDataSetFolders, 'S11/S11NM1/S11NM1HP'];
mkDataSetFolders = [mkDataSetFolders, 'S11/S11NM2/S11NM2HP'];
mkDataSetFolders = [mkDataSetFolders, 'S12/S12NM1/S12NM1HP'];
mkDataSetFolders = [mkDataSetFolders, 'S13/S13NM1/S13NM1HP'];
mkDataSetFolders = [mkDataSetFolders, 'S14/S14NM1/S14NM1HP'];
mkDataSetFolders = [mkDataSetFolders, 'S15/S15NM1/S15NM1HP'];
mkDataSetFolders = [mkDataSetFolders, 'S16/S16NM1/S16NM1HP'];
mkDataSetFolders = [mkDataSetFolders, 'S17/S17NM1/S17NM1HP'];
mkDataSetFolders = [mkDataSetFolders, 'S18/S18NM1/S18NM1HP'];
mkDataSetFolders = [mkDataSetFolders, 'S19/S19NM1/S19NM1HP'];
mkDataSetFolders = [mkDataSetFolders, 'S20/S20NM1/S20NM1HP'];
mkDataSetFolders = [mkDataSetFolders, 'S21/S21NM1/S21NM1HP'];
%

mkDataSetFolders = [mkDataSetFolders, 'S1/S1NM1/S1NM1NE'];
[~, n] = size(mkDataSetFolders);
[tmpStr, ~] = strsplit(mkDataSetFolders(n), '/');
mkLabelCur = tmpStr(1,3);
mkLabelCurN = strlength(mkLabelCur);
mkLabelCur = extractBetween(mkLabelCur, mkLabelCurN-1, mkLabelCurN);
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S2/S2NM1/S2NM1NE'];
mkDataSetFolders = [mkDataSetFolders, 'S3/S3NM1/S3NM1NE'];
mkDataSetFolders = [mkDataSetFolders, 'S3/S3NM2/S3NM2NE'];
mkDataSetFolders = [mkDataSetFolders, 'S4/S4NM1/S4NM1NE'];
mkDataSetFolders = [mkDataSetFolders, 'S5/S5NM1/S5NM1NE'];
mkDataSetFolders = [mkDataSetFolders, 'S5/S5NM2/S5NM2NE'];
mkDataSetFolders = [mkDataSetFolders, 'S5/S5NM3/S5NM3NE'];
mkDataSetFolders = [mkDataSetFolders, 'S6/S6NM1/S6NM1NE'];
mkDataSetFolders = [mkDataSetFolders, 'S7/S7NM1/S7NM1NE'];
mkDataSetFolders = [mkDataSetFolders, 'S7/S7NM2/S7NM2NE'];
mkDataSetFolders = [mkDataSetFolders, 'S7/S7NM3/S7NM3NE'];
mkDataSetFolders = [mkDataSetFolders, 'S8/S8NM1/S8NM1NE'];
mkDataSetFolders = [mkDataSetFolders, 'S9/S9NM1/S9NM1NE'];
mkDataSetFolders = [mkDataSetFolders, 'S10/S10NM1/S10NM1NE'];
mkDataSetFolders = [mkDataSetFolders, 'S11/S11NM1/S11NM1NE'];
mkDataSetFolders = [mkDataSetFolders, 'S11/S11NM2/S11NM2NE'];
mkDataSetFolders = [mkDataSetFolders, 'S12/S12NM1/S12NM1NE'];
mkDataSetFolders = [mkDataSetFolders, 'S13/S13NM1/S13NM1NE'];
mkDataSetFolders = [mkDataSetFolders, 'S14/S14NM1/S14NM1NE'];
mkDataSetFolders = [mkDataSetFolders, 'S15/S15NM1/S15NM1NE'];
mkDataSetFolders = [mkDataSetFolders, 'S16/S16NM1/S16NM1NE'];
mkDataSetFolders = [mkDataSetFolders, 'S17/S17NM1/S17NM1NE'];
mkDataSetFolders = [mkDataSetFolders, 'S18/S18NM1/S18NM1NE'];
mkDataSetFolders = [mkDataSetFolders, 'S19/S19NM1/S19NM1NE'];
mkDataSetFolders = [mkDataSetFolders, 'S20/S20NM1/S20NM1NE'];
mkDataSetFolders = [mkDataSetFolders, 'S21/S21NM1/S21NM1NE'];
%

mkDataSetFolders = [mkDataSetFolders, 'S1/S1NM1/S1NM1SA'];
[~, n] = size(mkDataSetFolders);
[tmpStr, ~] = strsplit(mkDataSetFolders(n), '/');
mkLabelCur = tmpStr(1,3);
mkLabelCurN = strlength(mkLabelCur);
mkLabelCur = extractBetween(mkLabelCur, mkLabelCurN-1, mkLabelCurN);
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S2/S2NM1/S2NM1SA'];
mkDataSetFolders = [mkDataSetFolders, 'S3/S3NM1/S3NM1SA'];
mkDataSetFolders = [mkDataSetFolders, 'S3/S3NM2/S3NM2SA'];
mkDataSetFolders = [mkDataSetFolders, 'S4/S4NM1/S4NM1SA'];
mkDataSetFolders = [mkDataSetFolders, 'S5/S5NM1/S5NM1SA'];
mkDataSetFolders = [mkDataSetFolders, 'S5/S5NM2/S5NM2SA'];
mkDataSetFolders = [mkDataSetFolders, 'S5/S5NM3/S5NM3SA'];
mkDataSetFolders = [mkDataSetFolders, 'S6/S6NM1/S6NM1SA'];
mkDataSetFolders = [mkDataSetFolders, 'S7/S7NM1/S7NM1SA'];
mkDataSetFolders = [mkDataSetFolders, 'S7/S7NM2/S7NM2SA'];
mkDataSetFolders = [mkDataSetFolders, 'S7/S7NM3/S7NM3SA'];
mkDataSetFolders = [mkDataSetFolders, 'S8/S8NM1/S8NM1SA'];
mkDataSetFolders = [mkDataSetFolders, 'S9/S9NM1/S9NM1SA'];
mkDataSetFolders = [mkDataSetFolders, 'S10/S10NM1/S10NM1SA'];
mkDataSetFolders = [mkDataSetFolders, 'S11/S11NM1/S11NM1SA'];
mkDataSetFolders = [mkDataSetFolders, 'S11/S11NM2/S11NM2SA'];
mkDataSetFolders = [mkDataSetFolders, 'S12/S12NM1/S12NM1SA'];
mkDataSetFolders = [mkDataSetFolders, 'S13/S13NM1/S13NM1SA'];
mkDataSetFolders = [mkDataSetFolders, 'S14/S14NM1/S14NM1SA'];
mkDataSetFolders = [mkDataSetFolders, 'S15/S15NM1/S15NM1SA'];
mkDataSetFolders = [mkDataSetFolders, 'S16/S16NM1/S16NM1SA'];
mkDataSetFolders = [mkDataSetFolders, 'S17/S17NM1/S17NM1SA'];
mkDataSetFolders = [mkDataSetFolders, 'S18/S18NM1/S18NM1SA'];
mkDataSetFolders = [mkDataSetFolders, 'S19/S19NM1/S19NM1SA'];
mkDataSetFolders = [mkDataSetFolders, 'S20/S20NM1/S20NM1SA'];
mkDataSetFolders = [mkDataSetFolders, 'S21/S21NM1/S21NM1SA'];
%

mkDataSetFolders = [mkDataSetFolders, 'S1/S1NM1/S1NM1SC'];
[~, n] = size(mkDataSetFolders);
[tmpStr, ~] = strsplit(mkDataSetFolders(n), '/');
mkLabelCur = tmpStr(1,3);
mkLabelCurN = strlength(mkLabelCur);
mkLabelCur = extractBetween(mkLabelCur, mkLabelCurN-1, mkLabelCurN);
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S2/S2NM1/S2NM1SC'];
mkDataSetFolders = [mkDataSetFolders, 'S3/S3NM1/S3NM1SC'];
mkDataSetFolders = [mkDataSetFolders, 'S3/S3NM2/S3NM2SC'];
mkDataSetFolders = [mkDataSetFolders, 'S4/S4NM1/S4NM1SC'];
mkDataSetFolders = [mkDataSetFolders, 'S5/S5NM1/S5NM1SC'];
mkDataSetFolders = [mkDataSetFolders, 'S5/S5NM2/S5NM2SC'];
mkDataSetFolders = [mkDataSetFolders, 'S5/S5NM3/S5NM3SC'];
mkDataSetFolders = [mkDataSetFolders, 'S6/S6NM1/S6NM1SC'];
mkDataSetFolders = [mkDataSetFolders, 'S7/S7NM1/S7NM1SC'];
mkDataSetFolders = [mkDataSetFolders, 'S7/S7NM2/S7NM2SC'];
mkDataSetFolders = [mkDataSetFolders, 'S7/S7NM3/S7NM3SC'];
mkDataSetFolders = [mkDataSetFolders, 'S8/S8NM1/S8NM1SC'];
mkDataSetFolders = [mkDataSetFolders, 'S9/S9NM1/S9NM1SC'];
mkDataSetFolders = [mkDataSetFolders, 'S10/S10NM1/S10NM1SC'];
mkDataSetFolders = [mkDataSetFolders, 'S11/S11NM1/S11NM1SC'];
mkDataSetFolders = [mkDataSetFolders, 'S11/S11NM2/S11NM2SC'];
mkDataSetFolders = [mkDataSetFolders, 'S12/S12NM1/S12NM1SC'];
mkDataSetFolders = [mkDataSetFolders, 'S13/S13NM1/S13NM1SC'];
mkDataSetFolders = [mkDataSetFolders, 'S14/S14NM1/S14NM1SC'];
mkDataSetFolders = [mkDataSetFolders, 'S15/S15NM1/S15NM1SC'];
mkDataSetFolders = [mkDataSetFolders, 'S16/S16NM1/S16NM1SC'];
mkDataSetFolders = [mkDataSetFolders, 'S17/S17NM1/S17NM1SC'];
mkDataSetFolders = [mkDataSetFolders, 'S18/S18NM1/S18NM1SC'];
mkDataSetFolders = [mkDataSetFolders, 'S19/S19NM1/S19NM1SC'];
mkDataSetFolders = [mkDataSetFolders, 'S20/S20NM1/S20NM1SC'];
mkDataSetFolders = [mkDataSetFolders, 'S21/S21NM1/S21NM1SC'];
%

mkDataSetFolders = [mkDataSetFolders, 'S1/S1NM1/S1NM1SR'];
[~, n] = size(mkDataSetFolders);
[tmpStr, ~] = strsplit(mkDataSetFolders(n), '/');
mkLabelCur = tmpStr(1,3);
mkLabelCurN = strlength(mkLabelCur);
mkLabelCur = extractBetween(mkLabelCur, mkLabelCurN-1, mkLabelCurN);
mkLabels = [mkLabels, mkLabelCur];
mkDataSetFolders = [mkDataSetFolders, 'S2/S2NM1/S2NM1SR'];
mkDataSetFolders = [mkDataSetFolders, 'S3/S3NM1/S3NM1SR'];
mkDataSetFolders = [mkDataSetFolders, 'S3/S3NM2/S3NM2SR'];
mkDataSetFolders = [mkDataSetFolders, 'S4/S4NM1/S4NM1SR'];
mkDataSetFolders = [mkDataSetFolders, 'S5/S5NM1/S5NM1SR'];
mkDataSetFolders = [mkDataSetFolders, 'S5/S5NM2/S5NM2SR'];
mkDataSetFolders = [mkDataSetFolders, 'S5/S5NM3/S5NM3SR'];
mkDataSetFolders = [mkDataSetFolders, 'S6/S6NM1/S6NM1SR'];
mkDataSetFolders = [mkDataSetFolders, 'S7/S7NM1/S7NM1SR'];
mkDataSetFolders = [mkDataSetFolders, 'S7/S7NM2/S7NM2SR'];
mkDataSetFolders = [mkDataSetFolders, 'S7/S7NM3/S7NM3SR'];
mkDataSetFolders = [mkDataSetFolders, 'S8/S8NM1/S8NM1SR'];
mkDataSetFolders = [mkDataSetFolders, 'S9/S9NM1/S9NM1SR'];
mkDataSetFolders = [mkDataSetFolders, 'S10/S10NM1/S10NM1SR'];
mkDataSetFolders = [mkDataSetFolders, 'S11/S11NM1/S11NM1SR'];
mkDataSetFolders = [mkDataSetFolders, 'S11/S11NM2/S11NM2SR'];
mkDataSetFolders = [mkDataSetFolders, 'S12/S12NM1/S12NM1SR'];
mkDataSetFolders = [mkDataSetFolders, 'S13/S13NM1/S13NM1SR'];
mkDataSetFolders = [mkDataSetFolders, 'S14/S14NM1/S14NM1SR'];
mkDataSetFolders = [mkDataSetFolders, 'S15/S15NM1/S15NM1SR'];
mkDataSetFolders = [mkDataSetFolders, 'S16/S16NM1/S16NM1SR'];
mkDataSetFolders = [mkDataSetFolders, 'S17/S17NM1/S17NM1SR'];
mkDataSetFolders = [mkDataSetFolders, 'S18/S18NM1/S18NM1SR'];
mkDataSetFolders = [mkDataSetFolders, 'S19/S19NM1/S19NM1SR'];
mkDataSetFolders = [mkDataSetFolders, 'S20/S20NM1/S20NM1SR'];
mkDataSetFolders = [mkDataSetFolders, 'S21/S21NM1/S21NM1SR'];


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

