%% test importation using readmatrix
% import data using readmatrix
file_name = "Y:\personal\ojuba.mezisashe\Sentences\TIMIT_dp\DEV\DR1\FAKS0\aa\SI943-1.csv";


readmatrix(file_name)

%% test normalizing different of the same phoneme (padding with zeros and aligning)

% look into using datastore instead of looping through data

target_directory = "Y:\personal\ojuba.mezisashe\Sentences\TIMIT_dp\DEV\DR1\FAKS0\aa";
% % files = dir(fullfile(target_directory, '*.csv'));
% % 
% % 
% % for n = 1 : length(files)
% %     file_name = files(n);
% % 
% %     % do processing here
% % end

% % feature_data = datastore(target_directory, "FileExtensions",[".csv"],"IncludeSubfolders",true, "Type", "tabulartext");
% % 
% % data = readall(feature_data);

S = dir(fullfile(target_directory,'*.csv'));
for k = 1:numel(S)
    F = fullfile(target_directory,S(k).name);
    S(k).data = readmatrix(F);
end

% pad all unfilled samples with zero then fill in the actual data
matrix_depth = 0;
for n = 1 : length(S)
    if length(S(n).data) > matrix_depth
        matrix_depth = length(S(n).data);
    end
    % do processing here
end
data = zeros(22, matrix_depth, length(S)); 

% fill in actual data
for n = 1 : length(S)
    data(:,1:length(S(n).data), n) = S(n).data;
    % do processing here
end

normalized_data = normalize(data,3);
average = mean(data,3);