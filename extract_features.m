%% check file directory listing all relevant files there.

file_directory = "Y:\personal\chu.kevin\Sentences\TIMIT_dp\DEV\DR1\FAKS0";
files = dir(fullfile(file_directory, '*.PHN'));

for n = 1 : length(files)
    file_name = files(n);

    % do processing here
end

%% test to extract feature from one file
file_name = "Y:\personal\chu.kevin\Sentences\TIMIT_dp\DEV\DR1\FAKS0\SI2203.";

phn_file = file_name+"PHN";
wav_file = file_name+"WAV";


opts = detectImportOptions(phn_file, "FileType","delimitedtext");
opts.ExtraColumnsRule = 'ignore';
disp(opts)
opts = setvartype(opts,{'Var1','Var2','Var3',},{'int32','int32','string'});
times_to_phoneme = table2array(readtable(phn_file,opts));

%% 		test opening one file with a single phoneme. 
% do for loop for all phonemes.

bits = 16;
file_name = "Y:\personal\chu.kevin\Sentences\TIMIT_dp\DEV\DR1\FAKS0\SI2203.WAV";
[wav_file_sound,Fs] = audioread(file_name);

for n = 1 : height(times_to_phoneme)
    phoneme_start_indx = str2double(times_to_phoneme(n,1))+1;
    phoneme_end_indx = str2double(times_to_phoneme(n,2))+1;
    phoneme_label = times_to_phoneme(n,3);
    sound(wav_file_sound(phoneme_start_indx:phoneme_end_indx),Fs,bits);
    pause(0.3);
end

%% feature extraction: (sample here https://www.mathworks.com/matlabcentral/fileexchange/159688-nucleus-toolbox)

% file_name = "Y:\personal\chu.kevin\Sentences\TIMIT_dp\DEV\DR1\FAKS0\SI2203.WAV";
% procs = ACE_map;
% processed_sound = Process(procs, wav_file_sound);
% Plot_sequence(processed_sound);

bits = 16;
file_name = "Y:\personal\chu.kevin\Sentences\TIMIT_dp\DEV\DR1\FAKS0\SI2203.WAV";
[wav_file_sound,Fs] = audioread(file_name);

procs = [];
procs = Ensure_field(procs, 'map_name', 'EXTRACT_FEATURES');

procs = Ensure_rate_params(procs);
procs = Append_front_end_processes(procs);
procs = Append_process(procs, 'FFT_filterbank_proc');
procs = Append_process(procs, 'Power_sum_envelope_proc');
procs = Append_process(procs, 'Gain_proc');

processed_sound = Process(procs, wav_file_sound(12248:14891));
writematrix(processed_sound, "processed_sound.csv")


%% feature extraction: 
		% extract features into similar directory with good naming convention.
		% 	look at the directory structure for a minute test. test the creation of folders.
		% 	plan how you will separate phonemes (maybe folders?)
		% 	run code on phn files to extract phoneme features.

source = "Y:/personal/chu.kevin/Sentences/TIMIT_dp/DEV/DR1/FAKS0"; % should start with Y:/
destination = "Y:/personal/ojuba.mezisashe"; % should start with Y:/

folders_source = strsplit(source, "/"); % the first three are what is switched out with destination.

for n = 4 : length(folders_source)
    mkdir(destination,folders_source(n))
    destination = destination + "/" + folders_source(n);
end

files = dir(fullfile(source, '*.PHN'));

for file_indx = 1:length(files)
    
    file_name = files(file_indx);

    if file_name.name(1)=="."
        continue % skip all the "._ ..." files
    end

    phn_file = file_name.folder + "/" + file_name.name;
    wav_file = file_name.folder + "/" + file_name.name(1:end-3)+"WAV";

    opts = detectImportOptions(phn_file, "FileType","delimitedtext");
    opts.ExtraColumnsRule = 'ignore';
    disp(opts)
    opts = setvartype(opts,{'Var1','Var2','Var3',},{'int32','int32','string'});
    times_to_phoneme = table2array(readtable(phn_file,opts));

    [wav_file_sound,Fs] = audioread(wav_file);
    
    
    
    % for n = 1 : height(times_to_phoneme)
    %     phoneme_start_indx = str2double(times_to_phoneme(n,1))+1;
    %     phoneme_end_indx = str2double(times_to_phoneme(n,2))+1;
    %     phoneme_label = times_to_phoneme(n,3);
    %     processed_sound = wav_file_sound(phoneme_start_indx:phoneme_end_indx);
    %     mkdir(destination,phoneme_label);
    % 
    %     while isfile(filename)
    %         statements
    %     end
    % end
end


% if isfile(filename)
%      % File exists.
% else
%      % File does not exist.
% end