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

processed_sound5 = Process(procs, wav_file_sound);