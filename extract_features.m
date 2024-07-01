%% feature extraction: 
		% extract features into similar directory with good naming convention.
		% 	look at the directory structure for a minute test. test the creation of folders.
		% 	plan how you will separate phonemes (maybe folders?)
		% 	run code on phn files to extract phoneme features.
procs = [];
procs = Ensure_field(procs, 'map_name', 'EXTRACT_FEATURES');

procs = Ensure_rate_params(procs);
procs = Append_front_end_processes(procs);
procs = Append_process(procs, 'FFT_filterbank_proc');
procs = Append_process(procs, 'Power_sum_envelope_proc');
procs = Append_process(procs, 'Gain_proc');

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
    
    
    for n = 1 : height(times_to_phoneme)
        phoneme_start_indx = str2double(times_to_phoneme(n,1))+1;
        phoneme_end_indx = str2double(times_to_phoneme(n,2))+1;
        phoneme_label = times_to_phoneme(n,3);
        processed_sound = Process(procs, wav_file_sound(phoneme_start_indx:phoneme_end_indx));
        mkdir(destination,phoneme_label);
        
        file_num = 1;
        while isfile(destination + "/"+ phoneme_label + "/" + file_name.name(1:end-4)+ "-"+ num2str(file_num) + ".csv")
            file_num = file_num +1;
        end
        writematrix(processed_sound, destination + "/"+ phoneme_label + "/" + file_name.name(1:end-4)+ "-"+ num2str(file_num) + ".csv")
    end

end

