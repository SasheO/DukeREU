% realtime audio processing: https://www.mathworks.com/help/audio/gs/real-time-audio-in-matlab.html
FREQUENCIES_FOR_5_CHANNELS_TABLE_1 = [350, 607, 1053,1827,3170, 8700];
FREQUENCIES_FOR_16_CHANNELS_TABLE_1 = [250, 416, 494, 587, 697, 828, 983, 1168, 1387, 1648, 1958, 2326, 2762, 3281, 3898, 4630, 8700];
FREQUENCIES_FOR_6_CHANNELS_TABLE_2 = [188, 563, 1063, 1813, 2938, 4813, 7938];
FREQUENCIES_FOR_22_CHANNELS_TABLE_2 = [188, 313, 438, 563, 688, 813, 938, 1063, 1188, 1438, 1688, 1938, 2313, 2688, 3188, 3688, 4313, 5603, 5938, 6938, 7938];

%%% CHANGE THESE %%%
frequency_ranges = FREQUENCIES_FOR_22_CHANNELS_TABLE_2;
input_filename = 'LL-Q1860_(eng)-Vealhurl-cosmos2.wav';
LEN_TIME_QUANTIZED_MS = 8;
OVERLAP_MS = 0;

if (OVERLAP_MS == 0 || OVERLAP_MS > LEN_TIME_QUANTIZED_MS)
    OVERLAP_MS = LEN_TIME_QUANTIZED_MS;
end

[ ~, Fs ] = audioread(input_filename,[1:2]);

frameLength = (Fs/1000)*LEN_TIME_QUANTIZED_MS; % the number of values equal to LEN_TIME_QUANTIZED_MS milliseconds. this should give an integer;
overlap_step = int32((Fs/1000)*OVERLAP_MS); % the number of values equal to the millisecond overlap

fileReader = dsp.AudioFileReader( ...
    input_filename, ...
    'SamplesPerFrame',frameLength);
deviceWriter = audioDeviceWriter( ...
    'SampleRate',fileReader.SampleRate);

bandpassed_frequencies_low_to_high = rand(length(frequency_ranges)-1,frameLength);
enveloped_frequencies_low_to_high = rand(size(bandpassed_frequencies_low_to_high, 1),frameLength);
rms_energy_values = rand(size(enveloped_frequencies_low_to_high, 1), ceil((length(enveloped_frequencies_low_to_high)-frameLength)/overlap_step)+1); % to do: fix the error with

while ~isDone(fileReader)                   %<--- new lines of code
    signal = fileReader();                  %<---
    
    disp(0);

    % slow as hell
    for indx = 1:1:length(frequency_ranges)-1
        low = frequency_ranges(indx); high = frequency_ranges(indx+1);
        bandpassed_frequencies_low_to_high(indx,:) = bandpass(signal,[low high],Fs);
    end

    disp(signal);

    disp(1);

    for indx = 1:1:length(frequency_ranges)-1
        enveloped_frequencies_low_to_high(indx,:) = abs(bandpassed_frequencies_low_to_high(indx,:));
    end

    % to do: fix this
    % for i=1:size(enveloped_frequencies_low_to_high, 1)
    %     frequencies = enveloped_frequencies_low_to_high(i, :);
    %     rms_energy_values(i)
    %     while final_indx < length(frequencies)
    %         rms_list(rms_list_indx) = sqrt(mean(frequencies(indx:final_indx).^2, "all"));
    %         indx = indx + overlap_step;
    %         final_indx = final_indx + overlap_step;
    %         rms_list_indx = rms_list_indx + 1;
    %     end
    %     final_frame_length = length(frequencies(indx:end));
    %     rms_list(end) = sqrt(mean(frequencies(indx:end).^2, "all"));
    %     rms_energy_values(i, :) = rms_list;
    % end

    disp(2);

    deviceWriter(signal);                   %<---
end                                         %<---

release(fileReader)                         %<---
release(deviceWriter)  