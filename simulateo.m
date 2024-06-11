% realtime audio processing: https://www.mathworks.com/help/audio/gs/real-time-audio-in-matlab.html
FREQUENCIES_FOR_5_CHANNELS_TABLE_1 = [350, 607, 1053,1827,3170, 8700];
FREQUENCIES_FOR_16_CHANNELS_TABLE_1 = [250, 416, 494, 587, 697, 828, 983, 1168, 1387, 1648, 1958, 2326, 2762, 3281, 3898, 4630, 8700];
FREQUENCIES_FOR_6_CHANNELS_TABLE_2 = [188, 563, 1063, 1813, 2938, 4813, 7938];
FREQUENCIES_FOR_22_CHANNELS_TABLE_2 = [188, 313, 438, 563, 688, 813, 938, 1063, 1188, 1438, 1688, 1938, 2313, 2688, 3188, 3688, 4313, 5603, 5938, 6938, 7938];

%%% CHANGE THESE %%%
frequency_ranges = FREQUENCIES_FOR_6_CHANNELS_TABLE_2;
filename = 'LL-Q1860_(eng)-Vealhurl-cosmos2.wav';
LEN_TIME_QUANTIZED_MS = 8;
OVERLAP_MS = 6;


[data,Fs] = audioread(filename);
times = 0:1/Fs:(length(data)-1)/Fs; % this time matrix will start from zero, increment at 1/Fs and last value will be at (length(X)-1)/Fs, which is equal to the duration of the signal (5s) in this case

disp(["length of data file:", length(data)]);
disp(["sample_rate:", num2str(Fs), " samples/sec"]);

% to do:
% plot spectrogram: do this later. check here https://www.youtube.com/watch?v=KU53LnEgn4w
% s = spectrogram(data);
% spectrogram(data,'yaxis');


% bandpass signal
bandpassed_frequencies_low_to_high = rand(length(frequency_ranges)-1,length(data));
for indx = [1:1:length(frequency_ranges)-1]
    low = frequency_ranges(indx); high = frequency_ranges(indx+1);
    bandpassed_frequencies_low_to_high(indx,:) = bandpass(data,[low high],Fs);
end

disp(["bandpassed_frequencies_low_to_high: ",size(bandpassed_frequencies_low_to_high)]);

% plot the bandpassed frequencies
subplot(size(bandpassed_frequencies_low_to_high,1)+1,1,1);
plot(times,data,"LineWidth",1.5);
for indx = 1:1:length(frequency_ranges)-1
    subplot(size(bandpassed_frequencies_low_to_high,1)+1,1,indx+1);
    low = frequency_ranges(indx); 
    high = frequency_ranges(indx+1);
    plot(times,bandpassed_frequencies_low_to_high(indx,:),"LineWidth",1.5);
    xlabel('time (sec)'); ylabel("Amplitude");
    title(sprintf('Band-Pass Filter ({%d}-{%d} kHz)', low, high));
end
figure();


% plot the enveloped/rectified
enveloped_frequencies_low_to_high = rand(size(bandpassed_frequencies_low_to_high, 1),length(data));
for indx = [1:1:length(frequency_ranges)-1]
    enveloped_frequencies_low_to_high(indx,:) = abs(bandpassed_frequencies_low_to_high(indx,:));
end
% plot the rectified frequencies
subplot(size(enveloped_frequencies_low_to_high,1)+1,1,1);
plot(times,data,"LineWidth",1.5);
for indx = [1:1:length(frequency_ranges)-1]
    subplot(size(enveloped_frequencies_low_to_high,1)+1,1,indx+1);
    low = frequency_ranges(indx); 
    high = frequency_ranges(indx+1);
    plot(times,enveloped_frequencies_low_to_high(indx,:),"LineWidth",1.5);
    xlabel('time (sec)'); ylabel("Amplitude");
    title(sprintf('Rectified (envelope) ({%d}-{%d} kHz)', low, high));
end
figure();

% to do: get energy -> current -> reconstruction
% to do: buffer sound and process 8ms each, do overlap
% to do: add reverb, noise
if (OVERLAP_MS == 0 || OVERLAP_MS > LEN_TIME_QUANTIZED_MS)
    OVERLAP_MS = LEN_TIME_QUANTIZED_MS;
end

% to do: calculate the pre-assigned size of rms energy values based on size
% and overlap
frame_step = int32((Fs/1000)*LEN_TIME_QUANTIZED_MS); % the number of values equal to LEN_TIME_QUANTIZED_MS milliseconds. this should give an integer
overlap_step = int32((Fs/1000)*OVERLAP_MS); % the number of values equal to the millisecond overlap
rms_energy_values = rand(size(enveloped_frequencies_low_to_high, 1), ceil((length(enveloped_frequencies_low_to_high)-frame_step)/overlap_step)+1);

disp(["rms_energy_values", size(rms_energy_values)]);

for i=1:size(enveloped_frequencies_low_to_high, 1)
    frequencies = enveloped_frequencies_low_to_high(i, :);
    disp(length(frequencies));
    rms_list = rand(1, size(rms_energy_values,2));
    indx = 1;
    final_indx = frame_step;
    rms_list_indx = 1;
    while final_indx < length(frequencies)
        rms_list(rms_list_indx) = sqrt(mean(frequencies(indx:final_indx).^2, "all"));
        indx = indx + overlap_step;
        final_indx = final_indx + overlap_step;
        rms_list_indx = rms_list_indx + 1;
    end
    final_frame_length = length(frequencies(indx:end));
    rms_list(end) = sqrt(mean(frequencies(indx:end).^2, "all"));
    rms_energy_values(i, :) = rms_list;
end

signal = rand(size(enveloped_frequencies_low_to_high));

for indx=1:length(frequency_ranges)-1
    start_indx = 1;
    average_frequency=(frequency_ranges(indx)+frequency_ranges(indx+1))/2;
    disp(average_frequency);
    period = OVERLAP_MS/1000;
    for rms_energy_indx=1:length(rms_energy_values)-1
        amplitude = rms_energy_values(indx,rms_energy_indx); % amplitude = rms energy
        samples = linspace(0, period, Fs*period);
        y = amplitude * sin(2 * pi * samples * average_frequency);
        signal(indx,start_indx:start_indx+overlap_step-1) = y;
        start_indx = start_indx + overlap_step;
    end
    % to do: construct sinusoid for last frame
    amplitude = rms_energy_values(indx,end);  % amplitude = rms energy
    period = period*(final_frame_length/double(overlap_step));
    samples = linspace(0, period, Fs*period);
    y = amplitude * sin(2 * pi * samples * average_frequency);
    signal(indx,start_indx:end) = y;
end

subplot(size(signal,1)+1,1,1);
plot(times,sum(signal, 1),"LineWidth",1.5);
title('Reconstructed signal ');
for indx = 1:1:length(frequency_ranges)-1
    subplot(size(signal,1)+1,1,indx+1);
    low = frequency_ranges(indx); 
    high = frequency_ranges(indx+1);
    plot(times,signal(indx,:),"LineWidth",1.5);
    xlabel('time (sec)'); ylabel("Amplitude");
    title(sprintf('Reconstructed signal ({%d}-{%d} kHz)', low, high));
end
figure();

% to do: make sound normal
filename = "cosmos_reconstructed.wav";
signal=sum(signal, 1);
audiowrite(filename,signal,Fs);

