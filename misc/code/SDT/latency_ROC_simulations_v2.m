num_trials=10000; %number of simulated trials
noise_dur=5;     %mean noise duration, in seconds
sig_dur=3;       %signal duration, in seconds
noise_hazard=6;  
signal_hazard=2; 

% noise_hazard=4;  %mean time to lick during noise (neutral bias)
% signal_hazard=4; %mean time to lick during signal (neutral bias)
signal_start_bins=10; % number of bins of signal start time

noise_durs=exprnd(noise_dur,num_trials,1);
noise_lick_RT=exprnd(noise_hazard,num_trials,1);
signal_lick_RT=exprnd(signal_hazard,num_trials,1);

%% Add heterogeneity to parameters
% noise_durs2=exprnd(noise_dur,num_trials,1);
% noise_lick_RT2=exprnd(noise_hazard*2,num_trials,1);
% signal_lick_RT2=exprnd(signal_hazard*2,num_trials,1);
% 
% noise_durs=[noise_durs ; noise_durs2];
% noise_lick_RT=[noise_lick_RT ; noise_lick_RT2];
% signal_lick_RT=[signal_lick_RT ; signal_lick_RT2];
% num_trials=length(signal_lick_RT);

%%
responses=nan(num_trials,1); % trial-wise outcome vector; 0, 'no response; 1, 'hit'; 2, 'false alarm'
Hit_RTs=nan(num_trials,1); % trial-wise reaction time vector; 0, in seconds
FA_RTs=nan(num_trials,1); % trial-wise reaction time vector; 0, in seconds
Hit_noise_durs=nan(num_trials,1); %
Miss_noise_durs=nan(num_trials,1); %

for i=1:num_trials
    if noise_lick_RT(i)<=noise_durs(i)
        responses(i)=2;
        FA_RTs(i)=noise_lick_RT(i);
    else
        if signal_lick_RT(i)<=sig_dur
            responses(i)=1;
            Hit_RTs(i)=signal_lick_RT(i);
            Hit_noise_durs(i)=noise_durs(i);
        else
            responses(i)=0;
            Miss_noise_durs(i)=noise_durs(i);
        end
    end
end
%figure; histogram(responses)

%% build false alarm rate cumulative curve
Hit_starts=Hit_noise_durs(~isnan(Hit_noise_durs));
Miss_starts=Miss_noise_durs(~isnan(Miss_noise_durs));
signal_starts_list=sort([Hit_starts ; Miss_starts]);
FA_RTs_list=sort(FA_RTs(~isnan(FA_RTs)));
FA_cum_data = [FA_RTs_list ; signal_starts_list];
FA_cum_censoring = [zeros(size(FA_RTs_list)) ; ones(size(signal_starts_list))];
[P_FA,t_FA] = ecdf(FA_cum_data,'censoring',FA_cum_censoring);
t_FA(1)=[];
P_FA(1)=[];

%plot FA rate cumulative probability distribution
figure; plot(t_FA,P_FA)
title('KM-based False alarm cumulative responses probability')
xlim([0 14])

%plot the cumulative hazard function
figure; plot(t_FA,-log(1-P_FA))
title('KM-based False alarm cumulative hazard function')
xlim([0 14])

%plot the (noisy!) hazard function
%figure; plot(t_FA(2:end),-diff(log(1-P_FA))./diff(t_FA))

%% build (binned) hit and FA rate cumulative curves
%figure; histogram(signal_starts_list)
Hit_RTs_list=Hit_RTs(~isnan(Hit_RTs));

sig_start_bin_edges=0;
sig_bin_start_means=[];
P_Hit_binned=[];
t_Hit_binned=[];
P_FA_binned=[];
d_prime_binned=[];
c_bias_binned=[];
% figure;
for i=1:signal_start_bins-1
    sig_start_bin_edges(i+1) = prctile(signal_starts_list,i*100/signal_start_bins);
    hits_temp=find(sig_start_bin_edges(i)<Hit_starts & Hit_starts<=sig_start_bin_edges(i+1));
    Hit_RTs_temp=Hit_RTs_list(hits_temp);
    Hit_starts_temp=Hit_starts(hits_temp);
    misses_temp=find(sig_start_bin_edges(i)<Miss_starts & Miss_starts<=sig_start_bin_edges(i+1));
    Miss_starts_temp=Miss_starts(misses_temp);

    Hit_cum_data_temp = [Hit_RTs_temp ; sig_dur.*ones(size(Miss_starts_temp))];
    Hit_cum_censoring_temp = [zeros(size(Hit_RTs_temp)) ; ones(size(Miss_starts_temp))];
    [P_Hit_binned{i},t_Hit_binned{i}] = ecdf(Hit_cum_data_temp,'censoring',Hit_cum_censoring_temp);
    
    sig_bin_start_means(i)=mean([Hit_starts_temp ; Miss_starts_temp]);
    P_FA_binned{i} = interp1(t_FA-sig_bin_start_means(i),P_FA,t_Hit_binned{i},'linear');
    P_FA_binned{i}=(P_FA_binned{i}-min(P_FA_binned{i}))./(1-min(P_FA_binned{i}));
    P_Hit_binned{i}(1)=[];
    t_Hit_binned{i}(1)=[];
    P_FA_binned{i}(1)=[];
    d_prime_binned{i}=norminv(P_Hit_binned{i})-norminv(P_FA_binned{i});
    c_bias_binned{i}=-0.5*(norminv(P_Hit_binned{i})+norminv(P_FA_binned{i}));
    hold on
    % plot(d_prime_binned{i})
end

%plot time-varying ROC curves
% figure;
% for i=1:signal_start_bins-1
%     plot(P_FA_binned{i},P_Hit_binned{i})
%     hold on
%     xlim([0 1])
%     ylim([0 1])
% end

c_bias_ends=[]; d_prime_ends=[]; P_FA_ends=[]; P_Hit_ends=[];
for i=1:length(d_prime_binned)
    c_bias_ends(i)=c_bias_binned{i}(end)
    d_prime_ends(i)=d_prime_binned{i}(end);
    P_FA_ends(i)=P_FA_binned{i}(end);
    P_Hit_ends(i)=P_Hit_binned{i}(end);
end

figure; 
scatter(P_FA_ends,P_Hit_ends)
xlim([0 1])
ylim([0 1])
title('start time-binned ROC plot')

figure; plot(sig_bin_start_means,d_prime_ends)
ylim([0 1.5])
title('start time-binned discriminability')

figure; plot(sig_bin_start_means,c_bias_ends)
ylim([-1 1])
title('start time-binned bias')
%%
