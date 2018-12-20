%Read in ROIs 
load dbt_savoi;  %3d signal-absent (SA) ROIs
load dbt_spvoi;  %3d signal-present (SP) ROIs

nsa = size(savoi,4); %number of SA cases
nsp = size(spvoi,4); %number of SP cases
    
ntrain = 40; 
id_sa_tr=[1:ntrain];
id_sp_tr=[1:ntrain];
id_sa_test=[ntrain+1:nsa];
id_sp_test=[ntrain+1:nsp];

%run CHO by setting the last parameter to 0
[snr1, t_sp1, t_sa1, chimg1,tplimg1,meanSP1,meanSA1,meanSig1, k_ch1]=conv_LG_CHO_3d(savoi(:,:,:,id_sa_tr), spvoi(:,:,:,id_sp_tr), savoi(:,:,:,id_sa_test), spvoi(:,:,:,id_sp_test),25,5,0);
%run convolutional CHO by setting the last parameter to 1
[snr2, t_sp2, t_sa2, chimg2,tplimg2,meanSP2,meanSA2,meanSig2, k_ch2]=conv_LG_CHO_3d(savoi(:,:,:,id_sa_tr), spvoi(:,:,:,id_sp_tr), savoi(:,:,:,id_sa_test), spvoi(:,:,:,id_sp_test),25,5,1);                     
  

disp(snr1);
disp(snr2);




