
%[snr, t_sp, t_sa, chimg,tplimg,meanSP,meanSA,meanSig, k_ch]=conv_LG_CHO_3d(trimg_sa, trimg_sp, testimg_sa, testimg_sp, ch_width, nch, b_conv)
%Filtered/convolutional Channels CHO, based on the paper:
%Diaz et al, IEEE-tmi-34(7), 2015, "Derivation of an observer model adapted
%to irregular signals based on covolution channels"
%
%3D MO for detecting calc clusters. The LG channel function is convolved with
%each slice of the mean signal to obtain the 3D channel function.
%Inputs
%   testimg_sa: the test set of signal-absent, a stack of 2D array;
%   testimg_sp: the test set of signal-present;
%   trimg_sa: the training set of signal-absent;
%   trimg_sp: the training set of signal-present;
%   ch_width: channel width parameter;
%   nch: number of channels to be used;
%   b_conv: 1 or 0 to indicate whether to apply a convolution of the signal
%   to the LG channels. Default is 1.
%   
%Outputs
%   snr: the detectibility SNR
%   t_sp: t-scores of signal-present cases
%   t_sa: t-scores of signal-absent cases
%   chimg: The channel matrix, Nx x Ny x Nz x nch, where Nx x Ny x Nz is the image size
%   tplimg: the model observer template, Nx x Ny x Nz
%   meanSP: the average signal-repsent image
%   meanSA: the average signal-absent image
%   meanSig: the average signal image
%   k_ch: data covariance matrix
%
%R Zeng, 8/2018, FDA/CDRH/OSEL/DIDSR
%==========================================================================
%           Legal Disclaimer
%This software and documentation (the "Software") were developed at the 
%Food and Drug Administration (FDA) by employees of the Federal Government
%in the course of their official duties. Pursuant to Title 17, Section 105 
%of the United States Code, this work is not subject to copyright 
%protection and is in the public domain. Permission is hereby granted, 
%free of charge, to any person obtaining a copy of the Software, to deal 
%in the Software without restriction, including without limitation the 
%rights to use, copy, modify, merge, publish, distribute, sublicense, or 
%sell copies of the Software or derivatives, and to permit persons to whom 
%the Software is furnished to do so. FDA assumes no responsibility 
%whatsoever for use by other parties of the Software, its source code, 
%documentation or compiled executables, and makes no guarantees, expressed 
%or implied, about its quality, reliability, or any other characteristic. 
%Further, use of this code in no way implies endorsement by the FDA or 
%confers any advantage in regulatory decisions. Although this software can 
%be redistributed and/or modified freely, we ask that any derivative works 
%bear some notice that they are derived from it, and any modified versions 
%bear some notice that they have been modified.
%==========================================================================

function [snr, t_sp, t_sa, chimg,tplimg,meanSP,meanSA,meanSig, k_ch]=conv_LG_CHO_3d(trimg_sa, trimg_sp,testimg_sa, testimg_sp,  ch_width, nch, b_conv)

if(nargin<7)
    b_conv=1;
end

[nx, ny, nz, nte_sa]=size(testimg_sa);

%Ensure the images all having the same x,y sizes. 
[nx1, ny1, nz1, nte_sp]=size(testimg_sp);
if(nx1~=nx | ny1~=ny | nz1~=nz)
    error('Image size does not match! Exit.');
end
[nx1, ny1, nz1, ntr_sa]=size(trimg_sa);
if(nx1~=nx | ny1~=ny | nz1~=nz)
    error('Image size does not match! Exit.');
end
[nx1, ny1, nz1, ntr_sp]=size(trimg_sp);
if(nx1~=nx | ny1~=ny | nz1~=nz)
    error('Image size does not match! Exit.');
end

%LG channels
xi=[0:nx-1]-(nx-1)/2;
yi=[0:ny-1]-(ny-1)/2;
[xxi,yyi]=meshgrid(xi,yi);
r=sqrt(xxi.^2+yyi.^2);
u=laguerre_gaussian_2d(r,nch-1,ch_width);
ch=reshape(u,nx*ny,size(u,3)); %if not applying the following filtering to the channels


%Create signal convolved channels
sig_mean=mean(trimg_sp,4)-mean(trimg_sa,4);
ch_sig = zeros(nx,ny,nz,nch);
if(b_conv)  
    for ich=1:nch
        for j=1:nz
            covsig=(ifft2(abs(fft2(u(:,:,ich))).^2 .* fft2(sig_mean(:,:,j))))/nx/ny;
            ch_sig(:,:,j,ich) = covsig/sqrt(sum(sum(covsig.^2))); %Normalize the energy of the channel function. but this does not affect the detectability at all.
        end
    end
else

    for ich=1:nch
        for j=1:nz
            ch_sig(:,:,j,ich) = u(:,:,ich); %Normalize the energy of the channel function. but this does not affect the detectability at all.
        end
    end    
        
end
ch=reshape(ch_sig, nx*ny*nz, nch);


%Training MO
nxyz=nx*ny*nz;
tr_sa_ch = zeros(nch, ntr_sa);
tr_sp_ch = zeros(nch, ntr_sp);
for i=1:ntr_sa
    tr_sa_ch(:,i) = reshape(trimg_sa(:,:,:,i), 1,nxyz)*ch;
end
for i=1:ntr_sp
    tr_sp_ch(:,i) = reshape(trimg_sp(:,:,:,i), 1,nxyz)*ch;
end
s_ch = mean(tr_sp_ch,2) - mean(tr_sa_ch,2);
k_sa = cov(tr_sa_ch');
k_sp = cov(tr_sp_ch');
k = (k_sa+k_sp)/2;
w = s_ch(:)'*pinv(k); %this is the hotelling template

%detection (testing)
for i=1:nte_sa
    te_sa_ch(:,i) = reshape(testimg_sa(:,:,:,i), 1, nxyz)*ch;
end
for i=1:nte_sp
    te_sp_ch(:,i) = reshape(testimg_sp(:,:,:,i), 1, nxyz)*ch;
end
t_sa=w(:)'*te_sa_ch;
t_sp=w(:)'*te_sp_ch;

snr = (mean(t_sp)-mean(t_sa))/sqrt((std(t_sp)^2+std(t_sa)^2)/2);

%Optional outputs
tplimg=(reshape(w*ch',nx,ny,nz)); % MO template
chimg=reshape(ch,nx,ny,nz,nch); %Channels
meanSP=mean(trimg_sp,4);
meanSA=mean(trimg_sa,4);
meanSig=sig_mean;
k_ch=k;
