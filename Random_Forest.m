%Random Forest implementation of the paper:
%Fiszeder P., Orzeszko W., Pietrzyk R., Dudek G., Identification of Bitcoin Volatility Drivers Using Statistical and Machine Learning Methods

T = readtable('Bitcoin_drivers_data.xlsx','Sheet','Data','Range','B2:BL1479'); %variables
vv = readcell('Bitcoin_drivers_data.xlsx','Sheet','Data','Range','B1:BL1'); %variable names
DD = readtable('Bitcoin_drivers_data.xlsx','Sheet','Data','Range','A2:A1479'); %dates
TD = readtable('Bitcoin_drivers_data.xlsx','Sheet','Data','Range','BM2:BM1479'); %day type (1 - Monday)

ntrees = 100; %number of trees
Dts = "01-Jan-2020"; %first test sample
otr = 630; %rolling window size
ots = 848; %test period size
options = statset('UseParallel',true);

r_out = 1; %replacement of outliers (1), without replacement (0)
d_std = 1; %standardization (1), without standardization (0) 

mms = [1, 3, 28, 53, 78, 103, 128, 153, 178, 203, 228, 253, 278, 303, 328, 353, ...
    378, 403, 428, 453, 478, 503, 528, 553, 578, 603, 628, 653, 678, 703, 728, ...
    753, 778, 803, 828, nan]; %splitting the test period into sub-periods; in each sub-period there is different MinLeafSize

%MinLeafSize for sub-periods and horizons (result of grid search)
Minleaf = [3, 2, 6, 4, 7, 7, 2, 9, 13, 5, 9, 8, 16, 11, 10, 16, 9, 11, 8, 8, 6, 3, 10, 9, 14, 13, 5, 4, 11, 5, 10, 10, 5, 5, 7
NaN, 18, 18, 10, 15, 16, 18, 8, 9, 12, 20, 5, 8, 6, 14, 15, 18, 8, 9, 10, 17, 9, 6, 10, 6, 20, 17, 10, 9, 12, 18, 18, 20, 16, 8
NaN, 3, 5, 6, 10, 6, 7, 7, 7, 5, 7, 11, 10, 3, 2, 9, 18, 8, 7, 10, 9, 11, 7, 13, 9, 15, 11, 9, 4, 11, 9, 12, 13, 17, 17
NaN, 16, 14, 18, 19, 6, 4, 3, 16, 6, 12, 9, 6, 5, 13, 13, 9, 13, 12, 16, 6, 17, 9, 3, 3, 8, 14, 7, 14, 13, 13, 3, 19, 10, 7
NaN, 3, 5, 12, 6, 5, 7, 5, 7, 6, 11, 4, 4, 4, 2, 11, 10, 10, 15, 14, 13, 18, 14, 12, 17, 11, 8, 11, 7, 13, 4, 17, 14, 9, 5]';

%input and output data
dane=table2array(T);
xx = dane(:,2:end);

[lw,lk] = size(xx);

yy = [dane(:,1), [dane(2:end,1); nan(1,1)], [dane(3:end,1); nan(2,1)], [dane(4:end,1); nan(3,1)], [dane(5:end,1); nan(4,1)]];  

DD=table2array(DD);
TD=table2array(TD);

Nts = find(DD==Dts); %first test sample

%% Trening
ysp=nan(ots,5);
es=ysp; maes=ysp; mapes=ysp; mses=ysp; qlikes=ysp; rse=ysp;
imp1=[]; imp2=[];

mm=1;
kk=1;
for ii=Nts:lw %iterations over test samples; for each sample, we train a new model

    if mm==mms(kk) %reading MinLeafSize for a given sub-period
        minleaf=nan(1,5);
        for h=1:5
            minleaf(h)=Minleaf(kk,h);
            if mm==1 break; end
        end
        kk=kk+1;
    end

    tdi=ii-otr:ii-1; %rolling window
    td=TD(ii); %test day type

    for h=1:5 %iterations over horizons

        if (h>1)&&(td~=1) %generate forecasts for horizons h>1 only when td==1
            continue;
        end

        xr = xx(tdi,:);
        yr = yy(tdi,h);
        xs = xx(ii,:);
        ys = yy(ii,h);

        %replacement of outliers
        if r_out==1
            [xo,ix,L,U]=filloutliers([xr; xs],"clip","quartiles");
            xro = xo(1:end-1,:);
            xso = xo(end,:);
        else
            xro = xr;
            xso = xs;
        end

        %standardization 
        if d_std==1
            [xd, mu, sigma] = zscore([xro;xso]);
            xro = xd(1:end-1,:);
            xso = xd(end,:);
        else
            xro = xr;
            xso = xs;
        end

        %preprocessing only exogenous variables
        xr = [xr(:,1:3) xro(:,4:end)];
        xs = [xs(1:3) xso(4:end)];
        
        %RF training
        B = TreeBagger(ntrees,xr,yr,'method','regression','MinLeafSize',minleaf(h),'OOBPrediction','on','OOBPredictorImportance','on','Options',options);
        imp1(h).i(mm,:) = B.OOBPermutedPredictorDeltaError; %importance estimation of predictors
        imp2(h).i(mm,:) = B.DeltaCriterionDecisionSplit; %importance estimation of predictors

        %forecast for test sample
        qq = B.predict(xs);
        ys=exp(ys);
        qq=exp(qq);

        ysp(mm,h)=qq;
        es(mm,h)=ys-qq;
        maes(mm,h) = maef(ys,qq);
        mapes(mm,h) = mapef(ys,qq);
        mses(mm,h) = msef(ys,qq);
        qlikes(mm,h) = mean(qlikef(ys,qq));
    end
    mm=mm+1;
end

MAE_h=nanmean(maes);
MSE_h=nanmean(mses);

%Predictor importance chart
figure
bar(mean(imp1(1).i));
ylabel('Predictor importance estimates');
xlabel('Predictor');
title(['OOBPermutedPredictorDeltaError, h=1']);
grid on;
xticks(1:62);
xticklabels(vv(2:end));

%% error metrics

function b = maef(y,p)
    b=abs(y-p);
end

function b = msef(y,p)
    b=(y-p).^2;
end

function b = mapef(y,p)
    b=abs((y-p)./y)*100;
end

function b = r2f(y,p)
    b=1-sum((y-p).^2)/sum((y-mean(y)).^2);
end
    
function b = qlikef(y,p)
    b=log(p) + y./p;
end