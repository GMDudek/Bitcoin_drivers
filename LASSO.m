%Output: One-day-ahead forecasts of lnRV.
clear all
T=readtable('Bitcoin_drivers_data.xlsx','Sheet','Data','Range','B2:BL1479')
dane=table2array(T); %Input data
liczba_zm=numel(dane(1,:))-1;
ileprognoz=848; %testing sample
n=numel(dane(:,1));
ilewalidacja=100; %validation sample
iledanych=n-ileprognoz-ilewalidacja; %training sample
 

for i=1:ileprognoz
    i
    clear MSE;
    clear Lambdy;
    
    if (i==1 | 4==mod(i,25)) %every fifth Friday
        [wynik,FitInfo]=lasso(dane(i:iledanych+i-1+ilewalidacja,2:(liczba_zm+1)),dane(i:iledanych+i-1+ilewalidacja,1),'CV',10);
        idxLambda = FitInfo.IndexMinMSE;
        Lambda_opt(1)=FitInfo.Lambda(idxLambda);
    end
    %forecasting
    
    [wynik,FitInfo]=lasso(dane(i:iledanych+i-1+ilewalidacja,2:(liczba_zm+1)),dane(i:iledanych+i-1+ilewalidacja,1),'Lambda',Lambda_opt(1));
    wspolczynniki(:,i) = wynik;
    coef0 = FitInfo.Intercept;
    XTest=dane(iledanych+i+ilewalidacja,2:(liczba_zm+1));
    YFit(i,1) = (XTest*wspolczynniki(:,i) + coef0);

    %the same for other horizons
    if 4==mod(i,5) %Friday     
         for horyzont=2:5
            if 4==mod(i,25) %every fifth Friday
                [wynik,FitInfo]=lasso(dane(i:iledanych+i-1+ilewalidacja-horyzont+1,2:(liczba_zm+1)),dane(i+horyzont-1:iledanych+i-1+ilewalidacja,1),'CV',10);
                idxLambda = FitInfo.IndexMinMSE;
                Lambda_opt(horyzont)=FitInfo.Lambda(idxLambda);
            end
            %forecasting:
            
            [wynik,FitInfo]=lasso(dane(i:iledanych+i-1+ilewalidacja-horyzont+1,2:(liczba_zm+1)),dane(i+horyzont-1:iledanych+i-1+ilewalidacja,1),'Lambda',Lambda_opt(horyzont));
            wspolczynniki_hor(:,i)=wynik;
            coef0 = FitInfo.Intercept;
            XTest=dane(iledanych+i+ilewalidacja,2:(liczba_zm+1));
            YFit(i,horyzont) = (XTest*wspolczynniki_hor(:,i) + coef0);
        end
    end
end
save forecasts.dat YFit -ASCII;
save coefficients.dat wspolczynniki -ASCII;
koniec


