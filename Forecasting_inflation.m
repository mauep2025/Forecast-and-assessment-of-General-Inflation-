%////////////////////////////////////////////////////////////////%
%//////////////Forecasting Inflation (Assessment 2)//////////////%
%////////////////////////////////////////////////////////////////%
%---------------------------------------------------
%--Data Transformation into quarterly frequency-----
%---------------------------------------------------
clc
clear 
tic 
dataname='UK_data.xlsx';
data.M=xlsread(dataname, 'Monthly'); %Real GDP
data.Q=xlsread(dataname, 'Quarterly');  % % CPI, IP commodity prices, FTSE index and unemployment rate
disp('------------------------------------------------------------------');
disp('             Time Series Models (Forecasting Inflation)           ');
disp('------------------------------------------------------------------');
[t_1,~]=size(data.Q);
for ii= 1 : t_1
    data.Q(ii,2)=mean(data.M(1+3*(ii-1):3*(ii),1))
    data.Q(ii,3)=mean(data.M(1+3*(ii-1):3*(ii),2)) 
    data.Q(ii,4)=mean(data.M(1+3*(ii-1):3*(ii),3))
    data.Q(ii,5)=mean(data.M(1+3*(ii-1):3*(ii),4))
    data.Q(ii,6)=mean(data.M(1+3*(ii-1):3*(ii),5))
end    

[T,~]=size(data.Q);
target_1=100*((data.Q(2:T,2)./data.Q(1:T-1,2))-1); % t/t-4 inflation

%GDP=100*((data.Q(5:T,1)./data.Q(1:T-4,1))-1); % annual GDP growth
%IP=100*((data.Q(5:T,3)./data.Q(1:T-4,3))-1); % annual IP growth
%COM=100*((data.Q(5:T,4)./data.Q(1:T-4,4))-1); % annual commodity price growth
%FTSE=100*((data.Q(5:T,5)./data.Q(1:T-4,5))-1); % annual FTSE growth
%urate=data.Q(5:T,6); % unemployment rate - NB in levels here
%urate=data.Q(2:T,3); % unemployment rate - NB in levels here

%----Quaterly Growth Rate |Order= % GDP, CPI, IP commodity prices, FTSE index and unemployment rate

for i =1: size(data.Q,2)
        for ii=2 : size(data.Q,1)
        data_gq(ii-1,i)=((data.Q(ii,i)/data.Q(ii-1,i))-1)*100;
        end
end       
 %data_gq(:,6)=data.Q(2:T,6) %It can be use in order to use the level of
 %the unemployment rate 
% Ploting Variables 
%-------Data Level 

t_1=(1980.083:0.083:2014.667)'
figure;
subplot(3,2,1)
plot(t_1,data.M(:,1), 'LineWidth', 1.5), title('CPI Index', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,2,2)
plot(t_1,data.M(:,2), 'LineWidth', 1.5), title('Industrial Production', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,2,3)
plot(t_1,data.M(:,3), 'LineWidth', 1.5), title('Commodity Prices Index', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,2,4)
plot(t_1,data.M(:,4), 'LineWidth', 1.5), title('FTSE Index', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,2,5)
plot(t_1,data.M(:,5), 'LineWidth', 1.5), title('Unemployment Rate', 'FontSize', 11), ylabel('Rate'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
sgtitle('Graph: Macro-Finance Variable Cast')

figure
t_2=(1980.25:.25:2014.75)'
data_q=data.Q(:,1)
plot(t_2,data_q,'LineWidth', 1.5), title('Graph: UK GDP Output')

%--Ploting Transformation (First Difference) Data % GDP, CPI, IP commodity prices, FTSE index and unemployment rate

t_3=(1980.25:.25:2014.50)';
figure;
subplot(3,2,1)
plot(t_3,data_gq(:,1), 'LineWidth', 1.5), title('GDP Output', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,2,2)
plot(t_3,data_gq(:,2), 'LineWidth', 1.5), title('CPI', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,2,3)
plot(t_3,data_gq(:,3), 'LineWidth', 1.5), title('Industrial Production', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,2,4)
plot(t_3,data_gq(:,4), 'LineWidth', 1.5), title('Commodity Prices', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,2,5)
plot(t_3,data_gq(:,5), 'LineWidth', 1.5), title('FTSE Index', 'FontSize', 11), ylabel('Rate'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,2,6)
plot(t_3,data_gq(:,6), 'LineWidth', 1.5), title('Unemployment Rate', 'FontSize', 11), ylabel('Rate'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
sgtitle('Graph: UK Macro-Finance Variable Cast (Quarterly Growth Rate)')

%%%-------------------------- Autocorrelation 
figure;
autocorr(data_gq(:,1)), legend('GDP');

% Autocorrelaton function (ACF)
% Identify series with serial correlation
% Determine whether an AR model is apprpriate
% Identify significant MA lags for model identification

%%%%%%%%%-- Multiple graphs 

% Autocorrelaton function (ACF)
% Identify series with serial correlation
% Determine whether an AR model is apprpriate
% Identify significant MA lags for model identification
figure;
subplot(3,2,1)
autocorr(data_gq(:,1)), legend('GDP Output');
subplot(3,2,2)
autocorr(data_gq(:,2)), legend('CPI');
subplot(3,2,3)
autocorr(data_gq(:,3)), legend('Industrial Production');
subplot(3,2,4)
autocorr(data_gq(:,4)), legend('Commodity Prices');
subplot(3,2,5)
autocorr(data_gq(:,5)), legend('FTSE Index');
subplot(3,2,6)
autocorr(data_gq(:,5)), legend('Uneployment Rate');
sgtitle('Graph: UK Macro-Finance Variable Cast (Quarterly Growth Rate)')

% Partial ACF (PACF)
% Identify Series with Serial Correlation
% Determine whether an MA model is appropriate
% Identify significant AR lags for model identification.

figure;
subplot(3,2,1)
parcorr(data_gq(:,1)), legend('GDP Output');
subplot(3,2,2)
parcorr(data_gq(:,2)), legend('CPI');
subplot(3,2,3)
parcorr(data_gq(:,3)), legend('Industrial Production');
subplot(3,2,4)
parcorr(data_gq(:,4)), legend('Commodity Prices');
subplot(3,2,5)
parcorr(data_gq(:,5)), legend('FTSE Index');
subplot(3,2,6)
parcorr(data_gq(:,5)), legend('Uneployment Rate');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%--- Producing forecast with ARIMA model for Inflation (CPI)----%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

inflation=data_gq(:,2)

ixd=all(~ismissing(inflation),2);%Removing the all the missing values 
%Index vector that will divides the response data into three periods:
%presample, estimation and forecast. 
% "presample" (period) is use to give hitorical data to the model
%The "estimation" (period) is in sample
% the "forecast" period is out of sample (""""backtesting/ or / EValuation period"""") 

idxPre=10:138;% Meaning that his goes from 1982q4 to 2014q3
TTT=ceil(.80*size(inflation,1)); %% the "forecast" period is the last 20% of rows of inflation
idxEst=10:TTT;   % Estimation perio, will go from the 16th row to the last information in the 90% of the  row inflation
% NOTE: here is important to say that the forecaster can change at its will
idxF=(TTT+1):size(inflation,1); %  "forecat" period 
fh=numel(idxF);
inflation=data_gq(:,2) %%%% Here I separete the inflation variable from others

AIC_spec=zeros(7,7);
porder=[0 1 2 3 4 5 6]';
qorder=[0 1 2 3 4 5 6]';

for i = 1:size(porder,1);
  p=porder(i);
    for j=1:size(qorder,1);
        q=qorder(j);
        model=arima(p,0,q);% p=AR order, d=order of difference, q=MA order
        [fit,VarCov,logL,info]=estimate(model,inflation(idxEst,:), 'Y0', inflation(idxPre,:) );
        s2=fit.Variance;
        T=size(inflation,1);
        %LL = -(T/2)*log(det(e'*e./T))-(T*m/2)*(1+log(2*pi));
        %sbc(nmodel,1) = LL - (p+q)*log(T);
        %BIC_spec(i,j)=T*log(s2)+ (p+q)*log(T);
        AIC_spec(i,j)=T*log(s2)+  2*(p+q);                 
    end            
end 

%%%%%%%%%% FORECASTING Using AIC ARIMA minimising model %%%%%%%
% Select model with minimum AIC
[num idx] = min(AIC_spec(:));
[xaic yaic] = ind2sub(size(AIC_spec),idx);
% Estimate using preferred ARIMA model  
p=xaic-1;q=yaic-1;
model=arima(p,0,q);% p=AR order, d=order of difference, q=MA order
[fit,VarCov,logL,info]=estimate(model,inflation(idxEst,:), 'Y0',inflation(idxPre,:));
[res, ~, logL]=infer(fit,inflation(idxEst,:), 'Y0',inflation(idxPre,:));
res_arima_inflation=res;
% Forecast h steps ahead 
h=fh;
[for_infl,for_infl_mse,V]=forecast(fit,h,'Y0',inflation(idxEst,:));
aic_for=for_infl;
mse_aic=for_infl_mse;

%%% In Matlab the comand "forecast" generates MMSE(Minimum mean square error)  forecast recursively
%%%% https://uk.mathworks.com/help/econ/mmse-forecasting-for-arima-models.html

%%%%-----------Ploting Forecast and data (Backtesting) ------

% Note: It is important to note how to change the date, in order to get the
% graph. For example  

TTF = (1980.25:.25:2014.75)';% this variable will alows to show only one part of the in-sample information

ann= {'Note: Estimation period:1982q2-2008q1';'         Presample period: 1982q2-2014q4';'         Forecast period: 2008q1-2014q3'}
dim=[.13 .16 .1 .1]
   figure; 
    h1= plot(TTF(100:138),inflation(100:138), 'LineWidth', 1.6); %Here we can control the sample of the graph, remember that TFF and inflation will have the same leght, in this case this index will help us to select a sample for the porpuse of the graph
    hold on; 
    h2=plot(TTF(idxF), aic_for, 'LineWidth', 1 );
    h3=plot(TTF(idxF),aic_for+1.96*sqrt(mse_aic), 'k--', 'LineWidth', 1);
    plot(TTF(idxF),aic_for-1.96*sqrt(mse_aic), 'k--', 'LineWidth', 1);
    title(['Graph: Unconditional Recursive Forecast of UK Inflation (AIC selected ARIMA)']), xlabel({''});
    ylim([-1 2.5]) % Here you can change the numbers of the "y" axes
    h=gca;
    fill([TTF(idxF(1)) h.XLim([2 2]) TTF(idxF(1))], h.YLim([1 1 2 2]), 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    legend([h1 h2 h3], 'Observed Figure', 'Forecast', '95% Forecast Interval', 'Location', 'best')
    annotation('textbox',dim,'String',ann,'EdgeColor', 'none' );
    set(gcf, 'color',  'w')
    legend('boxoff');
    hold off;

%Ploting the Real Forecast Observations (Out-of-Sample) from 2014q4 to
%2016q3
idx_arima=all(~ismissing(inflation ),2);
inflation_1=inflation(idx_arima,:);
[YPred_arima_4, YMSE_arima_4]=forecast(fit,8, 'Y0' , inflation(idxPre));  %%%%%%%% Change 

% the "YPred_arima will save the forecast values of quarterly growth
% inflation

YFirst_arima=data_gq(131:138,2);
YSecond_arima=inflation(131:138);
EndPt_1= YFirst_arima(end,:);
EndPt_2= YSecond_arima(end,:);
EndPt_3= YSecond_arima(end,:);

%EndPt_1(:,1:1)=log(EndPt_1);
%YPred_arima=YPred_arima/100; %Rescale percentage 
%Prepare to cumsum
YPred_arima_4=[EndPt_2; YPred_arima_4];
%YPred_arima(:,1:1)=cumsum(YPred_arima(:,1:1));
%YPred_arima(:,1:1)=exp(YPred_arima(:,1:1));

TTFF2= (2014.50:0.25:2016.50)' % the number of forecast observation that I choose where 8 quarter, therefore this sample goes from 2014q4 to 2016q3
TTTF3= (2012.75:0.25:2014.50)' % the sample must coincide with the variable "YFirst"
TTFF4= (2014.50:0.25:2016.25)' %Forecast out of the sample

%     Ploting only the forecast ( ARIMA) 
ann_2= {'Note: Estimation period:1982q2-2008q1';'         Presample period: 1982q2-2014q4';'         Forecast period: 2014q4-2016q3'}
dim_2=[.51 .7 .1 .1]

    figure
for j=1:1
    subplot(1,1,j)
    plot(TTFF2,YPred_arima_4(:,j),'--b')
    set(gcf, 'color',  'w')
    hold on 
    plot(TTTF3,YSecond_arima(:,j),'k')
    title('Graph: Recursive Forecast Inflation Out-of-Sample (Quarterly growth rate) ')
    annotation('textbox',dim_2,'String',ann_2,'EdgeColor', 'none' );
    h=gca
    fill([TTFF2(1) h.XLim([2 2]) TTFF2(1)], h.YLim([1 1 2 2]), 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    set(gcf, 'color',  'w')    
    hold off
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%--- Producing forecast with VAR model for Inflation (CPI)----%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%---------------------------------------------------------------------------
%%%% ----Unconditional and Conditional VAR={inflation, unemployment rate} --
%---------------------------------------------------------------------------
%%% Select and Fit the Model
 
A1=[inflation data_gq(:,6)];
u_rate=data_gq(:,6); 
numseries=2;
seriesnam_1={'Inflation',  'Unemployment Rate'  };
VARf_c=varm(numseries,3);
VARf_c.SeriesNames=seriesnam_1;
[EstMdl1_VARf_c,EstSE1_VARf_c,logL1_VARf_c,E1_VARf_c]= estimate(VARf_c,A1(idxEst,:), 'Y0',A1(idxPre,:));
%%Check Model Adequacy
EstMdl1_VARf_c.Description
results1_VARf_c=summarize(EstMdl1_VARf_c);
np1_VARf_c=results1_VARf_c.NumEstimatedParameters;

%%Forecast predictions. Supposing that a negative supply idiosyncratic shock happens
%at the begining of 2008 affecting the unmeployment rate (first), let say
%an increase of in the unemployment rate of 5%
[FY_VARf_u,Forecastmse_varu] = forecast(EstMdl1_VARf_c,fh,A1);
YF= [nan(27,1) repmat(u_rate(end),27,1)  ]; %this variable will create nan in the firts colum and will re-write the last 27 rows of unemployment increasing in 5%
[F_VARf_c,Forecastmse_varc] = forecast(EstMdl1_VARf_c,fh,A1, 'YF', YF );

EndPt_3=inflation(111,:);
%Comparing both forecast in a graph
YPred_varu=[EndPt_3; FY_VARf_u(:,1)];
YPred_varc=[EndPt_3; F_VARf_c(:,1)];
TTT_1=(2008:.25:2014.75)'
idxF_1=(TTT):size(inflation,1)'

ann_3={'Increase in 5% of the Unemployment Rate'};
figure; 
    h1= plot(TTF(100:TTT),inflation(100:TTT), 'LineWidth', 1.6); %Here we can control the sample of the graph, remember that TFF and inflation will have the same leght, in this case this index will help us to select a sample for the porpuse of the graph
    hold on; 
    h2=plot(TTF(idxF_1),YPred_varu, 'LineWidth', 1 );
    h3=plot(TTF(idxF_1),YPred_varc, 'k--', 'LineWidth', 1);
    title(['Graph: Conditional Recursive Forecast of UK Inflation (VAR)']), xlabel({''});
    ylim([-1 2.5]) % Here you can change the numbers of the "y" axes
    h=gca;
    fill([TTF(idxF_1(1)) h.XLim([2 2]) TTF(idxF_1(1))], h.YLim([1 1 2 2]), 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    legend([h1 h2 h3], 'Observed Figure', 'Unconditional Forecast', 'Conditional Forecast', 'Location', 'best')
    annotation('textbox',dim,'String',ann_3,'EdgeColor', 'none' );
    set(gcf, 'color',  'w')
    legend('boxoff');
    hold off;
    
    %The next figure is whether you will like to compare the forecasting
    %with the real figure 
    figure; 
    h1= plot(TTF(100:T),inflation(100:T), 'LineWidth', 1.6); %Here we can control the sample of the graph, remember that TFF and inflation will have the same leght, in this case this index will help us to select a sample for the porpuse of the graph
    hold on; 
    h2=plot(TTF(idxF_1), YPred_varu, 'LineWidth', 1 );
    h3=plot(TTF(idxF_1),YPred_varc, 'k--', 'LineWidth', 1);
    title(['Graph: Conditional Recursive Forecast of UK Inflation (VAR)']), xlabel({''});
    %ylim([-1 2.5]) % Here you can change the numbers of the "y" axes
    h=gca;
    fill([TTF(idxF_1(1)) h.XLim([2 2]) TTF(idxF_1(1))], h.YLim([1 1 2 2]), 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    legend([h1 h2 h3], 'Observed Figure (Inflation)', 'Unconditional Forecast', 'Conditional Forecast', 'Location', 'best')
    annotation('textbox',dim,'String',ann_3,'EdgeColor', 'none' );
    set(gcf, 'color',  'w')
    legend('boxoff');
    hold off;

%Estimating the 95% forecast interval for the best fitting model

extractMSE=@(x)diag(x)';
MSE_varc=cellfun(extractMSE, Forecastmse_varc, 'UniformOutput', false);
SE_varc=sqrt(cell2mat(MSE_varc));

%%%%%%%%%%%%%%%%%%%%%   
%%%%%%%%%%%%%%%%%%%%%% 
YFI_varc= zeros(fh,EstMdl1_VARf_c.NumSeries,2);
YFI_varc(:,:,1) = F_VARf_c - 2*SE_varc; %Lower Band
YFI_varc(:,:,2) = F_VARf_c + 2*SE_varc; %Upper Band
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%
 dim_2=[1 1 .2 .1] %This comand is for control the location of the annotation in the figure
 figure; % here the forecast for the unemployment is flat due we change the figures of the unmeployment rate 
 for j_2=1: numseries
    subplot(numseries,1,j_2);
    h1= plot(TTF(100:T),A1(100:T,j_2), 'LineWidth', 1.6); %Here we can control the sample of the graph, remember that TFF and inflation will have the same leght, in this case this index will help us to select a sample for the porpuse of the graph
    hold on; 
    h2=plot(TTF(idxF), F_VARf_c(:,j_2), 'LineWidth', 1 );
    h3=plot(TTF(idxF),YFI_varc(:,j_2,1), 'k--', 'LineWidth', 1);
    plot(TTF(idxF),YFI_varc(:,j_2,2), 'k--', 'LineWidth', 1);
    title(VARf_c.SeriesNames{j_2});
    h=gca;
    fill([TTF(idxF(1)) h.XLim([2 2]) TTF(idxF(1))], h.YLim([1 1 2 2]), 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    legend([h1 h2 h3], 'Observed Figure', 'Forecast', '95% Forecast Interval', 'Location', 'best')
    annotation('textbox',dim,'String',ann,'EdgeColor', 'none' );
    sgtitle('Graph: Unconditional Recursive Forecast for UK (VAR1)' )
    set(gcf, 'color',  'w')
    legend('boxoff');
    hold off;
 end
% Computing the impulse response funtions for the analysis with bootsstrap
% confidence intervals 
n=size(E1_VARf_c,1);
rng(1);
[Response,Lower,Upper]=irf(EstMdl1_VARf_c,"E",E1_VARf_c,"NumPaths",500,"Confidence",0.95);
irfshock2resp3 = Response(:,2,1);
IRFCIShock2Resp3 = [Lower(:,2,1) Upper(:,2,1)];
[YIRF,h_varc] = armairf(EstMdl1_VARf_c.AR,{},'InnovCov',EstMdl1_VARf_c.Covariance);
  figure; 
   h1 = plot(0:19,irfshock2resp3, 'LineWidth', 1.6);
   hold on
   h2 = plot(0:19,IRFCIShock2Resp3,'r--', 'LineWidth', 1.6);
   legend([h1 h2(1)],["IRF" "95% Confidence Interval"])
   xlabel("Quarters");
   ylabel("Response");
   title("IRF of Unemployment Rate When Inflation Is Shocked");
   grid on
   hold off
   %Graph with out confidence interval 
  figure;
   plot(0:19,irfshock2resp3, 'LineWidth', 1.6);
   xlabel("Quarters");
   ylabel("Response");
   title("IRF of Unemployment Rate When Inflation Is Shocked");

%---------------------------------------------------------------------------
%%%% ----Unconditional VAR={inflation, GDP Output} --
%---------------------------------------------------------------------------
A2=[inflation data_gq(:,1)];
%u_rate=data_gq(:,6); 
numseries=2;
seriesnam_1={'Inflation',  'GDP growth'  };
VARf_2=varm(numseries,3); %3 is the number of lags 
VARf_2.SeriesNames=seriesnam_1;
[EstMdl2_VARf,EstSE2_VARf,logL2_VARf,E2_VARf]= estimate(VARf_c,A2(idxEst,:), 'Y0',A1(idxPre,:));
%%Check Model Adequacy
EstMdl2_VARf.Description
results2_VARf=summarize(EstMdl2_VARf);
np2_VARf=results2_VARf.NumEstimatedParameters;

%%Forecast predictions. Supposing that a negative supply idiosyncratic shock happens
%at the begining of 2008 affecting the unmeployment rate (first), let say
%an increase of in the unemployment rate of 5%
[FY_VARf_2,Forecastmse_var2] = forecast(EstMdl2_VARf,fh,A2(idxEst,:));
YPred_var2=[EndPt_3; FY_VARf_2(:,1)];

%Estimating the 95% forecast interval for the best fitting model

MSE_var_2=cellfun(extractMSE, Forecastmse_var2, 'UniformOutput', false);
SE_var_2=sqrt(cell2mat(MSE_var_2));

%%%%%%%%%%%%%%%%%%%%%   
%%%%%%%%%%%%%%%%%%%%%% 
YFI_var_2= zeros(fh,EstMdl2_VARf.NumSeries,2);
YFI_var_2(:,:,1) = FY_VARf_2 - 2*SE_var_2; %Lower Band
YFI_var_2(:,:,2) = FY_VARf_2 + 2*SE_var_2; %Upper Band
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%

 figure; 
 for j_2=1:numseries
    subplot(numseries,1,j_2);
    h1= plot(TTF(100:T),A2(100:T,j_2), 'LineWidth', 1.6); %Here we can control the sample of the graph, remember that TFF and inflation will have the same leght, in this case this index will help us to select a sample for the porpuse of the graph
    hold on; 
    h2=plot(TTF(idxF), FY_VARf_2(:,j_2), 'LineWidth', 1 );
    h3=plot(TTF(idxF),YFI_var_2(:,j_2,1), 'k--', 'LineWidth', 1);
    plot(TTF(idxF),YFI_var_2(:,j_2,2), 'k--', 'LineWidth', 1);
    title(VARf_2.SeriesNames{j_2});
    % Here you can change the numbers of the "y" axes
    h=gca;
    fill([TTF(idxF(1)) h.XLim([2 2]) TTF(idxF(1))], h.YLim([1 1 2 2]), 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    legend([h1 h2 h3], 'Observed Figure', 'Forecast', '95% Forecast Interval', 'Location', 'best')
    annotation('textbox',dim,'String',ann,'EdgeColor', 'none' );
    sgtitle('Graph: Unconditional Recursive Forecast for UK (VAR2)' )
    set(gcf, 'color',  'w')
    legend('boxoff');
    %ylim([-8 12]);
    hold off;
 end
 
%--------------------------------------------------------------------------------------------------------------------------
%%%% ----Unconditional high-VAR={inflation, GDP Output, industrial prodution, Commodity Prices, FTSE, Unemployment Rate} --
%-------------------------------------------------------------------------------------------------------------------------


A3=[inflation  data_gq(:,3), data_gq(:,4) data_gq(:,5) data_gq(:,6)];
%u_rate=data_gq(:,6); 
numseries_2=5;
seriesnam_2={'Inflation','Industrial Production', 'Commodity Prices', 'FTSE'  ,'Unemployment Rate' };
VARf_3=varm(numseries_2,3); %3 is the number of lags 
VARf_3.SeriesNames=seriesnam_2;
[EstMdl3_VARf,EstSE3_VARf,logL3_VARf,E3_VARf]= estimate(VARf_3,A3(idxEst,:), 'Y0',A3(idxPre,:));
%%Check Model Adequacy
EstMdl3_VARf.Description
results3_VARf=summarize(EstMdl3_VARf);
np3_VARf=results3_VARf.NumEstimatedParameters;

%%Forecast predictions. Supposing that a negative supply idiosyncratic shock happens
%at the begining of 2008 affecting the unmeployment rate (first), let say
%an increase of in the unemployment rate of 5%
[FY_VARf_3,Forecastmse_var3] = forecast(EstMdl3_VARf,fh,A3(idxEst,:));
YPred_var3=[EndPt_3; FY_VARf_3(:,1)];

%Estimating the 95% forecast interval for the best fitting model

MSE_var_3=cellfun(extractMSE, Forecastmse_var3, 'UniformOutput', false);
SE_var_3=sqrt(cell2mat(MSE_var_3));

%%%%%%%%%%%%%%%%%%%%%   
%%%%%%%%%%%%%%%%%%%%%% 
YFI_var_3= zeros(fh,EstMdl3_VARf.NumSeries,2);
YFI_var_3(:,:,1) = FY_VARf_3 - 2*SE_var_3; %Lower Band
YFI_var_3(:,:,2) = FY_VARf_3 + 2*SE_var_3; %Upper Band
%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%

 figure; 
 for j_3=1:numseries_2
    subplot(numseries_2,1,j_3);
    h1= plot(TTF(100:T),A3(100:T,j_3), 'LineWidth', 1.6); %Here we can control the sample of the graph, remember that TFF and inflation will have the same leght, in this case this index will help us to select a sample for the porpuse of the graph
    hold on; 
    h2=plot(TTF(idxF), FY_VARf_3(:,j_3), 'LineWidth', 1 );
    h3=plot(TTF(idxF),YFI_var_3(:,j_3,1), 'k--', 'LineWidth', 1);
    plot(TTF(idxF),YFI_var_3(:,j_3,2), 'k--', 'LineWidth', 1);
    title(VARf_3.SeriesNames{j_3});
    % ylim([-1 2.5]) % Here you can change the numbers of the "y" axes
    h=gca;
    fill([TTF(idxF(1)) h.XLim([2 2]) TTF(idxF(1))], h.YLim([1 1 2 2]), 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    legend([h1 h2 h3], 'Observed Figure', 'Forecast', '95% Forecast Interval', 'Location', 'best')
    %annotation('textbox',dim,'String',ann,'EdgeColor', 'none' );
    sgtitle('Graph: Unconditional Recursive Forecast for UK (VAR3)' )
    set(gcf, 'color',  'w')
    legend('boxoff');
    hold off;
 end
 
%----------------------------------------------------------------
%%% Model and Forecast Error Assestment of ARIMA and  VAR models--
%----------------------------------------------------------------
%---Root mean Square Error(RMSE), MAE and Autocorrelation Funtion of the
%model(residuals)
   %----ARIMA
RMSE_aic=sqrt(mean(res_arima_inflation.^2));
MAE_aic_= mean(abs(res_arima_inflation));
acf_Error_aic = autocorr(res_arima_inflation);
   % VAR models (only for inflation variable)
RMSE_VAR1_c=sqrt(mean(E1_VARf_c(:,1).^2)); %VAR1(inflation, unemployment)
MAE1_VAR1_c= mean(abs(E1_VARf_c(:,1)));
acf_Error_VAR1_c = autocorr(E1_VARf_c(:,1));

RMSE_VAR2=sqrt(mean(E2_VARf(:,1).^2)); %VAR2(inflation,GDP)
MAE1_VAR2= mean(abs(E2_VARf(:,1)));
acf_Error_VAR2 = autocorr(E2_VARf(:,1));

RMSE_VAR3=sqrt(mean(E3_VARf(:,1).^2)); %VAR3(inflation,industrial production, commodity, FTSE, unemployment)
MAE1_VAR2= mean(abs(E2_VARf(:,1)));
acf_Error_VAR3 = autocorr(E3_VARf(:,1));

figure;
  subplot(2,2,1)
  autocorr(res_arima_inflation), legend('ARIMA-AIC');
  subplot(2,2,2)
  autocorr(E1_VARf_c(:,1)), legend('VAR1');
  subplot(2,2,3)
  autocorr(E2_VARf(:,1)), legend('VAR2');
  subplot(2,2,4)
  autocorr(E3_VARf(:,1)), legend('VAR3');
  sgtitle('Graph: Residuals of the model')


%----Root mean Square Error (RMSE), MAE and Autocorrelation Funtion of the Forecast
   %----ARIMA
Error1_arima_f=inflation(idxF,:)-for_infl;   
   
RMSE_aic_f=sqrt(mean(Error1_arima_f.^2));
MAE_aic_f= mean(abs(Error1_arima_f));
acf_Error_aic_f = autocorr(Error1_arima_f);
   % VAR models (only for inflation variable)
Error1_VAR1_f=A1(idxF,:)-F_VARf_c;
Error2_VAR2_f=A2(idxF,:)-FY_VARf_2;
Error3_VAR3_f=A3(idxF,:)-FY_VARf_3;

RMSEVAR1_f=sqrt(mean(Error1_VAR1_f(:,1).^2))
RMSEVAR2_f=sqrt(mean(Error2_VAR2_f.^2))
RMSEVAR3_f=sqrt(mean(Error3_VAR3_f(:,1).^2))

MAE1_VAR1_f= mean(abs(Error1_VAR1_f(:,1)))
MAE2_VAR2_f= mean(abs(Error2_VAR2_f(:,1)))
MAE3_VAR3_f= mean(abs(Error3_VAR3_f(:,1)))

acf_Error1_VAR1_f = autocorr(Error1_VAR1_f(:,1));
acf_Error2_VAR2_f = autocorr(Error2_VAR2_f(:,1));
acf_Error3_VAR3_f = autocorr(Error3_VAR3_f(:,1));

SSerror_arima_f=Error1_arima_f(:)'*Error1_arima_f(:);
SSerror1_VAR1_f=Error1_VAR1_f(:)'*Error1_VAR1_f(:);
SSerror2_VAR2_f=Error2_VAR2_f(:)'*Error2_VAR2_f(:);
SSerror3_VAR3_f=Error3_VAR3_f(:)'*Error3_VAR3_f(:);

figure;
 bar([SSerror_arima_f SSerror1_VAR1_f SSerror2_VAR2_f SSerror3_VAR3_f], .5);
 ylabel('Sum of squared error');
 set(gca, 'XTickLabel', {'ARIMA' 'VAR1' 'VAR2' 'VAR3'});
 title('Sum of Squared Forecast Errors');

figure;
  subplot(2,2,1)
  autocorr(Error1_arima_f), legend('ARIMA-AIC');
  subplot(2,2,2)
  autocorr(Error1_VAR1_f(:,1)), legend('VAR1');
  subplot(2,2,3)
  autocorr(Error2_VAR2_f(:,1)), legend('VAR2');
  subplot(2,2,4)
  autocorr(Error3_VAR3_f(:,1)), legend('VAR3');
  sgtitle('Graph: Forecast Error of the Models')

% Plotinf the Forecast Errors 
  
 TTF_1 = (2008.25:.25:2014.75)';
 figure;
 plot(TTF_1',[Error1_arima_f,Error1_VAR1_f(:,1),Error2_VAR2_f(:,1),Error3_VAR3_f(:,1)],'LineWidth', 1)
 legend('ARIMA-AIC','VAR1','VAR2','VAR3','Location','best ') 
 legend('boxoff');
 annotation('textbox',dim,'String',ann,'EdgeColor', 'none' );
 title(['Graph:Forecast Errors(h=27)']), xlabel({''});

 %Diebold-Mariano Test
 DM_1 = dmtest(Error1_arima_f, Error1_VAR1_f(:,1), 1); %Arima vs VAR1
 DM_1
 DM_2 = dmtest(Error1_arima_f, Error2_VAR2_f(:,1), 1); %ARIMA vs VAR2
 DM_2
 DM_3 = dmtest(Error1_arima_f, Error3_VAR3_f(:,1), 1); %ARIMA vs VAR3
 DM_3
 DM_4 = dmtest(Error2_VAR2_f(:,1), Error1_VAR1_f(:,1), 1); %VAR2 vs VAR3
 DM_4
 DM_5 = dmtest(Error2_VAR2_f(:,1), Error1_VAR1_f(:,1), 1); %VAR2 vs VAR1
 DM_5

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%--- Producing forecast with VAR model for Inflation (CPI)= {GDP growth, unemployment Rate}----%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 horiz=27; % forecasting horizon
 z=27
 S=84
 u=1
 F_error_var1=zeros(z,horiz);
 F_error_var2=zeros(z,horiz);
 F_error_var3=zeros(z,horiz);
 
for c= 1:z
  Yi=target_1(1:S+c);
  y_actual=target_1(S+c+1:S+c+horiz);
 % defining the set of indicators for VAR models
  indicator2i=data_gq(1:S+c,1); % GDP growth
  indicator3i=data_gq(1:S+c,3); % IP growth
  indicator4i=data_gq(1:S+c,4); % Commodity price growth
  indicator5i=data_gq(1:S+c,5); % FTSE growth
  indicator6i=data_gq(1:S+c,6); % unemployment rate - NB in levels
%%%%-------Model 1. VAR{CPI, GDP grwoth}
  Y_1=[Yi,indicator2i];
  numseries=2;
  seriesnam_1={'Inflation',  'GDP growth'  };
  VAR1f=varm(numseries,3); % the number is the lag 
  VAR1f.SeriesNames=seriesnam_1;
  [EstMdl1,EstSE1,logL1,E1]= estimate(VAR1f,Y_1);
  EstMdl1.Description
  results1=summarize(EstMdl1);
  np1=results1.NumEstimatedParameters;
  summarize(EstMdl1);
  numperiods = horiz;
  [FY1,Forecastmse1]=forecast(EstMdl1,numperiods,Y_1);
  F_error_var1(c,:)=(y_actual-FY1(:,1))'; 

%%%% -------Model 2. VAR= {CPI, Unemployment Rate }
  Y_2=[Yi,indicator6i];
  numseries=2;
  seriesnam_2={'Inflation',  'Unemployment Rate' };
  VAR2f=varm(numseries,3); % the number is the lag 
  VAR2f.SeriesNames=seriesnam_2;
  [EstMdl2,EstSE2,logL2,E2]= estimate(VAR2f,Y_2);
  EstMdl2.Description
  results2=summarize(EstMdl2);
  np2=results2.NumEstimatedParameters;
  summarize(EstMdl2);
  [FY2,Forecastmse2]=forecast(EstMdl2,numperiods,Y_2);
  % Evaluating the forecast 
  F_error_var2(c,:)=(y_actual-FY2(:,1))';  
 
 %%%% -------Model 3. VAR_higer= {CPI, Unemployment Rate, Industrial Production, Commodity Prices, FTSE index }
  Y_3=[Yi, indicator3i, indicator4i, indicator5i, indicator6i];
  numseries_2=5;
  seriesnam_3={'Inflation', 'Industrial Production', 'Commodity Prices', 'FTSE'  ,'Unemployment Rate'  };% the growth rate it is excluded in order to avoid collinearity with the unemployment rate 
  VAR3f=varm(numseries_2,3); % the number is the lag 
  VAR3f.SeriesNames=seriesnam_3;
  [EstMdl3,EstSE3,logL3,E3]= estimate(VAR3f,Y_3);
  EstMdl3.Description
  results3=summarize(EstMdl3);
  np3=results3.NumEstimatedParameters;
  summarize(EstMdl3);
  [FY3,Forecastmse3]=forecast(EstMdl3,numperiods,Y_3);
  % Evaluating the forecast 
  F_error_var3(c,:)=(y_actual-FY3(:,1))';  

 %%%--------- Model 4. Estimating an AR(2)
  [ti,~]=size(Yi);
  po=2; 
  Xar=[ones(ti-po,1),Yi(po+1-1:ti-1),Yi(po+1-2:ti-2)];
  Yar=Yi(po+1:ti); 
  bar=inv(Xar'*Xar)*(Xar'*Yar); 
  % computing forecasts from AR(2) by iteration
  FY4=zeros(po+horiz,1);
  FY4(1:po)=Yar(end-po+1:end); 
   for j_1=po+1:horiz+po
      FY4(j_1)=[1,FY4(j_1-1),FY4(j_1-2)]*bar; 
   end 
  % saving forecasting errors
  F_error_ar2(c,:)=(y_actual-FY4(po+1:po+horiz))';
  
%%%--- Model 5. Computing forecasts from Random Walk
  FY5=zeros(1+horiz,1);
  FY5(1:1)=Yar(end-1+1:end); 
   for j_1=1+1:horiz+1
      FY5(j_1)=[1,FY5(j_1-1)]*[0 1]'; 
   end 
  % saving forecasting errors
  F_error_RW(c,:)=(y_actual-FY5(1+1:1+horiz))';
  
 %%%--- Model 6. Unconditional forecast - recursive regression on an intercept
  [ti,~]=size(Yi);
   Xar=[ones(ti,1)];
   Yar=Yi(1:ti); 
   bar_1=inv(Xar'*Xar)*(Xar'*Yar); % OLS estimation
   bar_1=2; % overwrite and set equal to inflation target
  % computing forecasts 
   FY6=zeros(1+horiz,1);
    for j_1=1:horiz
      FY6(j_1)=[1]*bar_1; 
    end 
  
  % saving forecasting errors
   F_error_U(c,:)=(y_actual-FY6(1:horiz))';
   
   
  %%%--- Model 7. ARIMA-AIC (selected model)--> ARIMA(p,0,q)  
   
   model_1=arima(p,0,q);% p=AR order, d=order of difference, q=MA order
   [fit_1,VarCov_1,logL_1,info_1]=estimate(model_1,inflation(idxEst,:), 'Y0',inflation(idxPre,:));
   [res_1, ~, logL_1]=infer(fit_1,inflation(idxEst,:), 'Y0',inflation(idxPre,:)); 
   [FY7,Forecastmse7,V]=forecast(fit_1,numperiods,'Y0',inflation(idxEst,:));
   % Evaluating the forecast 
   F_error_arima(c,:)=(y_actual-FY7(:,1))';
   
   end
 
 %Estimating the 95% forecast interval for the best fitting model
 extractMSE=@(x)diag(x)';
 MSE_VAR1=cellfun(extractMSE, Forecastmse1, 'UniformOutput', false);
 SE_VAR1=sqrt(cell2mat(MSE_VAR1));
 %%%%%%%%%%%%%%%%%%%%%   
 %%%%%%%%%%%%%%%%%%%%%% 
 YFI_VAR1= zeros(fh,EstMdl1.NumSeries,2);
 YFI_VAR1(:,:,1) = FY1 - 2*SE_VAR1; %Lower Band 
 YFI_VAR1(:,:,2) = FY1 + 2*SE_VAR1; %Upper Band
 %%%%%%%%%%%%%%%%%%%%%%%
 %%%%%%%%%%%%%%%%%%%%%%%%

%%%Ploting Forecast of VAR againts the real figures 
DY_1= [data_gq(:,2),data_gq(:,1)]; 
 figure; 
 for j_2=1: numseries
    subplot(2,1,j_2);
    h1= plot(TTF(100:138),DY_1(100:138,j_2), 'LineWidth', 1.6); %Here we can control the sample of the graph, remember that TFF and inflation will have the same leght, in this case this index will help us to select a sample for the porpuse of the graph
    hold on; 
    h2=plot(TTF(idxF), FY1(:,j_2), 'LineWidth', 1 );
    h3=plot(TTF(idxF),YFI_VAR1(:,j_2,1), 'k--', 'LineWidth', 1);
    plot(TTF(idxF),YFI_VAR1(:,j_2,2), 'k--', 'LineWidth', 1);
    title(VAR1f.SeriesNames{j_2});
    ylim([-3 3]) % Here you control the size of "y" axes
    h=gca;
    fill([TTF(idxF(1)) h.XLim([2 2]) TTF(idxF(1))],h.YLim([1 1 2 2]) , 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    legend([h1 h2 h3], 'Observed Figure', 'Forecast', '95% Forecast Interval', 'Location', 'best')
    %annotation('textbox',dim,'String',ann,'EdgeColor', 'none' );
    set(gcf, 'color',  'w')
    legend('boxoff');
    sgtitle('Graph: Unconditional Rolling Forecast for UK (VAR1)' )
    hold off;
 end
%%%%%%%%%%%%%%%%%%%%%%
%Estimating the 95% forecast interval for the best fitting model
 MSE_VAR2=cellfun(extractMSE, Forecastmse2, 'UniformOutput', false);
 SE_VAR2=sqrt(cell2mat(MSE_VAR2));
%%%%%%%%%%%%%%%%%%%%%   
%%%%%%%%%%%%%%%%%%%%%% 
 YFI_VAR2= zeros(fh,EstMdl2.NumSeries,2);
 YFI_VAR2(:,:,1) = FY2 - 2*SE_VAR2; %Lower Band 
 YFI_VAR2(:,:,2) = FY2 + 2*SE_VAR2; %Upper Band
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%
%%%Ploting Forecast of VAR againts the real figures 
 DY_2= [data_gq(:,2),data_gq(:,6)];
 figure; 
 for j_2=1: numseries
    subplot(numseries,1,j_2);
    h1= plot(TTF(100:138),DY_2(100:138,j_2), 'LineWidth', 1.6); %Here we can control the sample of the graph, remember that TFF and inflation will have the same leght, in this case this index will help us to select a sample for the porpuse of the graph
    hold on; 
    h2=plot(TTF(idxF), FY2(:,j_2), 'LineWidth', 1 );
    h3=plot(TTF(idxF),YFI_VAR2(:,j_2,1), 'k--', 'LineWidth', 1);
    plot(TTF(idxF),YFI_VAR2(:,j_2,2), 'k--', 'LineWidth', 1);
    title(VAR2f.SeriesNames{j_2});
    %ylim([-1 2.5]) % Here you can change the numbers of the "y" axes
    h=gca;
    fill([TTF(idxF(1)) h.XLim([2 2]) TTF(idxF(1))], h.YLim([1 1 2 2]), 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    legend([h1 h2 h3], 'Observed Figure', 'Forecast', '95% Forecast Interval', 'Location', 'best')
    %annotation('textbox',dim,'String',ann,'EdgeColor', 'none' );
    set(gcf, 'color',  'w')
    legend('boxoff');
    sgtitle('Graph: Unconditional Rolling Forecast for UK (VAR2)' )
    hold off;
 end
 
%Estimating the 95% forecast interval for the best fitting model
%%%%%%%
extractMSE=@(x)diag(x)';
MSE_VAR3=cellfun(extractMSE, Forecastmse3, 'UniformOutput', false);
SE_VAR3=sqrt(cell2mat(MSE_VAR3));
%%%%%%%%%%%%%%%%%%%%%   
%%%%%%%%%%%%%%%%%%%%%% C
YFI_VAR3= zeros(fh,EstMdl3.NumSeries,5);
YFI_VAR3(:,:,1) = FY3 - 2*SE_VAR3; %Lower Band 
YFI_VAR3(:,:,2) = FY3 + 2*SE_VAR3; %Upper Band
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%
%%%Ploting Forecast of VAR againts the real figures 
DY_3= [data_gq(:,2),data_gq(:,3),data_gq(:,4),data_gq(:,5),data_gq(:,6)];
 figure; 
 for j_3=1: numseries_2
    subplot(numseries_2,1,j_3);
    h1= plot(TTF(100:138),DY_3(100:138,j_3), 'LineWidth', 1.6); %Here we can control the sample of the graph, remember that TFF and inflation will have the same leght, in this case this index will help us to select a sample for the porpuse of the graph
    hold on; 
    h2=plot(TTF(idxF), FY3(:,j_3), 'LineWidth', 1 );
    h3=plot(TTF(idxF),YFI_VAR3(:,j_3,1), 'k--', 'LineWidth', 1);
    plot(TTF(idxF),YFI_VAR3(:,j_3,2), 'k--', 'LineWidth', 1);
    title(VAR3f.SeriesNames{j_3});
    %ylim([-1 2.5]) % Here you can change the numbers of the "y" axes
    h=gca;
    fill([TTF(idxF(1)) h.XLim([2 2]) TTF(idxF(1))], h.YLim([1 1 2 2]), 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    legend([h1 h2 h3], 'Observed Figure', 'Forecast', '95% Forecast Interval', 'Location', 'best')
    %annotation('textbox',dim,'String',ann,'EdgeColor', 'none' );
    set(gcf, 'color',  'w')
    legend('boxoff');
    sgtitle('Graph: Unconditional Rolling Forecast for UK (VAR3)' )
    hold off;
 end

%%%%%%%
 RMSEVAR1=sqrt(mean(F_error_var1.^2));
 MAE1= mean(abs(F_error_var1));
 RMSEVAR2=sqrt(mean(F_error_var2.^2)); % Root Mean Squared Error
 MAE_var2= mean(abs(F_error_var2)); % Mean absolute error
 RMSEVAR3=sqrt(mean(F_error_var3.^2)); % Root Mean Squared Error
 MAE_var3= mean(abs(F_error_var3)) ;
 RMSEAR2=sqrt(mean(F_error_ar2.^2)); % Root Mean Squared Error
 MAE_ar= mean(abs(F_error_ar2)) ;
 RMSEARW=sqrt(mean(F_error_RW.^2)); % Root Mean Squared Error
 MAE_RW= mean(abs(F_error_RW)) ;
 RMSEU=sqrt(mean(F_error_U.^2)); % Root Mean Squared Error
 MAE_U= mean(abs(F_error_U)) ;
 RMSEARIMA=sqrt(mean(F_error_arima.^2)); % Root Mean Squared Error
 MAE_U= mean(abs(F_error_arima)) ;
 
disp('---------------------------------------------------------------------------');
disp('             Root Mean Squared Error for each horizon (=27)                ');
disp('---------------------------------------------------------------------------');
disp('    VAR1      VAR2      VAR3      AR(2)     RW        UN        ARIMA-AIC'); 
 [RMSEVAR1'   ,RMSEVAR2' ,RMSEVAR3' , RMSEAR2' , RMSEARW' , RMSEU',  RMSEARIMA' ]


% Diebold-Mariano and Giacomini-White tests
DMs=zeros(horiz,1); % saves DM tests across horizons
%GWs=zeros(horizon,1); % saves GW tests across horizons
alpha=0.05; % alpha, niminal risk level: 1%, 5%, 10% for GW test
choice= 2; % if unconditional ; 2 if conditional for GW test

for hor=1:horiz
DM = dmtest(F_error_var3(:,hor), F_error_ar2(:,hor), hor); % procedure itself squares forecast errors. 
                                                     % Here set up to run test on errors from model 1 and 3
DMs(hor,1)=DM;

%[teststat,critval,pval]=CPAtest(F_error1(:,hor).^2, F_error3(:,hor).^2,hor, alpha, choice);
%GWs(hor,1)=pval; % save the p-value from the GW test
end

%--- Generating the forecast errors for the the autocorrelation function


 
F_Error1_VAR1_f=A1(idxF,1)-FY1(:,1);
F_Error2_VAR2_f=A2(idxF,1)-FY2(:,1);
F_Error3_VAR3_f=A3(idxF,1)-FY3(:,1);
F_Error4_ar_f=inflation(110:138,1)-FY4;
F_Error5_rw_f=inflation(111:138,:)-FY5;
F_Error6_uc_f=inflation(111:138,:)-FY6;
F_Error7_arima_f=inflation(idxF,:)-FY7;

%%Ploting the autocoorelation funtion of the Forecast error  
  figure;
   subplot(4,2,1)
   autocorr(F_Error1_VAR1_f), legend('Forecast Error (VAR1)');
   subplot(4,2,2)
   autocorr(F_Error2_VAR2_f), legend('Forecast Error (VAR2)');
   subplot(4,2,3)
   autocorr(F_Error3_VAR3_f), legend('Forecast Error (VAR3)');
   subplot(4,2,4)
   autocorr(F_Error4_ar_f), legend('Forecast Error (AR)');
   subplot(4,2,5)
   autocorr(F_Error5_rw_f), legend('Forecast Error (RW)');
   subplot(4,2,6)
   autocorr(F_Error6_uc_f), legend('Forecast Error (uc)');
   subplot(4,2,7)
   autocorr(F_Error7_arima_f), legend('Forecast Error (ARIMA)');
   sgtitle('Graph: Forecast Error of the Models')
 % Partial corelation funtion  
  figure;
   subplot(4,2,1)
   parcorr(F_Error1_VAR1_f), legend('Forecast Error (VAR1)');
   subplot(4,2,2)
   parcorr(F_Error2_VAR2_f), legend('Forecast Error (VAR2)');
   subplot(4,2,3)
   parcorr(F_Error3_VAR3_f), legend('Forecast Error (VAR3)');
   subplot(4,2,4)
   parcorr(F_Error4_ar_f), legend('Forecast Error (AR)');
   subplot(4,2,5)
   parcorr(F_Error5_rw_f), legend('Forecast Error (RW)');
   subplot(4,2,6)
   parcorr(F_Error6_uc_f), legend('Forecast Error (UC)');
   subplot(4,2,7)
   parcorr(F_Error7_arima_f), legend('Forecast Error (ARIMA)');
   sgtitle('Graph: Forecast Error of the Models')
    
 %%% Ploting the error of the models 
 % plot errors at a specific horizon

  
 TTF_1 = (2008.25:.25:2014.75)';
  figure;
   h1=plot(TTF_1,[,inflation(idxF,:),F_error_var1(:,horiz),F_error_var2(:,horiz),F_error_var3(:,horiz),F_error_ar2(:,horiz),F_error_RW(:,horiz),F_error_U(:,horiz),F_error_arima(:,horiz)],'LineWidth', 1)
   hold on
   legend('Observed Figure','VAR1','VAR2','VAR3','AR(2)','Random Walk','Unconditional','ARIMA-AIC','Location','best ') 
   legend('boxoff');
   title(['Graph: Forecast Errors at the specific horizon (=27)']), xlabel({''});
   hold off

%%%%
%%%%  https://uk.mathworks.com/help/econ/converting-from-vgx-functions-to-model-objects.html
%%%%  https://uk.mathworks.com/help/econ/var-model-forecasting-simulation-and-analysis.html

    








