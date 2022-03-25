library(fredr)
library(fpp2)
library(forecast)
library(rstan)
library(prophet)
library(ggplot2)
library(MLmetrics)
library(tseries)

rm(list=ls(all=TRUE))

# Insert own API key from FRED
fredr_set_key(fred_key)

sales    <- fredr(series_id="MRTSSM4453USN")
sales_ts <- ts(log(sales$value), start=1992, frequency=12) 

# Check for NAs 
any(is.na(sales_ts))

# Plot 
autoplot(sales_ts)

#Decomposition 
autoplot(stl(sales_ts, s.window="periodic"), main="STL Decomposition")

#TSdisplay
ggtsdisplay(sales_ts, main="Series with ACF and PACF") 

n     <- round((2/3)*length(sales_ts))
train <- ts(sales_ts[1:n], start=1992, frequency=12)
test  <- ts(sales_ts[n+1:length(sales_ts)], start=2011.917, frequency=12)
test  <- na.remove(test)

# ARIMA 
f_arima <- forecast(auto.arima(train), h=length(sales_ts)-n)

autoplot(train, series="Train") +
  autolayer(test, series="Test") + 
  autolayer(f_arima$mean, series="ARIMA") + 
  ggtitle("Forecasting with ARIMA") + 
  ylab("Log(Sales)") + 
  labs(colour="")

arima_recur <- c() 
for (i in n:length(sales_ts)) { 
  df          <- sales_ts[1:i] 
  model       <- auto.arima(df)
  predict     <- forecast(model, h=12) 
  actual      <- sales_ts[(i+1):(i+12)] 
  arima_recur <- cbind(arima_recur, MAPE(predict$mean, actual))
}

arima_recur    <- na.omit(as.data.frame(t(arima_recur)))
h              <- c(1:nrow(arima_recur))
arima_recur_df <- cbind.data.frame(h, arima_recur)

ggplot(arima_recur_df, mapping=aes(x=h, y=V1)) +
  geom_line() +
  ggtitle("Recursive Backtesting for ARIMA, 12 Steps Ahead") +
  xlab("Iteration") +
  ylab("MAPE") +
  ylim(0, 0.05) + 
  annotate("text", x=25, y=0.03, 
           label=paste("Average MAPE Score = ", 
                       as.character(round(mean(arima_recur$V1, na.rm=T),5))))
  
print(mean(arima_recur$V1, na.rm=T))

arima_roll <- c()
for (i in 0:(length(sales_ts)-n)) { 
  df         <- sales_ts[(1+i):(n+i)] 
  model      <- auto.arima(df)
  predict    <- forecast(model, h=12) 
  actual     <- sales_ts[(n+i+1):(n+i+12)]
  arima_roll <- cbind(arima_roll, MAPE(predict$mean, actual))
}

arima_roll    <- na.omit(as.data.frame(t(arima_roll)))
h             <- c(1:nrow(arima_roll))
arima_roll_df <- cbind.data.frame(h, arima_roll) 

ggplot(arima_roll_df, mapping=aes(x=h, y=V1)) +
  geom_line() +
  ggtitle("Rolling Window Backtesting for ARIMA, 12 Steps Ahead") +
  xlab("Iteration") +
  ylab("MAPE") +
  ylim(0, 0.05) + 
  annotate("text", x=25, y=0.03, 
           label=paste("Average MAPE Score = ", 
                       as.character(round(mean(arima_roll$V1, na.rm=T),5))))

print(mean(arima_roll$V1, na.rm=T))


  



# ETS 
f_ets <- forecast(ets(train), h=length(sales_ts)-n)

autoplot(train, series="Train") +
  autolayer(test, series="Test") + 
  autolayer(f_ets$mean, series="ETS") + 
  ggtitle("Forecasting with ETS") + 
  ylab("Log(Sales)") + 
  labs(colour="")  

ets_recur <- c() 
for (i in n:length(sales_ts)) { # i in 239:358
  df        <- sales_ts[1:i] # 1:239 (train)
  model     <- ets(df)
  predict   <- forecast(model, h=12) 
  actual    <- sales_ts[(i+1):(i+12)] # i + 1 -> i + 12 
  ets_recur <- cbind(ets_recur, MAPE(predict$mean, actual))
}

ets_recur    <- na.omit(as.data.frame(t(ets_recur)))
h            <- c(1:nrow(ets_recur))
ets_recur_df <- cbind.data.frame(h, ets_recur) 

ggplot(ets_recur_df, mapping=aes(x=h, y=V1)) +
  geom_line() +
  ggtitle("Recursive Backtesting for ETS, 12 Steps Ahead") +
  xlab("Iteration") +
  ylab("MAPE") +
  ylim(0, 0.05) + 
  annotate("text", x=25, y=0.03, 
           label=paste("Average MAPE Score = ", 
                       as.character(round(mean(ets_recur$V1, na.rm=T),5))))

print(mean(ets_recur$V1, na.rm=T))

ets_roll <- c()
for (i in 0:(length(sales_ts)-n)) { # i in 0:119
  df       <- sales_ts[(1+i):(n+i)] # train: 1 + i : 239 + i
  model    <- ets(df)
  predict  <- forecast(model, h=12) 
  actual   <- sales_ts[(n+i+1):(n+i+12)]
  ets_roll <- cbind(ets_roll, MAPE(predict$mean, actual))
}

ets_roll    <- na.omit(as.data.frame(t(ets_roll)))
h           <- c(1:nrow(ets_roll))
ets_roll_df <- cbind.data.frame(h, ets_roll) 

ggplot(ets_roll_df, mapping=aes(x=h, y=V1)) +
  geom_line() +
  ggtitle("Rolling Window Backtesting for ETS, 12 Steps Ahead") +
  xlab("Iteration") +
  ylab("MAPE") +
  ylim(0, 0.05) + 
  annotate("text", x=25, y=0.03, 
           label=paste("Average MAPE Score = ", 
                       as.character(round(mean(ets_roll$V1, na.rm=T),5))))

print(mean(ets_roll$V1, na.rm=T))





# Holt Winters
f_hw <- forecast(HoltWinters(train), h=length(sales_ts)-n)
autoplot(train, series="Train") + 
  autolayer(test, series="Test") + 
  autolayer(f_hw$mean, series="HW") + 
  labs(colour="") + 
  ggtitle("Forecasting with Holt Winters")

hw_recur <- c() 
for (i in n:length(sales_ts)) { # i in 239:358
  df       <- ts(sales_ts[1:i], frequency=12) # 1:239 (train)
  model    <- HoltWinters(df)
  predict  <- forecast(model, h=12) 
  actual   <- sales_ts[(i+1):(i+12)] # i + 1 -> i + 12 
  hw_recur <- cbind(hw_recur, MAPE(predict$mean, actual))
}

hw_recur    <- na.omit(as.data.frame(t(hw_recur)))
h           <- c(1:nrow(hw_recur))
hw_recur_df <- cbind.data.frame(h, hw_recur) 

ggplot(hw_recur_df, mapping=aes(x=h, y=V1)) +
  geom_line() +
  ggtitle("Recursive Backtesting for HW, 12 Steps Ahead") +
  xlab("Iteration") +
  ylab("MAPE") +
  ylim(0, 0.05) + 
  xlim(0, 120) +
  annotate("text", x=25, y=0.03, 
           label=paste("Average MAPE Score = ", 
                       as.character(round(mean(hw_recur$V1, na.rm=T),5))))

print(mean(hw_recur$V1, na.rm=T))

hw_roll <- c()
for (i in 0:(length(sales_ts)-n)) { # i in 0:119
  df      <- ts(sales_ts[(1+i):(n+i)], frequency=12) # train: 1 + i : 239 + i
  model   <- HoltWinters(df)
  predict <- forecast(model, h=12) 
  actual  <- sales_ts[(n+i+1):(n+i+12)]
  hw_roll <- cbind(hw_roll, MAPE(predict$mean, actual))
}

hw_roll    <- na.omit(as.data.frame(t(hw_roll)))
h          <- c(1:nrow(hw_roll))
hw_roll_df <- cbind.data.frame(h, hw_roll) 

ggplot(hw_roll_df, mapping=aes(x=h, y=V1)) +
  geom_line() +
  ggtitle("Rolling Window Backtesting for HW, 12 Steps Ahead") +
  xlab("Iteration") +
  ylab("MAPE") +
  ylim(0, 0.05) + 
  xlim(0, 120) +
  annotate("text", x=25, y=0.03, 
           label=paste("Average MAPE Score = ", 
                       as.character(round(mean(hw_roll$V1, na.rm=T),5))))

print(mean(hw_roll$V1, na.rm=T))






# Prophet
sales_df <- cbind.data.frame(sales$date, log(sales$value))
colnames(sales_df) <- c("ds", "y")

train_prophet <- sales_df[1:n,]
test_prophet  <- sales_df[n+1:nrow(sales_df),]

prophet  <- prophet(train_prophet)
future   <- make_future_dataframe(prophet, periods=nrow(sales_df)-n, freq='month')
f_prophet <- predict(prophet, future)
f_prophet_ts <- na.remove(ts(f_prophet$yhat[n+1:nrow(f_prophet)], start=2011.917, 
                   frequency=12))

autoplot(train, series="Train") + 
  autolayer(test, series="Test") + 
  autolayer(f_prophet_ts, series="FBP") + 
  ggtitle("Forecasts from Facebook Prophet") + 
  ylab("Log(Sales)") + 
  labs(colour="")

fbp_recur <- c() 
for (i in n:length(sales_ts)) { # i in 239:358
  df        <- sales_df[1:i,] # 1:239 (train)
  model     <- prophet(df)
  future    <- make_future_dataframe(model, periods=12, freq='month')
  f_prophet <- predict(model, future)
  predict   <- f_prophet$yhat[(nrow(f_prophet)-11):nrow(f_prophet)]
  actual    <- sales_ts[(i+1):(i+12)] # i + 1 -> i + 12 
  fbp_recur <- cbind(fbp_recur, MAPE(predict, actual))
}

fbp_recur    <- na.omit(as.data.frame(t(fbp_recur)))
h            <- c(1:nrow(fbp_recur))
fbp_recur_df <- cbind.data.frame(h, fbp_recur) 

ggplot(fbp_recur_df, mapping=aes(x=h, y=V1)) +
  geom_line() +
  ggtitle("Recursive Backtesting for FBP, 12 Steps Ahead") +
  xlab("Iteration") +
  ylab("MAPE") +
  ylim(0, 0.05) + 
  annotate("text", x=25, y=0.03, 
           label=paste("Average MAPE Score = ", 
                       as.character(round(mean(fbp_recur$V1, na.rm=T),5))))

print(mean(fbp_recur$V1, na.rm=T))

fbp_roll <- c()
for (i in 0:(length(sales_ts)-n)) { # i in 0:119
  df        <- sales_df[(1+i):(n+i),] # train: 1 + i : 239 + i
  model     <- prophet(df)
  future    <- make_future_dataframe(model, periods=12, freq='month')
  f_prophet <- predict(model, future)
  predict   <- f_prophet$yhat[(nrow(f_prophet)-11):nrow(f_prophet)]
  actual    <- sales_ts[(n+i+1):(n+i+12)]
  fbp_roll  <- cbind(fbp_roll, MAPE(predict, actual))
}

fbp_roll    <- na.omit(as.data.frame(t(fbp_roll)))
h           <- c(1:nrow(fbp_roll))
fbp_roll_df <- cbind.data.frame(h, fbp_roll) 

ggplot(fbp_roll_df, mapping=aes(x=h, y=V1)) +
  geom_line() +
  ggtitle("Rolling Window Backtesting for FBP, 12 Steps Ahead") +
  xlab("Iteration") +
  ylab("MAPE") +
  ylim(0, 0.05) + 
  annotate("text", x=25, y=0.03, 
           label=paste("Average MAPE Score = ", 
                       as.character(round(mean(fbp_roll$V1, na.rm=T),5))))

print(mean(fbp_roll$V1, na.rm=T))



prophet   <- prophet(train_prophet)
future    <- make_future_dataframe(prophet, periods=nrow(sales_df)-n, 
                                   freq='month')
f_prophet <- predict(prophet, future)
fbp       <- ts(na.remove(f_prophet$yhat[n+1:nrow(f_prophet)]), start=2012,
                frequency=12)

arima  <- forecast(auto.arima(train), h=length(sales_ts)-n)
ets    <- forecast(ets(train), h=length(sales_ts)-n)
holtw  <- forecast(HoltWinters(train), h=length(sales_ts)-n)
comb   <- (fbp + arima[["mean"]] + ets[["mean"]] + holtw[["mean"]])/4

png(file="forecast1.png", height=4000, width=7000, res=500)
autoplot(sales_ts) + 
  autolayer(arima$mean, series="ARIMA") + 
  autolayer(ets$mean, series="ETS") + 
  autolayer(holtw$mean, series="HW") +
  autolayer(fbp, series="FBP") +
  autolayer(comb, series="Combined") +
  ylab("Log(Sales)") + 
  labs(color="Forecasts") +
  ggtitle("Forecasting Using Combined Model") + 
  xlim(2008,2022) + 
  ylim(7.75, 9)

dev.off()

print("Hello")

comb_recur <- c() 
for (i in n:length(sales_ts)) {
  df1 <- sales_ts[1:i]
  
  f1 <- forecast(auto.arima(df1), h=12)
  f2 <- forecast(ets(df1), h=12)
  f3 <- forecast(hw(ts(sales_ts[1:i], start=1, frequency=12)), h=12)
  
  df2 <- sales_df[1:i,]
  m4  <- prophet(df2)
  
  future    <- make_future_dataframe(m4, periods=12, freq='month')
  f_prophet <- predict(m4, future)
  f4        <- f_prophet$yhat[(nrow(f_prophet)-11):nrow(f_prophet)]
  
  f1 <- as.vector(f1$mean)
  f2 <- as.vector(f2$mean)
  f3 <- as.vector(f3$mean)
  f4 <- as.vector(f4)
  
  predict <- (f1 + f2 + f3 + f4)/4
  
  actual     <- sales_ts[(i+1):(i+12)] 
  comb_recur <- cbind(comb_recur, MAPE(predict, actual))
}

comb_recur    <- na.omit(as.data.frame(t(comb_recur)))
h             <- c(1:nrow(comb_recur))
comb_recur_df <- cbind.data.frame(h, comb_recur) 

ggplot(comb_recur_df, mapping=aes(x=h, y=V1)) +
  geom_line() +
  ggtitle("Recursive Backtesting for Combined Model, 12 Steps Ahead") +
  xlab("Iteration") +
  ylab("MAPE") +
  ylim(0, 0.05) + 
  annotate("text", x=25, y=0.03, 
           label=paste("Average MAPE Score = ", 
                       as.character(round(mean(comb_recur$V1, na.rm=T),5))))

print(mean(comb_recur$V1, na.rm=T))

comb_roll <- c()
for (i in 0:(length(sales_ts)-n)) { # i in 0:119
  df1     <- sales_ts[(1+i):(n+i)]# train: 1 + i : 239 + i
  
  f1 <- forecast(auto.arima(df1), h=12)
  f2 <- forecast(ets(df1), h=12)
  f3 <- forecast(hw(ts(sales_ts[(1+i):(n+i)], start=1, frequency=12)), h=12)
  
  df2 <- sales_df[(1+i):(n+i),]
  m4  <- prophet(df2)
  
  future    <- make_future_dataframe(m4, periods=12, freq='month')
  f_prophet <- predict(m4, future)
  f4        <- f_prophet$yhat[(nrow(f_prophet)-11):nrow(f_prophet)]
  
  f1 <- as.vector(f1$mean)
  f2 <- as.vector(f2$mean)
  f3 <- as.vector(f3$mean)
  f4 <- as.vector(f4)
  
  predict <- (f1 + f2 + f3 + f4)/4
  
  actual     <- sales_ts[(n+i+1):(n+i+12)] 
  comb_roll <- cbind(comb_roll, MAPE(predict, actual))
 
}

comb_roll    <- na.omit(as.data.frame(t(comb_roll)))
h            <- c(1:nrow(comb_roll))
comb_roll_df <- cbind.data.frame(h, comb_roll) 

ggplot(comb_roll_df, mapping=aes(x=h, y=V1)) +
  geom_line() +
  ggtitle("Rolling Window Backtesting for Combined Model, 12 Steps Ahead") +
  xlab("Iteration") +
  ylab("MAPE") +
  ylim(0, 0.05) + 
  xlim(0, 120) +
  annotate("text", x=25, y=0.03, 
           label=paste("Average MAPE Score = ", 
                       as.character(round(mean(comb_roll$V1, na.rm=T),5))))

print(mean(comb_roll$V1, na.rm=T))


mape1 <- cbind.data.frame(h, arima_recur$V1, ets_recur$V1, 
                         hw_recur$V1, fbp_recur$V1, comb_recur$V1)
colnames(mape1) <- c("Iteration", "ARIMA", "ETS", "HW", "FBP", "Combined")

ggplot() + 
  geom_line(mape1, mapping=aes(x=Iteration, y=ARIMA, color="ARIMA"), size=1) + 
  geom_line(mape1, mapping=aes(x=h, y=ETS, color="ETS"), size=1) + 
  geom_line(mape1, mapping=aes(x=h, y=HW, color="HW"), size=1) + 
  geom_line(mape1, mapping=aes(x=h, y=FBP, color="FBP"), size=1) + 
  geom_line(mape1, mapping=aes(x=h, y=Combined, color="Combined"), size=1) +
  ggtitle("Recursive Backtesting, 12 Steps Ahead") + 
  ylab("MAPE") + 
  labs(colour="Model")



mape2 <- cbind.data.frame(h, arima_roll$V1, ets_roll$V1, 
                          hw_roll$V1, fbp_roll$V1, comb_roll$V1)
colnames(mape2) <- c("Iteration", "ARIMA", "ETS", "HW", "FBP", "Combined")

ggplot() + 
  geom_line(mape2, mapping=aes(x=Iteration, y=ARIMA, color="ARIMA"), size=1) + 
  geom_line(mape2, mapping=aes(x=h, y=ETS, color="ETS"), size=1) + 
  geom_line(mape2, mapping=aes(x=h, y=HW, color="HW"), size=1) + 
  geom_line(mape2, mapping=aes(x=h, y=FBP, color="FBP"), size=1) + 
  geom_line(mape2, mapping=aes(x=h, y=Combined, color="Combined"), size=1) +
  ggtitle("Rolling Window Backtesting, 12 Steps Ahead") + 
  ylab("MAPE") + 
  labs(colour="Model")


x <- 0
for (i in n:length(sales_ts)) { 
  predict     <- forecast(auto.arima(ts(sales_ts[1:i], start=1, frequency=12)), h=12) 
  x <- x + 1 
  print(x)
  actual      <- sales_ts[(i+1):(i+12)] 
  arima_recur <- cbind(arima_recur, MAPE(predict$mean, actual))
}








