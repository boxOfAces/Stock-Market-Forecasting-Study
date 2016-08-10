#ARIMA code
library("quantmod")
library("forecast")

startDate = as.Date("2010-01-01")
endDate = as.Date("2016-01-01")

getSymbols("INFY.NS", src = "yahoo", from = startDate, to = endDate)
plot(INFY.NS$INFY.NS.Close)
data<-ts(INFY.NS[,4],start = c(2010,1), end = c(2016,1), frequency = 365)

plot(data, xlab="Years", ylab = "Infosys Closing Price")

plot(diff(log10(data)),ylab="Differenced Log (Infosys Closing Price)")

par(mfrow = c(1,2))
acf(ts(diff(log10(data))),main="ACF Infy Close Price")
pacf(ts(diff(log10(data))),main="PACF Infy Close Price")

ARIMAfit <- auto.arima(log10(data), approximation=FALSE,trace=FALSE)
summary(ARIMAfit)

#forecasting
pred <- forecast(ARIMAfit, 100)
plot(pred, xlab = "Year",ylab = "Infy Close Price")
