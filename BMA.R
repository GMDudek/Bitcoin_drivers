library(dplyr)
library(readxl)
library(xts)
library(BMS)

new_data <- read_excel("Bitcoin_drivers_data.xlsx")
df <- xts(new_data[, -which(names(new_data) == "Data")],  order.by = new_data$Data)
df<- df[,-64]
pred <- data.frame(pred = 0)
df1 <- cbind(df, pred)
col_nam <- colnames(df)
col_nam_new <- paste0("beta_", col_nam)
col_nam_new_df1 <- paste0("PIP_", col_nam)
col_nam_new_df1 <- col_nam_new_df1[-1]
new_col <- data.frame(setNames(replicate(62, numeric()), col_nam_new_df1))
df1 <- cbind(df1, new_col)
for (i in 150:848) 
  {
    j=i+629
    estym <- df[i:j,1:63]
    # outliery ze wszystkich 1:63
    quartiles <- apply(estym[, 1:63], 2, quantile, probs = c(0.25, 0.75)) 
    inter_quartiles <- quartiles[2,] - quartiles[1,]
    u_limit <- quartiles[2,]+1.5 * inter_quartiles
    d_limit <- quartiles[1,]-1.5 * inter_quartiles
    # outliery ze wszystkich 1:63
    for (col in 1:63) 
      {
        estym[,col] <- ifelse(estym[,col] > u_limit[col], u_limit[col], estym[, col])
        estym[,col] <- ifelse(estym[,col] < d_limit[col], d_limit[col], estym[, col])
      }
    BMSres <- bms(estym, nmodel=10000000)
    test <- df[(i+1):(j+1),1:63]
    quartiles <- apply(test[, 1:63], 2, quantile, probs = c(0.25, 0.75)) 
    inter_quartiles <- quartiles[2,] - quartiles[1,]
    u_limit <- quartiles[2,]+1.5 * inter_quartiles
    d_limit <- quartiles[1,]-1.5 * inter_quartiles
    for (col in 2:63) 
      { 
        test[,col] <- ifelse(test[,col] > u_limit[col], u_limit[col], test[, col])
        test[,col] <- ifelse(test[,col] < d_limit[col], d_limit[col], test[, col])
      }
    wektor_test <- test[630,2:63]
    predBMS <- predict(BMSres, exact=TRUE, newdata=wektor_test)
    df1[j+1,64] <- predBMS
    show(i)
    est <- estimates.bma(BMSres, order.by.pip = FALSE)
    PIP <- t(est[,1])
    df1[j+1,65:126] <- PIP
  }