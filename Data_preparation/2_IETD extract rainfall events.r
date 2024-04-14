library(dplyr)
library(tidyr)
library(lubridate)
devtools::install_github("lfduquey/IETD")
library(IETD)
library(ggplot2)
library(scales)
library(tidyverse)

# Read processed data, for example rainfallkp_2020.txt
file_path <- "/content/rainfallkp_2020.txt"
data <- read.table(file_path, header = TRUE, sep = "\t")
data <- data[, 1:2]
print(data)

# Check the number of rows and columns
num_rows <- dim(data)[1]
num_cols <- dim(data)[2]
cat("Number of rows:", num_rows, "\n")
cat("Number of columns:", num_cols, "\n")

# Unit of raw data is in 0.1 mm, now convert to 1 mm
data$rainfall <- data$rainfall/10
Time_series<-data
Time_series

# Create an empty dataframe to store the rows with interpolation
interpolated_rows <- data.frame()

# Check for missing values in the rainfall column
if (sum(is.na(Time_series$rainfall)) > 0) {
  # Get the indices of NA values
  na_indices <- which(is.na(Time_series$rainfall))

  # Loop through the NA indices
  for (i in na_indices) {
    # Check if there is at least one non-NA value before and after the current NA value
    if (!is.na(Time_series$rainfall[i-1]) && !is.na(Time_series$rainfall[i+1])) {
      # Interpolate the missing value using linear interpolation
      Time_series$rainfall[i] <- (Time_series$rainfall[i-1] + Time_series$rainfall[i+1]) / 2
      print(Time_series$rainfall[i])

      # Append the row to the interpolated_rows dataframe
      interpolated_rows <- rbind(interpolated_rows, Time_series[i, ])
    }
  }
}

# Convert the time column to POSIXct format
Time_series$DateAndTime <- as.POSIXct(Time_series$DateAndTime, format="%Y-%m-%d %H:%M:%S")

# Sort the data frame by time
Time_series <- Time_series[order(Time_series$DateAndTime),]

# Print the rows with interpolation
print(interpolated_rows)

# Replace NaN with 0
Time_series$rainfall[is.na(Time_series$rainfall)] <- 0
# Check if there is NaN
na_rows <- Time_series[is.na(Time_series$rainfall), ]
na_rows
# Check for missing values
if (length(which(is.na(Time_series[,2]))) >= 1) {
  stop("There are missing values!")
}

ggplot(Time_series,aes(x=DateAndTime,y=rainfall)) +
theme_bw()+
geom_bar(stat = 'identity',colour="black",lwd=1, fill="gray")+
scale_x_datetime(labels = date_format("%Y-%m-%d %H"))+
ylab("Rainfall depth [mm]")

# IETD method 1 - CVA (Coefficient of variation analysis), REFERENCE:https://cran.r-project.org/web/packages/IETD/IETD.pdf
Results_CVA<-CVA(Time_series,100)
IETD_CVA<-Results_CVA$EITD
Results_CVA$Figure

# IETD method 2 - AutoA (Autocorrelation analysis), REFERENCE:https://cran.r-project.org/web/packages/IETD/IETD.pdf
Results_AutoA<-AutoA(Time_series,100)
IETD_AutoA<-Results_AutoA$EITD
Results_AutoA$Figure

# IETD method 3 - AAEA (Average annual number of events analysis), REFERENCE:https://cran.r-project.org/web/packages/IETD/IETD.pdf
Results_AAEA<-AAEA(Time_series,100)
IETD_AAEA<-Results_AAEA$EITD
Results_AAEA$Figure

# Extract rainfall with IETD = 3h, rainfall depth threshold = 0.5(default)
Rainfall <- drawre(Time_series,IETD=3,Thres=0.5)
Rainfall.Characteristics <-Rainfall$Rainfall_Characteristics
Rainfall.Events<-Rainfall$Rainfall_Events

file_path <- "/content/2018_2023/2020"
dir.create(file_path)
# Save rainfall characteristics in csv
write.csv(Rainfall.Characteristics, file = paste0(file_path, "/events_2020.csv"), row.names = TRUE)
# Save extracted rainfall events in csv
for (i in seq_along(Rainfall.Events)) {
  df <- Rainfall.Events[[i]]
  filename <- paste0("event", i, ".csv")
  write.csv(df, file.path(file_path, filename), row.names = FALSE)
}


