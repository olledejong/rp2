library(readxl)
library(mgcv) # creating GAM
library(itsadug) #GAM analysis
library(ggplot2)
library(ggpubr)
library(gplots)
library(plotrix)
library(extrafont)
library(dplyr)
library(tidyr)
library(DHARMa)
library(writexl)


#________________________ Variables ___________________________________________

chan_names <- list(
  'OFC_R' = 'Right Orbitofrontal Cortex',
  'OFC_L' = 'Left Orbitofrontal Cortex',
  'CG' = 'Cingulate Cortex',
  'STR_R' = 'Right Striatum',
  'S1_L' = 'Left Somatosensory',
  'S1_R' = 'Right Somatosensory',
  'V1_R' = 'Right Visual'
)

#________________________ Data cleaning _______________________________________
# Load data
df <- read_excel("C:\\Users\\Olle de Jong\\Desktop\\dat_pref.xlsx")
df <- subset(df, select = -...1)
df <- subset(df, select = -psd)

df$batch <- as.numeric(gsub("\\D", "", df$batch))
df$batch <- as.factor(df$batch)

df$genotype <- as.factor(df$genotype)
df$subject_id <- as.factor(df$subject_id)

# Make log cols
df <- df %>% rename(psd = `psd (norm)`)

#_______________ Plotting of relevant figures __________________________

save_plots <- function(g1, output_path, channel, experiment_name, event_type) {
  pdf(sprintf("%s/power_diff_%s_%s_%s.pdf", output_path, experiment_name, event_type, channel), width = 10, height = 5)
  plt_diff <- plot_diff(
    g1, view = "freq", comp = list(genotype = c("DRD2-WT", "DRD2-KO")),
    n.grid = max_freq, 
    xlab = "Frequency (Hz)", ylab = "Estimated difference in power",
    col = "#17A398", col.diff = "#e74c3c",
    main = sprintf("Power difference in the %s between DRD2-WT and DRD2-KO\nanimals during %s events", chan_names[[channel]], gsub("_", " ", event_type)),
    lwd = 2.5
  )
  dev.off()
  pdf(sprintf("%s/est_power_%s_%s_%s.pdf", output_path, experiment_name, event_type, channel), width = 10, height = 8)
  plot_smooth(
    g1, view = "freq", rug = F, plot_all = "genotype", 
    main = sprintf("Modelled power data (%s, %s events)", chan_names[[channel]], gsub("_", " ", event_type)),
    xlab = "Frequency (Hz)", ylab = "Estimated power (dB/Hz)",
    col = c('#EB5E55', '#419D78'), lwd = 2.5
  )
  dev.off()
}

#_______________ Making and exporting GAMs _____________________________
# Main function
make_gams <- function(df, event_type, channel, gam_formula, max_freq, experiment_name,
                      output_path) {
  # Subset data
  sub <- df[df$interaction_kind == event_type, ]
  
  # Make GAM
  g1 <- gam(gam_formula,
            family = Gamma(link = "log"),
            data=sub)
  
  # Make predictions
  predAct = get_predictions(g1, 
                            cond = list(genotype = c("DRD2-WT", "DRD2-KO"), 
                                        freq=seq(0,max_freq,length=max_freq+1)),
                            print.summary = TRUE)
  predAct$CI<-predAct$CI/1.96
  
  # Export Predictions
  write_xlsx(predAct, sprintf(
    "%s/pred_%s_%s_%s.xlsx", output_path, experiment_name, event_type, channel)
  )
  
  # Summary Sink
  sink(sprintf("%s/summary_%s_%s_%s.txt", output_path, experiment_name, event_type, channel))
  summary(g1)
  save_plots(g1, output_path, channel, experiment_name, event_type)
  sink()
  
  print("Done!")
  return(g1)
}

#_______________________________________________________________________________
# ____________ GAMs for all behaviors (Adjust the gam_formula) _________________
#_______________________________________________________________________________

# change to your liking
experiment_name <- '3_chamber_preference'
all_events <- c("familiar_cup", "novel_cup")
max_freq <- 100
output_path <- 'C:/Users/Olle de Jong/Documents/MSc Biology/rp2/rp2_data/3C_preference/output/gams'

# Define gma formula
gam_formula <- psd ~ s(freq, by=genotype, k=40) +
  genotype +
  s(subject_id, bs='fs', m=1) +
  s(batch, bs="fs", m=1)

# Loop over behaviors
for (event_type in all_events) {
  sprintf("Working on %s events", event_type)
  for (channel in unique(df$channel)) {
    sprintf("Working with data from channel %s", channel)
    channel_df <- df[df$channel == channel, ] 
    
    # s(subject_id, bs='fs', m=1)
    #
    # This part of the formula specifies a smooth term
    # for the 'subject_id' variable using a fixed-slope spline. Including smooth 
    # terms for subject IDs can help capture any non-linear variation in the response
    # variable 'psd' associated with individual subjects.
    
    # s(batch, bs="fs", m=1)
    #
    # Similarly, this part of the formula specifies a smooth
    # term for the 'batch' variable using a fixed-slope spline. Including smooth terms
    # for batches can help account for any non-linear variation in 'psd' associated 
    # with different batches of experiments.
    
    gam_formula <- psd ~ s(freq, by=genotype, k=40) +
      genotype +
      s(subject_id, bs='fs', m=1) +
      s(batch, bs="fs", m=1)
    
    g1 <- make_gams(channel_df, event_type, channel, as.formula(gam_formula), max_freq, experiment_name,
                    output_path)
  }
}
