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
df <- read_excel("C:\\Users\\Olle de Jong\\Desktop\\data_mim_mic_3cs_DMN.xlsx")
df <- subset(df, select = -...1)

df <- subset(df, batch %in% c('batch4', 'batch5', 'batch5b', 'batch6'))

df$batch <- as.numeric(gsub("\\D", "", df$batch))
df$batch <- as.factor(df$batch)

df$genotype <- as.factor(df$genotype)
df$animal_id <- as.factor(df$animal_id)
df$transmitter <- as.factor(df$transmitter)



#_______________ Plotting of relevant figures __________________________

save_plots <- function(g1, output_path, experiment_name, event_type, max_freq) {
  pdf(sprintf("%s/conn_diff_%s_%s.pdf", output_path, experiment_name, event_type), width = 10, height = 5)
  plt_diff <- plot_diff(
    g1, view = "freqs", comp = list(genotype = c("DRD2-WT", "DRD2-KO")),
    n.grid = max_freq, 
    xlab = "Frequency (Hz)", ylab = "Estimated difference in connectivity / MIM",
    col = "#17A398", col.diff = "#e74c3c",
    main = sprintf("Connectivity / MIM difference between DRD2-WT and DRD2-KO animals"),
    lwd = 2.5
  )
  dev.off()
  pdf(sprintf("%s/est_conn_%s_%s.pdf", output_path, experiment_name, event_type), width = 10, height = 8)
  plot_smooth(
    g1, view = "freqs", rug = F, plot_all = "genotype", 
    main = sprintf("Modelled Multivariate Interaction Measure"),
    xlab = "Frequency (Hz)", ylab = "Estimated connectivity / MIM",
    col = c('#EB5E55', '#419D78'), lwd = 2.5
  )
  dev.off()
}

#_______________ Making and exporting GAMs _____________________________
# Main function
make_gams <- function(df, interaction_type, gam_formula, max_freq, experiment_name,
                      output_path) {
  # Subset data
  sub <- df[df$behaviour == interaction_type, ]
  
  # Make GAM
  g1 <- gam(gam_formula,
            family = gaussian(),
            method = 'ML',
            data=sub)
  
  # Make predictions
  predAct = get_predictions(g1,
                            cond = list(genotype = c("DRD2-WT", "DRD2-KO"),
                                        freqs=seq(0,max_freq,length=max_freq+1)),
                            print.summary = TRUE)
  predAct$CI<-predAct$CI/1.96
  
  # Export Predictions
  write_xlsx(predAct, sprintf("%s/pred_%s_%s.xlsx", output_path, experiment_name, interaction_type))
  
  # Summary Sink
  sink(sprintf("%s/summary_%s_%s.txt", output_path, experiment_name, interaction_type))
  summary(g1)
  save_plots(g1, output_path, experiment_name, interaction_type, max_freq)
  sink()
  
  print("Done!")
  return(g1)
}

#_______________________________________________________________________________
# ____________ GAMs for all behaviors (Adjust the gam_formula) _________________
#_______________________________________________________________________________

# change to your liking
experiment_name = '3_chamber_sociability'
interaction_types = c("social_cup", "non-social_cup")
max_freq <- 100
output_path <- "C:/Users/Olle de Jong/Documents/MSc Biology/rp2/rp2_data/3C_sociability/output/gams/MIM_DMN/batches4-6"

# Define gma formula
gam_formula <- mim ~ s(freqs, by=genotype, k=80) +
  genotype +
  s(transmitter, bs='fs', m=1) +
  s(animal_id, bs='fs', m=1) +
  s(batch, bs="fs", m=1)

# Loop over behaviors
for (type in interaction_types) {
  sprintf("Working on %s events", type)
  g1 <- make_gams(df, type, as.formula(gam_formula), max_freq, experiment_name, output_path)
}
