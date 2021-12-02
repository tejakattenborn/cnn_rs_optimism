
# libraries + path -----------------------------------------------------------

pkgs <- c("keras", "tidyverse", "tibble", "tensorflow", "reticulate", "doParallel", "foreach", "abind", "raster")
sapply(pkgs, require, character.only = TRUE)

# gpu1 <- tf$config$experimental$get_visible_devices('GPU')[[1]]
# gpu2 <- tf$config$experimental$get_visible_devices('GPU')[[2]]
# tf$config$experimental$set_memory_growth(device = gpu1, enable = TRUE)
# tf$config$experimental$set_memory_growth(device = gpu2, enable = TRUE)


tf$compat$v1$set_random_seed(as.integer(28))
# when runing multi gpu model
# strategy <- tf$distribute$MirroredStrategy()
# strategy$num_replicas_in_sync


source("00_helper_functions.R")


# Load Data ---------------------------------------------------------------

fold <- 5


load(paste0("dataSplit_fold", fold, ".RData"))

pathImg <- list.files("02_pipeline/img", pattern = ".png", recursive = T, full.names = T)
pathMsk <- list.files("02_pipeline/msk", pattern = ".png", recursive = T, full.names = T)


# Parameters --------------------------------------------------------------

tilesize   <- 256L
noBands    <- 3L
noSpec     <- 12L
all_sites  <- unique(substr(pathImg, 36, 41))

# Model evaluation --------------------------------------------------------

## accuracy testsac
pb <- txtProgressBar(min = 0, max = 12, style = 3)
for (i in 1:12) {
  accuracies_testsac = list()

  model_ID <- i
  
  aug_rad  <- dataSplit[[model_ID]]$aug_rad
  aug_geo  <- dataSplit[[model_ID]]$aug_geo
  
  split <- dataSplit[[model_ID]]
  
  data_testsac    <- tibble(img = pathImg[split$data$test_sac == 1], msk = pathMsk[split$data$test_sac == 1])

  dataset_testsac   <- createDataset(data_testsac, aug_rad = aug_rad, aug_geo = aug_geo, train = FALSE,
                                     batch = 1L, epochs = 1L, shuffle = FALSE, datasetSize = 1L)

  
  checkpoint_dir <- paste0("02_pipeline/run/fold", fold, "/ID", model_ID, "/checkpoints")
  
  with(tf$device('/cpu:0'), {
    model <- loadModel(checkpoint_dir, compile = F)
    pred_testsac   <- predict(model, dataset_testsac)
  })

  pred_testsac_files   <- substr(data_testsac$img, 28, 50)

  # save(list = c("pred_testsac", "pred_testsac_files"), file = paste0("/media/sysgen/Volume/Felix/UAVforSAT/CNN_SAC/ID", model_ID, "_pred_testsac.RData"))

  pred_testsac   <- decodeOneHot(pred_testsac)

  cl <- makeCluster(19)
  registerDoParallel(cl)
  msks_testsac = foreach(k = 1:nrow(data_testsac), .inorder = T, .packages = "raster") %dopar% {
    r = raster(data_testsac$msk[k])
    as.array(r)
  }
  stopCluster(cl)
  
  msks_testsac = do.call(abind, list(msks_testsac))
  msks_testsac = aperm(msks_testsac, c(3,1,2))

  
  predVec = as.vector(pred_testsac)+1
  obsVec  = as.vector(msks_testsac)
  u       = sort(union(obsVec, predVec))
  conmat  = caret::confusionMatrix(data = factor(predVec, u), reference = factor(obsVec, u))

  accuracies_testsac$names     = model_ID
  accuracies_testsac$conMat    = conmat$table
  accuracies_testsac$accuracy  = conmat$overall["Accuracy"]
  accuracies_testsac$kappa     = conmat$overall["Kappa"]
  accuracies_testsac$precision = conmat$byClass[,"Precision"]
  accuracies_testsac$recall    = conmat$byClass[,"Recall"]
  accuracies_testsac$F1Score   = conmat$byClass[,"F1"]

  testsac_F1_site_species <- matrix(data = NA, nrow = length(all_sites), ncol = 12, dimnames = list(all_sites, paste0("Class: ", 1:12)))
  testsac_Acc_site        <- matrix(data = NA, nrow = length(all_sites), ncol = 1, dimnames = list(all_sites, "Accuracy"))
  site_files              <- substr(pathImg[split$data$test_sac == 1], 36, 41)
  
  for(l in 1:length(unique(site_files))) {
    idx <- which(site_files == unique(site_files)[l])

    predVec <- as.vector(pred_testsac[idx,,])+1
    obsVec  <- as.vector(msks_testsac[idx,,])
    u       <- sort(union(obsVec, predVec))
    conmat  <- caret::confusionMatrix(data = factor(predVec, u), reference = factor(obsVec, u))

    mIdx <- match(unique(site_files)[l], rownames(testsac_Acc_site))
    testsac_Acc_site[mIdx, "Accuracy"] <- conmat$overall["Accuracy"]
    cIdx <- match(names(conmat$byClass[,"F1"]), colnames(testsac_F1_site_species))
    testsac_F1_site_species[mIdx, cIdx]    <- conmat$byClass[,"F1"]
  }
  
  accuracies_testsac$F1_site_species <- testsac_F1_site_species
  accuracies_testsac$accuracy_site   <- testsac_Acc_site
  
  
  save(list = c("accuracies_testsac"), file = paste0("02_pipeline/run/fold", fold, "/ID", model_ID, "/evaluation_testsac_ID", model_ID, ".RData"))
  rm(msks_testsac, conmat, dataset_testsac, data_testsac, pred_testsac, 
     accuracies_testsac, testsac_Acc_site, testsac_F1_site_species, obsVec, predVec)
  
  raster::removeTmpFiles(h=0); gc()
  
  setTxtProgressBar(pb, i)
}

## accuracy testnosac
pb <- txtProgressBar(min = 0, max = 12, style = 3)
for (i in 1:12) {
  accuracies_testnosac = list()
  
  model_ID <- i
  
  aug_rad  <- dataSplit[[model_ID]]$aug_rad
  aug_geo  <- dataSplit[[model_ID]]$aug_geo
  
  split <- dataSplit[[model_ID]]
  
  data_testnosac  <- tibble(img = pathImg[split$data$test_nosac == 1], msk = pathMsk[split$data$test_nosac == 1])
  
  dataset_testnosac <- createDataset(data_testnosac, aug_rad = aug_rad, aug_geo = aug_geo, train = FALSE,
                                     batch = 1L, epochs = 1L, shuffle = FALSE, datasetSize = 1L)
  
  
  checkpoint_dir <- paste0("02_pipeline/run/fold", fold, "/ID", model_ID, "/checkpoints")  
  
  with(tf$device('/cpu:0'), {
    model <- loadModel(checkpoint_dir, compile = F)
    pred_testnosac <- predict(model, dataset_testnosac)
  })
  
  pred_testnosac_files <- substr(data_testnosac$img, 28, 50)
  
  # save(list = c("pred_testnosac", "pred_testnosac_files"), file = paste0("/media/sysgen/Volume/Felix/UAVforSAT/CNN_SAC/ID", model_ID, "_pred_testnosac.RData"))
  
  pred_testnosac <- decodeOneHot(pred_testnosac)
  
  cl <- makeCluster(19)
  registerDoParallel(cl)
  msks_testnosac = foreach(k = 1:dim(pred_testnosac)[1], .inorder = T, .packages = "raster") %dopar% {
    r = raster(data_testnosac$msk[k])
    as.array(r)
  }
  stopCluster(cl)
  
  msks_testnosac = do.call(abind, list(msks_testnosac))
  msks_testnosac = aperm(msks_testnosac, c(3,1,2))
  

  predVec = as.vector(pred_testnosac)+1
  obsVec  = as.vector(msks_testnosac)
  u       = sort(union(obsVec, predVec))
  conmat  = caret::confusionMatrix(data = factor(predVec, u), reference = factor(obsVec, u))

  accuracies_testnosac$names     = model_ID
  accuracies_testnosac$conMat    = conmat$table
  accuracies_testnosac$accuracy  = conmat$overall["Accuracy"]
  accuracies_testnosac$kappa     = conmat$overall["Kappa"]
  accuracies_testnosac$precision = conmat$byClass[,"Precision"]
  accuracies_testnosac$recall    = conmat$byClass[,"Recall"]
  accuracies_testnosac$F1Score   = conmat$byClass[,"F1"]

  testnosac_F1_site_species <- matrix(data = NA, nrow = length(all_sites), ncol = 12, dimnames = list(all_sites, paste0("Class: ", 1:12)))
  testnosac_Acc_site        <- matrix(data = NA, nrow = length(all_sites), ncol = 1, dimnames = list(all_sites, "Accuracy"))
  site_files                <- substr(pathImg[split$data$test_nosac == 1], 36, 41)
  
  for(l in 1:length(unique(site_files))) {
    idx <- which(site_files == unique(site_files)[l])
    
    predVec <- as.vector(pred_testnosac[idx,,])+1
    obsVec  <- as.vector(msks_testnosac[idx,,])
    u       <- sort(union(obsVec, predVec))
    conmat  <- caret::confusionMatrix(data = factor(predVec, u), reference = factor(obsVec, u))
    
    mIdx <- match(unique(site_files)[l], rownames(testnosac_Acc_site))
    testnosac_Acc_site[mIdx, "Accuracy"] <- conmat$overall["Accuracy"]
    cIdx <- match(names(conmat$byClass[,"F1"]), colnames(testnosac_F1_site_species))
    testnosac_F1_site_species[mIdx, cIdx]    <- conmat$byClass[,"F1"]
  }
  
  accuracies_testnosac$F1_site_species <- testnosac_F1_site_species
  accuracies_testnosac$accuracy_site   <- testnosac_Acc_site
  
  
  save(list = c("accuracies_testnosac"), file = paste0("02_pipeline/run/fold", fold, "/ID", model_ID, "/evaluation_testnosac_ID", model_ID, ".RData"))
  rm(msks_testnosac, conmat, dataset_testnosac, data_testnosac, pred_testnosac,
     accuracies_testnosac, testnosac_Acc_site, testnosac_F1_site_species, obsVec, predVec)
  
  raster::removeTmpFiles(h=0); gc()
  
  setTxtProgressBar(pb, i)
}


for(i in 1:12) {
  model_ID = i
  load(paste0("02_pipeline/run/fold", fold, "/ID", model_ID, "/evaluation_testnosac_ID", model_ID, ".RData"))
  load(paste0("02_pipeline/run/fold", fold, "/ID", model_ID, "/evaluation_testsac_ID", model_ID, ".RData"))
  save(list = c("accuracies_testnosac", "accuracies_testsac"),
       file = paste0("02_pipeline/run/fold", fold, "/ID", model_ID, "/evaluation_ID", model_ID, ".RData"))
}

# evaluate(object = model, x = dataset_testsac)
# evaluate(object = model, x = dataset_testnosac)

tab <- read.csv("CNN_SAC-Sheet1.csv", stringsAsFactors = F)
# tab$augmentation_code <- factor(tab$augmentation_code, levels = c("none", "rad", "geo", "both"))
tab$augmentation_code <- factor(tab$augmentation_code, levels = c("both", "geo", "none", "rad"))
# levels(tab$augmentation_code) <- c("none", "rad", "geo", "both")

testsac <- testnosac <- list(precision = matrix(NA, nrow = 12, ncol = 12, dimnames = list(paste0("ID", 1:12), paste0("Class: ", 1:12))),
                             recall = matrix(NA, nrow = 12, ncol = 12, dimnames = list(paste0("ID", 1:12), paste0("Class: ", 1:12))),
                             F1Score = matrix(NA, nrow = 12, ncol = 12, dimnames = list(paste0("ID", 1:12), paste0("Class: ", 1:12))))

for (i in 1:12) {
  
  load(paste0("02_pipeline/run/ID", i, "/evaluation_ID", i, ".RData"))
  
  testsac$conmat[[i]]                                       <- accuracies_testsac$conMat
  testsac$accuracy[i]                                       <- accuracies_testsac$accuracy
  testsac$kappa[i]                                          <- accuracies_testsac$kappa
  testsac$precision[i, names(accuracies_testsac$precision)] <- accuracies_testsac$precision
  testsac$recall[i, names(accuracies_testsac$recall)]       <- accuracies_testsac$recall
  testsac$F1Score[i, names(accuracies_testsac$F1Score)]     <- accuracies_testsac$F1Score
  
  testnosac$conmat[[i]]                                         <- accuracies_testnosac$conMat
  testnosac$accuracy[i]                                         <- accuracies_testnosac$accuracy
  testnosac$kappa[i]                                            <- accuracies_testnosac$kappa
  testnosac$precision[i, names(accuracies_testnosac$precision)] <- accuracies_testnosac$precision
  testnosac$recall[i, names(accuracies_testnosac$recall)]       <- accuracies_testnosac$recall
  testnosac$F1Score[i, names(accuracies_testnosac$F1Score)]     <- accuracies_testnosac$F1Score
  
}

# plot(testsac$accuracy, ylim = c(0,1))
# points(testnosac$accuracy)

boxplot(accuracies_testsac$perSite[,2], ylim = c(0,1))
points(accuracies_testsac$perSite[,2] ~ rep(1, length(accuracies_testsac$perSite[,2])), ylim = c(0,1))

ord <- c(3,2,4,1, 7,6,8,5, 11,10,12,9)
pos = c(1:4, 6:9, 11:14)

F1_testsac <- F1_testnosac <- matrix(NA, nrow = 47, ncol = 12)
for (i in 1:12) {
  load(paste0("02_pipeline/run/ID", i, "/evaluation_ID", i, ".RData"))
  F1_testsac[,i]   <- accuracies_testsac$perSite[,"F1"]
  F1_testnosac[,i] <- accuracies_testnosac$perSite[,"F1"]
}

par(mfrow = c(1,2), mar = c(3,2.5,1,0.5))
boxplot(F1_testsac[,ord], at = pos, ylim = c(0,1), names = tab$augmentation_code[ord], las = 2); grid()
for (i in 1:12) points(rep(pos[i], 47), F1_testsac[,ord[i]], col = alpha("black", 0.5))
boxplot(F1_testnosac[,ord], at = pos, ylim = c(0,1), names = tab$augmentation_code[ord], las = 2); grid()
for (i in 1:12) points(rep(pos[i], 47), F1_testnosac[,ord[i]], col = alpha("black", 0.5))


par(mfrow = c(1,2), mar = c(3,2.5,1,0.5))
boxplot(t(testsac$F1Score)[,ord], at = pos, ylim = c(0,1), names = tab$augmentation_code[ord], las = 2); grid()
for (i in 1:12) points(rep(pos[i], 12), testsac$F1Score[ord[i],], col = alpha("black", 0.5))
boxplot(t(testnosac$F1Score)[,ord], at = pos, ylim = c(0,1), names = tab$augmentation_code[ord], las = 2); grid()
for (i in 1:12) points(rep(pos[i], 12), testnosac$F1Score[ord[i],], col = alpha("black", 0.5))



sac <- data.frame(code = tab$augmentation_code,
                  acc = testsac$accuracy,
                  pre = rowMeans(testsac$precision, na.rm = T),
                  rec = rowMeans(testsac$recall, na.rm = T),
                  F1 = apply(testsac$F1Score, 1, median, na.rm = T), #rowMeans(testsac$F1Score, na.rm = T),
                  cohort = tab$model_cohort)
nosac <- data.frame(code = tab$augmentation_code,
                    acc = testnosac$accuracy,
                    pre = rowMeans(testnosac$precision, na.rm = T),
                    rec = rowMeans(testnosac$recall, na.rm = T),
                    F1 = apply(testnosac$F1Score, 1, median, na.rm = T), #rowMeans(testnosac$F1Score, na.rm = T),
                    cohort = tab$model_cohort)
save(list = c("sac", "nosac"), file = "accuracy_assessment.RData")

gg_acc_sac <- ggplot(sac, aes(x = cohort, y = acc)) +
  geom_col(
    aes(fill = code),
    position = position_dodge(0.8), width = 0.6,
    show.legend = F
  ) +
  ggtitle("SAC") +
  scale_y_continuous("Accuracy", limits = c(0,1)) +
  scale_x_discrete("Model cohort") +
  scale_fill_brewer(palette="Spectral") +
  theme_minimal()

gg_acc_nosac <- ggplot(nosac, aes(x = cohort, y = acc)) +
  geom_col(
    aes(fill = code),
    position = position_dodge(0.8), width = 0.6,
    show.legend = T
  ) +
  ggtitle("no SAC") +
  scale_y_continuous(limits = c(0,1), labels = NULL, name = NULL) +
  scale_x_discrete("Model cohort") +
  scale_fill_brewer(name = "Augmentation", palette="Spectral") +
  theme_minimal()



gg_F1_sac <- ggplot(sac, aes(x = cohort, y = F1)) +
  geom_col(
    aes(fill = code),
    position = position_dodge(0.8), width = 0.6,
    show.legend = F
  ) +
  ggtitle("SAC") +
  scale_y_continuous("F1Score", limits = c(0,1)) +
  scale_x_discrete("Model cohort") +
  scale_fill_brewer(palette="Spectral") +
  theme_minimal()

gg_F1_nosac <- ggplot(nosac, aes(x = cohort, y = F1)) +
  geom_col(
    aes(fill = code),
    position = position_dodge(0.8), width = 0.6,
    show.legend = T
  ) +
  ggtitle("no SAC") +
  scale_y_continuous(limits = c(0,1), labels = NULL, name = NULL) +
  scale_x_discrete("Model cohort") +
  scale_fill_brewer(name = "Augmentation", palette="Spectral") +
  theme_minimal()




gridExtra::grid.arrange(gg_acc_sac, gg_acc_nosac, ncol = 2, widths = c(1,1.25))
gridExtra::grid.arrange(gg_F1_sac, gg_F1_nosac, ncol = 2, widths = c(1,1.25))
