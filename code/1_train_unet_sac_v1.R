
# libraries + path -----------------------------------------------------------

pkgs <- c("keras", "tidyverse", "tibble", "tensorflow")
sapply(pkgs, require, character.only = TRUE)

# gpu1 <- tf$config$experimental$get_visible_devices('GPU')[[1]]
# gpu2 <- tf$config$experimental$get_visible_devices('GPU')[[2]]
# tf$config$experimental$set_memory_growth(device = gpu1, enable = TRUE)
# tf$config$experimental$set_memory_growth(device = gpu2, enable = TRUE)

tf$compat$v1$set_random_seed(as.integer(28))
# when runing multi gpu model
strategy <- tf$distribute$MirroredStrategy()
strategy$num_replicas_in_sync


# mixedPrecision <- tf$keras$mixed_precision$experimental
# policy <- mixedPrecision$Policy('mixed_float16')
# mixedPrecision$set_policy(policy)
# policy$compute_dtype  # datatype of tensors
# policy$variable_dtype


source("00_helper_functions.R")

# Load Data ---------------------------------------------------------------

load("dataSplit.RData")

pathImg <- list.files("02_pipeline/img", pattern = ".png", recursive = T, full.names = T)
pathMsk <- list.files("02_pipeline/msk", pattern = ".png", recursive = T, full.names = T)


# Parameters --------------------------------------------------------------

model_ID <- 12

tilesize   <- 256L
noEpochs   <- 60L
noBands    <- 3L
noSpec     <- 12L
batch_size <- 28L
aug_rad  <- dataSplit[[model_ID]]$aug_rad
aug_geo  <- dataSplit[[model_ID]]$aug_geo

outDir <- paste0("02_pipeline/run/ID", model_ID)
dir.create(outDir, recursive = TRUE)


# Data split --------------------------------------------------------------

split <- dataSplit[[model_ID]]

data_testsac    <- tibble(img = pathImg[split$data$test_sac == 1], msk = pathMsk[split$data$test_sac == 1])
data_testnosac  <- tibble(img = pathImg[split$data$test_nosac == 1], msk = pathMsk[split$data$test_nosac == 1])
data_train      <- tibble(img = pathImg[split$data$train == 1], msk = pathMsk[split$data$train == 1])
data_valid      <- tibble(img = pathImg[split$data$valid == 1], msk = pathMsk[split$data$valid == 1])

dataset_size <- length(pathImg[split$data$train == 1])

dataset_train <- createDataset(data_train, aug_rad = aug_rad, aug_geo = aug_geo, train = TRUE,
                               batch = batch_size, epochs = noEpochs, datasetSize = dataset_size)
dataset_valid <- createDataset(data_valid, aug_rad = FALSE, aug_geo = FALSE, train = FALSE,
                               batch = batch_size, epochs = noEpochs)



# Define U-net CNN --------------------------------------------------------

# U-net code is based on the following github example:
# https://github.com/rstudio/keras/blob/master/vignettes/examples/unet_linux.R

getUnet <- function(inputShape = c(tilesize, tilesize, noBands),
                    numClasses = noSpec) {
  
  # create blocks
  inputs <- layer_input(shape = inputShape)
  
  down1 <- inputs %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") 
  down1_pool <- down1 %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
  
  down2 <- down1_pool %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") 
  down2_pool <- down2 %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
  
  down3 <- down2_pool %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") 
  down3_pool <- down3 %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
  
  down4 <- down3_pool %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") 
  down4_pool <- down4 %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
  
  center <- down4_pool %>%
    layer_conv_2d(filters = 1024, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 1024, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") 
  
  up4 <- center %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down4, .), axis = 3)} %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  
  up3 <- up4 %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down3, .), axis = 3)} %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  
  up2 <- up3 %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down2, .), axis = 3)} %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  
  up1 <- up2 %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down1, .), axis = 3)} %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  
  classify <- layer_conv_2d(up1,
                            filters = numClasses, 
                            kernel_size = c(1, 1),
                            activation = "softmax")
  
  # build model
  model <- keras_model(
    inputs = inputs,
    outputs = classify
  )
  
  # model %>% compile(
  #   optimizer = tf$keras$optimizers$RMSprop(0.0001),
  #   # loss = "categorical_crossentropy",
  #   loss = weightedCategoricalCrossentropy,
  #   metrics = c("accuracy", "categorical_crossentropy")
  #   # metrics = custom_metric("wcce", wcce_loss)
  # )
  
  return(model)
}


# multiple gpu (custom loss/metric not supported)
with(strategy$scope(), {
  model <- getUnet()
})


model %>% compile(
  optimizer = tf$keras$optimizers$RMSprop(0.0001),
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)


# Train U-net -------------------------------------------------------------

checkpoint_dir <- paste0(outDir, "/checkpoints/")
# unlink(checkpoint_dir, recursive = TRUE)
dir.create(checkpoint_dir, recursive = TRUE)
filepath <- file.path(checkpoint_dir, "weights.{epoch:02d}-{val_loss:.5f}.hdf5")

cp_callback <- callback_model_checkpoint(filepath = filepath,
                                         monitor = "val_loss",
                                         save_weights_only = FALSE,
                                         save_best_only = FALSE,
                                         verbose = 1,
                                         mode = "auto",
                                         save_freq = "epoch")

history <- model %>% fit(x = dataset_train,
                         epochs = noEpochs,
                         steps_per_epoch = dataset_size/batch_size,
                         callbacks = list(cp_callback,
                                          callback_terminate_on_naan()),
                         validation_data = dataset_valid)

pdf(paste0(outDir, "/training.pdf"))
plot(history)
dev.off()


metrics <- history$metrics
save(metrics, file = paste0(outDir, "/history_metrics.RData"))


# Model evaluation --------------------------------------------------------


# dataset_testsac   <- createDataset(data_testsac, aug_rad = FALSE, aug_geo = FALSE, train = FALSE,
#                                    batch = 1L, epochs = 1L)
# dataset_testnosac <- createDataset(data_testnosac, aug_rad = FALSE, aug_geo = FALSE, train = FALSE,
#                                    batch = 1L, epochs = 1L)
# 
# with(tf$device('/cpu:0'), {
#   model <- loadModel(checkpoint_dir, compile = T)
# })
# 
# evaluate(object = model, x = dataset_testsac)
# evaluate(object = model, x = dataset_testnosac)



# pred_testsac <- predict(model, dataset_testsac)
# pred_testsac <- decodeOneHot(pred_testsac)
# 
# pred_testnosac <- predict(model, dataset_testnosac)
# pred_testnosac <- decodeOneHot(pred_testnosac)
# 
# library(doParallel)
# library(foreach)
# library(abind)
# cl <- makeCluster(19)
# registerDoParallel(cl)
# imgs = foreach(k = 1:dim(data_testnosac)[1], .inorder = T, .packages = "raster") %dopar% {
#   r = stack(data_testnosac$img[k])
#   as.array(r)
# }
# msks = foreach(k = 1:dim(pred_testsac)[1], .inorder = T, .packages = "raster") %dopar% {
#   r = raster(data_testsac$msk[k])
#   as.array(r)
# }
# stopCluster(cl)
# msks = do.call(abind, list(msks))
# msks = aperm(msks, c(3,1,2))
# 
# imgs2 = do.call(abind, list(imgs))
# imgs2 = aperm(imgs2, c(3,1,2))
# 
# # par(mfrow = c(1,2))
# # plot(as.raster((pred_testsac[i,,]+1)/12))
# # plot(as.raster(msks[i,,]/12))
# # i=i+1
# 
# predVec = as.vector(pred_testsac)+1
# obsVec  = as.vector(msks)
# u       = sort(union(obsVec, predVec))
# conmat  = caret::confusionMatrix(data = factor(predVec, u), reference = factor(obsVec, u))
# 
# conmat$overall["Accuracy"]