



#### CHECK / TODO

# write metadata of test files to disk
# maybe set float 32 here as in cvae example: decoder(z_sample)
# check further integers (e.g. in show classes or fashion output)

# https://github.com/rstudio/keras/blob/master/vignettes/examples/eager_cvae.R

library(keras)
library(tensorflow)
library(tfdatasets)
library(dplyr)
library(ggplot2)
library(glue)
require(data.table)
require(abind) # binding RGB arrays
require(countcolors) # RGB plotting


tf$keras$backend$set_floatx('float32')
gpus = tf$config$experimental$list_physical_devices('GPU')
tf$config$experimental$set_memory_growth(gpus[[1]], TRUE)


# Setup and preprocessing -------------------------------------------------


#set dirs
#workdir = "/home/sysgen/Teja/"
workdir = "D:/"
setwd(workdir)
path_img = "cvae_sac/input_img_jpeg"
outdir = "cvae_sac/1_results/"
checkpoint_dir = paste0(outdir, "checkpoints_cvae")

# workdir = "D:/"
# setwd(workdir)
# path_img = "data_ortho_confobi_sac/input_img_jpeg"
# #path_meta = paste0(path_img,"metadata_with_try.txt")
# outdir = "./data_ortho_confobi_sac/1_results/"
# checkpoint_dir = paste0(outdir, "checkpoints_cvae")


# image settings
xres = 256L #256L
yres = 256L #256L
no_bands = 3L
no_test_samples = 100


# hyperparameters
latent_dim <- 200L
buffer_size <- 60000
batch_size <- 12L
num_epochs <- 300
batches_per_epoch <- buffer_size / batch_size
grid_axis = 0.3 # range for the latent feature space for generating synthetic imagery 

# set seed
#tf$compat$v1$set_random_seed(as.integer(28))

# set memory growth policy
#gpu1 <- tf$config$experimental$get_visible_devices('GPU')[[1]]
#gpu2 <- tf$config$experimental$get_visible_devices('GPU')[[2]]
#tf$config$experimental$set_memory_growth(device = gpu1, enable = TRUE)
#tf$config$experimental$set_memory_growth(device = gpu2, enable = TRUE)

#strategy <- tf$distribute$MirroredStrategy()
#strategy$num_replicas_in_sync



# Loading Data ----------------------------------------------------------------

# list all img  data
path_img = list.files(path_img, full.names = T, pattern = "_CFB", recursive = T)
path_img = path_img[sample(1:length(path_img), size = buffer_size)]

# split test data (10%) and save to disk
#testIdx = sample(x = 1:nrow(img_meta), size = floor(nrow(img_meta)/10), replace = F)
testIdx = sample(x = 1:length(path_img), size = no_test_samples, replace = F)
test_img = path_img[testIdx]
#save(test_img, file = paste0(outdir, "test_img.RData"), overwrite = T)
#test_img_meta = img_meta[testIdx,]
#save(test_img_meta, file = paste0(outdir, "test_img_meta.RData"), overwrite = T)
train_img = path_img[-testIdx]
#train_img_meta = img_meta[-testIdx,]


# just for testing
#train_img_meta$mean_3113[is.na(train_img_meta$mean_3113)] = 0
#test_img_meta$mean_3113[is.na(test_img_meta$mean_3113)] = 0

#train_data = tibble(img = train_img, meta = train_img_meta$mean_3113)
#test_data = tibble(img = test_img, meta = test_img_meta$mean_3113)

train_data = tibble(img = train_img)
test_data = tibble(img = test_img)

head(train_data)
dim(train_data)
train_data$img

#sum(is.na(train_data$meta))


# tfdatasets input pipeline -----------------------------------------------


create_dataset <- function(data,
                           train, # logical. TRUE for augmentation of training data
                           batch, # numeric. multiplied by number of available gpus since batches will be split between gpus
                           epochs,
                           shuffle, # logical. default TRUE, set FALSE for test data
                           dataset_size){ # numeric. number of samples per epoch the model will be trained on
  
  
  if(shuffle){
    dataset = data %>%
      tensor_slices_dataset() %>%
      dataset_shuffle(buffer_size = length(data$img), reshuffle_each_iteration = TRUE)
  } else {
    dataset = data %>%
      tensor_slices_dataset() 
  } 
  
  dataset = dataset %>%
    dataset_map(~.x %>% purrr::list_modify( # read files and decode png
      #img = tf$image$decode_png(tf$io$read_file(.x$img), channels = no_bands)
      img = tf$image$decode_jpeg(tf$io$read_file(.x$img), channels = no_bands, try_recover_truncated = TRUE, acceptable_fraction=0.5) %>%
        
        tf$image$convert_image_dtype(dtype = tf$float32) %>% 
        tf$image$resize(preserve_aspect_ratio = TRUE, size = as.integer(c(ceiling(xres*2.1), ceiling(yres*2.1)))) %>%
        tf$image$resize_with_crop_or_pad(target_height = yres, target_width = xres)
      
      #tf$image$resize_with_crop_or_pad(target_height = yres, target_width = xres)
      #tf$squeeze() %>% # removes dimensions of size 1 from the shape of a tensor
      # # resize all images to common resolution  https://www.tensorflow.org/api_docs/python/tf/image/resize_with_crop_or_pad
      #), num_parallel_calls = parallel::detectCores())  %>%
    #), num_parallel_calls = tf$data$experimental$AUTOTUNE) %>%
    ), num_parallel_calls = NULL) %>%
    #), num_parallel_calls = 4)
    
    
    #dataset = dataset %>%
    #dataset_prepare(x = img, named_features = TRUE, batch_size = batch, drop_remainder = TRUE) %>% 
    dataset_batch(batch, drop_remainder = TRUE) %>%
    #dataset_prepare(x = img, named_features = TRUE, batch_size = batch, drop_remainder = TRUE) %>% # to include further variables
    #dataset_prefetch(buffer_size = tf$data$experimental$AUTOTUNE)
    dataset_prefetch_to_device("/gpu:0", buffer_size = tf$data$experimental$AUTOTUNE)
}




# Parameters ----------------------------------------------------------------


dataset_size <- length(train_data$img)

train_dataset <- create_dataset(train_data,  batch = batch_size, epochs = num_epochs, dataset_size = dataset_size, shuffle = FALSE)
test_dataset <- create_dataset(test_data,  batch = dim(test_data)[1], shuffle = FALSE)

dataset_iter = reticulate::as_iterator(train_dataset)
example = dataset_iter %>% reticulate::iter_next()
example
plot(as.raster(as.array(example$img[[1]][,,1:3])))






# Model -------------------------------------------------------------------


encoder_model <- function(name = NULL) {
  keras_model_custom(name = name, function(self) {
    
    
    self$conv1 <-
      layer_conv_2d( # to 128
        #input_shape=c(xres, yres, no_bands),
        filters = 32L,
        kernel_size = 3L,
        strides = 2L,
        activation = "relu",
        dtype = "float32"
      )
    
    self$conv2 <- # to 64
      layer_conv_2d(
        filters = 64L,
        kernel_size = 3L,
        strides = 2L,
        activation = "relu"
      )
    
    self$conv3 <- # to 32
      layer_conv_2d(
        filters = 64L,
        kernel_size = 3L,
        strides = 2L,
        activation = "relu"
      )
    
    self$conv4 <- # to 16
      layer_conv_2d(
        filters = 64L,
        kernel_size = 3L,
        strides = 2L,
        activation = "relu"
      )
    
    self$conv5 <- # to 8
      layer_conv_2d(
        filters = 128L,
        kernel_size = 3L,
        strides = 2L,
        activation = "relu"
      )
    
    self$flatten <- layer_flatten()
    self$dense <- layer_dense(units = latent_dim, dtype = "float32")
    
    function (x, mask = NULL) {
      x %>%    # XXX
        self$conv1() %>%
        self$conv2() %>%
        self$conv3() %>%
        self$conv4() %>%
        self$conv5() %>%
        self$flatten() %>%
        self$dense() #%>%
    }
  })
}



decoder_model <- function(name = NULL) {
  
  keras_model_custom(name = name, function(self) {
    self$dense <- layer_dense(units = 8L * 8L * 128L, activation = "relu", dtype = "float32")
    self$reshape <- layer_reshape(target_shape = c(8L, 8L, 128L))
    
    self$deconv1 <- #16
      layer_conv_2d_transpose(
        filters = 128L,
        kernel_size = 3L,
        strides = 2L,
        padding = "same",
        activation = "relu"
      )
    self$deconv2 <- #32
      layer_conv_2d_transpose(
        filters = 64L,
        kernel_size = 3L,
        strides = 2L,
        padding = "same",
        activation = "relu"
      )
    self$deconv2a <- #64
      layer_conv_2d_transpose(
        filters = 64L,
        kernel_size = 3,
        strides = 2,
        padding = "same",
        activation = "relu"
      )
    self$deconv2b <- #128
      layer_conv_2d_transpose(
        filters = 64L,
        kernel_size = 3,
        strides = 2,
        padding = "same",
        activation = "relu"
      )
    self$deconv2c <- #256
      layer_conv_2d_transpose(
        filters = 32L,
        kernel_size = 3,
        strides = 2,
        padding = "same",
        activation = "relu"
      )
    self$deconv4 <-
      layer_conv_2d_transpose(
        filters = no_bands,
        kernel_size = 3L,
        strides = 1L,
        padding = "same",
        activation = "sigmoid",
        dtype = "float32"
      )
    
    
    function (x, mask = NULL) {
      x %>%
        self$dense() %>%
        self$reshape() %>%
        self$deconv1() %>%
        self$deconv2() %>%
        self$deconv2a() %>%
        self$deconv2b() %>%
        self$deconv2c() %>%
        self$deconv4()
    }
  })
}



optimizer <- tf$keras$optimizers$Adam(1e-3)



compute_kernel <- function(x, y) {
  x_size <- k_shape(x)[1] # batch size
  y_size <- k_shape(y)[1] # batch size
  dim <- k_shape(x)[2] # img width
  
  tiled_x <- k_tile(k_reshape(x, k_stack(list(x_size, 1L, dim))), k_stack(list(1L, y_size, 1L)))
  tiled_y <- k_tile(k_reshape(y, k_stack(list(1L, y_size, dim))), k_stack(list(x_size, 1L, 1L)))
  
  k_exp(-k_mean(k_square(tiled_x - tiled_y), axis = 3L) / k_cast(dim, tf$float32))
}

compute_mmd <- function(x, y, sigma_sqr = 1) {
  x_kernel <- compute_kernel(x, x)
  y_kernel <- compute_kernel(y, y)
  xy_kernel <- compute_kernel(x, y)
  k_mean(x_kernel) + k_mean(y_kernel) - 2 * k_mean(xy_kernel)
}




# Output utilities --------------------------------------------------------

num_examples_to_generate <- 64

random_vector_for_generation <-
  k_random_normal(shape = list(num_examples_to_generate, latent_dim),
                  stddev = grid_axis/2, dtype = tf$float32)

generate_random_tiles <- function(epoch) {
  predictions <- decoder(random_vector_for_generation) #%>% tf$nn$sigmoid()
  png(paste0(outdir,"cvae_tiles_epoch_", epoch, ".png"), width = 3000, height = 3000)
  par(mfcol = c(8, 8))
  #par(mar = c(0.5, 0.5, 0.5, 0.5),
  par(mar = c(0.01, 0.01, 0.01, 0.01),
      oma = c(0.01, 0.01, 0.01, 0.01),
      xaxs = 'i',
      yaxs = 'i')
  for (i in 1:64) {
    
    # for grey-scale only
    #img <- t(apply(img, 2, rev))
    # image(
    #   1:xres,
    #   1:yres,
    #   img * 127.5 + 127.5,
    #   col = gray((0:255) / 255),
    #   xaxt = 'n',
    #   yaxt = 'n'
    #   )
    
    #for RGB
    img <- as.array(predictions[i, , , ])
    plotArrayAsImage(as.array(img))
  }
  dev.off()
}



show_latent_space <- function(epoch) {
  
  iter <- make_iterator_one_shot(test_dataset)
  x <-  iterator_get_next(iter)
  x_test_encoded <- encoder(x$img)  #check that [[1]] must not be added
  #x_test_encoded %>%
  #as.matrix() %>%
  #as.data.frame() %>%
  prediction = as.data.frame(as.matrix(x_test_encoded))
  #mutate(class = class_names[fashion$test$y + 1]) %>%
  #mutate(class = test_img_meta$species) %>%
  #mutate(variable = test_data$meta) %>%
  #mutate(variable = as.numeric(x$y)) %>% # variable to be mapped accross latent space
  #ggplot(aes(x = V1, y = V2, colour = variable)) + geom_point() +
  
  if(latent_dim>3){
    ggplot(aes(x = V1, y = V2, colour = 1), data = prediction) + geom_point() +
      geom_point(data = prediction, aes(x = V3, y = V4, colour = 2)) +
      theme(aspect.ratio = 1) +
      theme(plot.margin = unit(c(0, 0, 0, 0), "null")) +
      theme(panel.spacing = unit(c(0, 0, 0, 0), "null"))
  }else{
    ggplot(aes(x = V1, y = V2, colour = 1), data = prediction) + geom_point() +
      theme(aspect.ratio = 1) +
      theme(plot.margin = unit(c(0, 0, 0, 0), "null")) +
      theme(panel.spacing = unit(c(0, 0, 0, 0), "null"))
  }
  
  ggsave(
    paste0(outdir,"mmd_latentspace_epoch", epoch, ".png"),
    width = 10,
    height = 10,
    units = "cm"
  )
}




show_grid <- function(epoch) {
  png(paste0(outdir,"mmd_grid_epoch_", epoch, ".png"), width = 3000, height = 3000)
  par(mar = c(0.5, 0.5, 0.5, 0.5),
      xaxs = 'i',
      yaxs = 'i')
  n <- 16
  img_size <- xres
  grid_x <- seq(-grid_axis, grid_axis, length.out = n)
  grid_y <- seq(-grid_axis, grid_axis, length.out = n)
  
  
  rows <- NULL
  
  for (i in 1:length(grid_x)) {
    column <- NULL
    for (j in 1:length(grid_y)) {
      z_sample <- matrix(c(grid_x[i], grid_y[j], sample(seq(-grid_axis, grid_axis, length.out = latent_dim*5), latent_dim-2)), ncol = latent_dim)
      
      column <-
        abind(column,
              #(decoder(z_sample, 'float32') %>% as.numeric()) %>% matrix(ncol = img_size))
              as.array(decoder(z_sample, 'float32')[1,,,]), along = 1)
    }
    rows <- abind(rows, column, along=2)
  }
  plotArrayAsImage(rows)
  #rows %>% as.raster() %>% plot()
  dev.off()
}


show_grid_v1_v2 <- function(epoch) {
  png(paste0(outdir,"mmd_grid_v1_v2_epoch_", epoch, ".png"), width = 3000, height = 3000)
  par(mar = c(0.5, 0.5, 0.5, 0.5),
      xaxs = 'i',
      yaxs = 'i')
  n <- 16
  img_size <- xres
  grid_x <- seq(-grid_axis, grid_axis, length.out = n)
  grid_y <- seq(-grid_axis, grid_axis, length.out = n)
  
  rows <- NULL
  
  for (i in 1:length(grid_x)) {
    column <- NULL
    for (j in 1:length(grid_y)) {
      #z_sample <- matrix(c(grid_x[i], grid_y[j]), ncol = 2)
      z_sample <- matrix(c(grid_x[i], grid_y[j], rep(0, latent_dim-2)), ncol = latent_dim)
      
      column <-
        abind(column,
              #(decoder(z_sample, 'float32') %>% as.numeric()) %>% matrix(ncol = img_size))
              as.array(decoder(z_sample, 'float32')[1,,,]), along = 1)
    }
    rows <- abind(rows, column, along=2)
  }
  plotArrayAsImage(rows)
  #rows %>% as.raster() %>% plot()
  dev.off()
}

show_grid_v3_v4 <- function(epoch) {
  png(paste0(outdir,"mmd_grid_v3_v4_epoch_", epoch, ".png"), width = 3000, height = 3000)
  par(mar = c(0.5, 0.5, 0.5, 0.5),
      xaxs = 'i',
      yaxs = 'i')
  n <- 16
  img_size <- xres
  grid_x <- seq(-grid_axis, grid_axis, length.out = n)
  grid_y <- seq(-grid_axis, grid_axis, length.out = n)
  
  rows <- NULL
  
  for (i in 1:length(grid_x)) {
    column <- NULL
    for (j in 1:length(grid_y)) {
      z_sample <- matrix(c( rep(0, 2), grid_x[i], grid_y[j], rep(0, latent_dim-4)), ncol = latent_dim)
      
      column <-
        abind(column,
              #(decoder(z_sample, 'float32') %>% as.numeric()) %>% matrix(ncol = img_size))
              as.array(decoder(z_sample, 'float32')[1,,,]), along = 1)
    }
    rows <- abind(rows, column, along=2)
  }
  plotArrayAsImage(rows)
  #rows %>% as.raster() %>% plot()
  dev.off()
}

encode_vs_decode <- function(epoch) {
  png(paste0(outdir,"encode_vs_decode_epoch_", epoch, ".png"), width = 3000, height = 3000)
  #x <-  iterator_get_next(iter)
  x = reticulate::as_iterator(test_dataset) %>% reticulate::iter_next()
  plot(as.raster(as.array(example$img[[1]][,,1:3])))
  par(mfrow=c(4,4))
  for(i in 1:8){
    plotArrayAsImage(as.array(x$img)[i,,,])
    plotArrayAsImage(as.array(decoder(matrix(as.numeric(encoder(x$img)[i,]), ncol=latent_dim)))[1,,,])
  }
  dev.off()
}



# Training loop -----------------------------------------------------------


encoder <- encoder_model()
decoder <- decoder_model()


checkpoint_prefix <- file.path(checkpoint_dir, "ckpt")
checkpoint <-  tf$train$Checkpoint(optimizer = optimizer,
                                   encoder = encoder,
                                   decoder = decoder)
#manager <- tf$train$CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep = 50)

# generate_random_tiles(0)
# show_latent_space(0)
# show_grid(0)
# show_grid_v1_v2(0)
# show_grid_v3_v4(0)

metrics = data.frame(epoch = 1:num_epochs, loss_nll_total = rep(NA, num_epochs), loss_mmd_total = rep(NA, num_epochs), total = rep(NA, num_epochs))


for (epoch in seq_len(num_epochs)) {
  iter <- make_iterator_one_shot(train_dataset)
  
  # total_loss <- 0
  # logpx_z_total <- 0
  # logpz_total <- 0
  # logqz_x_total <- 0
  
  total_loss <- 0
  loss_nll_total <- 0
  loss_mmd_total <- 0
  
  until_out_of_range({
    x <-  iterator_get_next(iter)
    
    
    with(tf$GradientTape(persistent = TRUE) %as% tape, {
      
      mean <- encoder(x$img)
      preds <- decoder(mean)
      
      true_samples <- k_random_normal(shape = c(batch_size, latent_dim), dtype = tf$float32)
      loss_mmd <- compute_mmd(true_samples, mean)
      loss_nll <- k_mean(k_square(x$img - preds))
      loss <- loss_nll + loss_mmd
      
    })
    
    total_loss <- total_loss + loss
    loss_mmd_total <- loss_mmd + loss_mmd_total
    loss_nll_total <- loss_nll + loss_nll_total
    
    encoder_gradients <- tape$gradient(loss, encoder$variables)
    decoder_gradients <- tape$gradient(loss, decoder$variables)
    
    optimizer$apply_gradients(purrr::transpose(list(
      encoder_gradients, encoder$variables
    )))
    optimizer$apply_gradients(purrr::transpose(list(
      decoder_gradients, decoder$variables
    )))
    
    
    # TODO
    # https://stackoverflow.com/questions/54284274/tensorflow-how-to-initialize-global-step
    # check how global step is defined in apply_gradients
    
    
    
  })
  
  checkpoint$save(file_prefix = checkpoint_prefix)
  #manager$save()
  
  cat(
    glue(
      "Losses (epoch): {epoch}:",
      "  {(as.numeric(loss_nll_total)/batches_per_epoch) %>% round(4)} loss_nll_total,",
      "  {(as.numeric(loss_mmd_total)/batches_per_epoch) %>% round(4)} loss_mmd_total,",
      "  {(as.numeric(total_loss)/batches_per_epoch) %>% round(4)} total"
    ),
    "\n"
  )
  
  metrics$loss_nll_total[epoch] = (as.numeric(loss_nll_total)/batches_per_epoch) %>% round(4)
  metrics$loss_mmd_total[epoch] = (as.numeric(loss_mmd_total)/batches_per_epoch) %>% round(4)
  metrics$total[epoch] = (as.numeric(total_loss)/batches_per_epoch) %>% round(4)
  
  write.csv(metrics, paste0(outdir, "cvae_loss_per_epoch.csv"), row.names = F)
  
  #if (epoch %% 3 == 0) {
  
  encode_vs_decode(epoch)
  generate_random_tiles(epoch)
  show_latent_space(epoch)
  show_grid(epoch)
  show_grid_v1_v2(epoch)
  show_grid_v3_v4(epoch)
  #}
}





############################
#APLICATION

encoder


checkpoint2 = checkpoint


status= checkpoint2$restore(tf$train$latest_checkpoint("D:/data_ortho_confobi_sac/1_results_256_200lv_250epoch/checkpoints_cvae")) # checkpoint file can also be modified to point not to the latest but another epoch
status$assert_existing_objects_matched()

root_dir = "D:/data_ortho_confobi_sac/input_img_jpeg/data_ortho_confobi_sac/input_img"
#root_dir = "cvae_sac/input_img_jpeg/data_ortho_confobi_sac/input_img"
plot_dir = list.dirs(root_dir, recursive = F)

for(i in 1:length(plot_dir)){
  
  test_img = list.files(plot_dir[i], full.names = T, pattern = "jpg", recursive = T)
  test_data = tibble(img = test_img)
  test_dataset <- create_dataset(test_data, train = FALSE, shuffle = FALSE, batch = 1) # increasing batch increases speed, but last elements might be skipped
  
  meta = list.files(plot_dir[i], full.names = T, pattern = "metadataXYpos", recursive = T)
  meta = read.csv(meta)
  
  preds = matrix(1, nrow = 0, ncol=latent_dim)
  
  iter <- make_iterator_one_shot(test_dataset)
  until_out_of_range({
    x <-  iterator_get_next(iter)
    preds = rbind(preds, as.matrix(encoder(x$img)))
  })
  
  meta = cbind(meta, preds)
  write.csv(meta, paste0("D:/data_ortho_confobi_sac/1_results_256_200lv_250epoch_latent/", basename(plot_dir[i]),".csv"))
  
}




# info on checkpoints, etc...:
#https://tensorflow.rstudio.com/tutorials/beginners/basic-ml/tutorial_save_and_restore/
#https://tensorflow.rstudio.com/guide/saving/checkpoints/

#fresh_model %>% load_model_weights_tf(filepath = checkpoint_path)
#fresh_model %>% evaluate(test_images, test_labels, verbose = 0)

# 
# encoder2 = encoder
# encoder2 %>% load_model_weights_tf(filepath = "D:/data_ortho_confobi_sac/1_results/checkpoints_cvae/ckpt-1")
# 
# 
# encoder2 %>% save_model_weights_tf("checkpoints/cp.ckpt")
# 
# 
# new_model <- encoder
# new_model %>% load_model_weights_tf('checkpoints/cp.ckpt')
