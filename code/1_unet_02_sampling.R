


sites <- list.files("02_pipeline/img/")
pathImg <- list.files("02_pipeline/img", pattern = ".png", recursive = T, full.names = T)

fold <- 5

seeds <- c(28, 1, 6, 19, 43)
seed  <- seeds[fold]

## sample 7 test sites
set.seed(seed)
sites_nosac    <- sample(sites, size = 7, replace = F)
sites_sac      <- sites[-which(sites %in% sites_nosac)]
idx_path_nosac <- which(substr(pathImg, 36, 41) %in% sites_nosac)

## sample first 10 train sites
set.seed(seed)
sites_sac_10 <- sample(sites_sac, 10, replace = F)
sites_sac    <- sites_sac[-which(sites_sac %in% sites_sac_10)]

## sample additional 15 train sites
set.seed(seed)
sites_sac_25 <- sample(sites_sac, 15, replace = F)
sites_sac_40 <- sites_sac[-which(sites_sac %in% sites_sac_25)]

## check sampling
length(unique(c(sites_sac_10, sites_sac_25, sites_sac_40)))



## get img/msk paths per subset
idx_path_sites_sac_10 <- which(substr(pathImg, 36, 41) %in% sites_sac_10)
pathImg_sites_sac_10  <- pathImg[idx_path_sites_sac_10]

idx_path_sites_sac_25 <- which(substr(pathImg, 36, 41) %in% sites_sac_25)
pathImg_sites_sac_25  <- pathImg[idx_path_sites_sac_25]

idx_path_sites_sac_40 <- which(substr(pathImg, 36, 41) %in% sites_sac_40)
pathImg_sites_sac_40  <- pathImg[idx_path_sites_sac_40]

length(c(pathImg_sites_sac_10, pathImg_sites_sac_25, pathImg_sites_sac_40, idx_path_nosac)) == length(pathImg)


## data splitting
# 10 flights
set.seed(seed)
n_data_sac_10  <- length(pathImg_sites_sac_10)
n_test_sac_10  <- round(n_data_sac_10 * 0.2)
n_train_sac_10 <- round((n_data_sac_10-n_test_sac_10) * 0.75)

pathImg_test_sac_10 <- sample(pathImg_sites_sac_10, n_test_sac_10, replace = F)
pathImg_sites_sac_10 <- pathImg_sites_sac_10[-which(pathImg_sites_sac_10 %in% pathImg_test_sac_10)]
pathImg_train_sac_10 <- sample(pathImg_sites_sac_10, n_train_sac_10, replace = F)
pathImg_valid_sac_10 <- pathImg_sites_sac_10[-which(pathImg_sites_sac_10 %in% pathImg_train_sac_10)]

length(c(pathImg_test_sac_10, pathImg_train_sac_10, pathImg_valid_sac_10)) == n_data_sac_10

# 25 flights
set.seed(seed)
n_data_sac_25  <- length(pathImg_sites_sac_25)
n_test_sac_25  <- round(n_data_sac_25 * 0.2)
n_train_sac_25 <- round((n_data_sac_25-n_test_sac_25) * 0.75)

pathImg_test_sac_25 <- sample(pathImg_sites_sac_25, n_test_sac_25, replace = F)
pathImg_sites_sac_25 <- pathImg_sites_sac_25[-which(pathImg_sites_sac_25 %in% pathImg_test_sac_25)]
pathImg_train_sac_25 <- sample(pathImg_sites_sac_25, n_train_sac_25, replace = F)
pathImg_valid_sac_25 <- pathImg_sites_sac_25[-which(pathImg_sites_sac_25 %in% pathImg_train_sac_25)]

length(c(pathImg_test_sac_25, pathImg_train_sac_25, pathImg_valid_sac_25)) == n_data_sac_25

# 40 flights
set.seed(seed)
n_data_sac_40  <- length(pathImg_sites_sac_40)
n_test_sac_40  <- round(n_data_sac_40 * 0.2)
n_train_sac_40 <- round((n_data_sac_40-n_test_sac_40) * 0.75)

pathImg_test_sac_40 <- sample(pathImg_sites_sac_40, n_test_sac_40, replace = F)
pathImg_sites_sac_40 <- pathImg_sites_sac_40[-which(pathImg_sites_sac_40 %in% pathImg_test_sac_40)]
pathImg_train_sac_40 <- sample(pathImg_sites_sac_40, n_train_sac_40, replace = F)
pathImg_valid_sac_40 <- pathImg_sites_sac_40[-which(pathImg_sites_sac_40 %in% pathImg_train_sac_40)]

length(c(pathImg_test_sac_40, pathImg_train_sac_40, pathImg_valid_sac_40)) == n_data_sac_40


## create sampling matrices
data10 <- matrix(data = 0, nrow = length(pathImg), ncol = 4,
                 dimnames = list(NULL, c("train", "valid", "test_sac", "test_nosac")))
data10 <- as.data.frame(data10)
data10$test_nosac[idx_path_nosac] = 1
data10$test_sac[which(pathImg %in% pathImg_test_sac_10)] = 1
data10$train[which(pathImg %in% pathImg_train_sac_10)]   = 1
data10$valid[which(pathImg %in% pathImg_valid_sac_10)]   = 1

data25 <- data10
data25$test_sac[which(pathImg %in% pathImg_test_sac_25)] = 1
data25$train[which(pathImg %in% pathImg_train_sac_25)]   = 1
data25$valid[which(pathImg %in% pathImg_valid_sac_25)]   = 1

data40 <- data25
data40$test_sac[which(pathImg %in% pathImg_test_sac_40)] = 1
data40$train[which(pathImg %in% pathImg_train_sac_40)]   = 1
data40$valid[which(pathImg %in% pathImg_valid_sac_40)]   = 1

sum(rowSums(data40)) == length(pathImg)


## create final list
dataSplit <- list()

dataSplit[[1]]  <- list(data = data10, aug_rad = T, aug_geo = F)
dataSplit[[2]]  <- list(data = data10, aug_rad = F, aug_geo = T)
dataSplit[[3]]  <- list(data = data10, aug_rad = T, aug_geo = T)
dataSplit[[4]]  <- list(data = data10, aug_rad = F, aug_geo = F)

dataSplit[[5]]  <- list(data = data25, aug_rad = T, aug_geo = F)
dataSplit[[6]]  <- list(data = data25, aug_rad = F, aug_geo = T)
dataSplit[[7]]  <- list(data = data25, aug_rad = T, aug_geo = T)
dataSplit[[8]]  <- list(data = data25, aug_rad = F, aug_geo = F)

dataSplit[[9]]  <- list(data = data40, aug_rad = T, aug_geo = F)
dataSplit[[10]] <- list(data = data40, aug_rad = F, aug_geo = T)
dataSplit[[11]] <- list(data = data40, aug_rad = T, aug_geo = T)
dataSplit[[12]] <- list(data = data40, aug_rad = F, aug_geo = F)


save(dataSplit, file = paste0("dataSplit_fold", fold, ".RData"))

