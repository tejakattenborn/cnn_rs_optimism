

############## Description

# Preprocessing of orthoimages and reference data into standardized image tiles as required for CNN training
# Manuscript: Kattenborn et al. > Spatially autocorrelated training and validation samples inflate performance assessment of convolutional neural networks
# teja dot kattenborn at uni uni minus leipzig dot de


############## Code


# libraries + path --------------------------------------------------------

pkgs <- c("raster", "rgdal", "rgeos", "foreach", "doParallel", "magick")
sapply(pkgs, require, character.only = TRUE)


source("00_helper_functions.R")


dataDir <- "PATH"



# segment images function -------------------------------------------------

data  <- list.files(dataDir, pattern = "ortho", recursive = T)

AOIDir <- list.files(dataDir, pattern = "TreeSeg_plots.shp", recursive = T, full.names = T)
AOI    <- readOGR(AOIDir)

shape  <- readOGR(dsn = "poly_mask_clean.shp", verbose = FALSE)
shape  <- gBuffer(shape, byid = TRUE, width = 0)


# sites <- substr(files, 13, 18)

segmentImages <- function(files, # character
                          # outDir, # path to outputfolder
                          useDSM = FALSE,
                          aggregation, # numeric, factor to aggregate pixels
                          tilesize, # numeric, no of pixels (x and y direction) of input raster
                          plot = TRUE,
                          overwrite = FALSE) { # NAvalue, histogram stretch
  
  sites <- substr(files, 13, 18)
  # years <- substr(files, 1, 4)
  
  ## loop over sites
  for(j in 1:length(files)) {
    
    site <- sites[j]
    # year <- years[j]
    
    #### load data ####
    message(paste0("loading data ", j, "/", length(sites)))
    
    ## create outputfolder
    # if(useDSM) DSMtag <- "DSM/" else DSMtag <- "noDSM/"
    tileTag <- paste0("t", tilesize/aggregation)
    # resTag  <- paste0("r", aggregation)
    
    imgDir <- paste0("02_pipeline/", "img/", site, "/", tilesize/aggregation, "/")
    mskDir <- paste0("02_pipeline/", "msk/", site, "/", tilesize/aggregation, "/")
    if(!dir.exists(imgDir)){
      dir.create(imgDir, recursive = TRUE)
      dir.create(mskDir, recursive = TRUE)
    }
    
    ## remove old files if overwrite == TRUE
    if(overwrite) {
      unlink(list.files(imgDir, full.names = TRUE))
      unlink(list.files(mskDir, full.names = TRUE))
    }
    if(length(list.files(imgDir)) > 0 & overwrite == FALSE) {
      stop(paste0("Can't overwrite files in ", imgDir, " -> set 'overwrite = TRUE'"))
    }
    
    ## load area of interest
    # AOI <- readOGR(dsn = AOIDir, verbose = FALSE)
    # AOI <- gBuffer(AOI, byid = TRUE, width = 0)
    AOIsite <- AOI[AOI$plot_no == site, ]
    
    ## load ortho
    orthoFile <- paste0(dataDir, "ortho/ortho_", site, "_res_32N.tif")
    ortho     <- stack(orthoFile)
    ortho     <- ortho[[-4]] # remove alpha channel
    
    ## crop ortho to AOI
    ortho <- crop(ortho, AOIsite)
    if(substr(site,1,3) == "HAI") ortho <- mask(ortho, AOIsite)
    
    ## apply histogram stretch
    q     <- quantile(ortho, probs = c(.001, .999))
    ortho <- (ortho-min(q[,1])) * 255 / (max(q[,2]) - min(q[,1]))
    beginCluster()
    ortho <- clusterR(ortho, fun = reclassify, args = list(rcl = c(-Inf,0,0, 255,Inf,255)), datatype = "INT1U")
    endCluster()
    
    ## set NA values
    if(substr(site,1,3) == "CFB") {
      values(ortho)[values(ortho[[1]]) == 0 & values(ortho[[2]]) == 0 & values(ortho[[3]]) == 0] = NA
    } else {
      values(ortho)[values(ortho[[1]]) == 255 & values(ortho[[2]]) == 255 & values(ortho[[3]]) == 255] = NA
      values(ortho)[values(ortho[[1]]) == 0 & values(ortho[[2]]) == 0 & values(ortho[[3]]) == 0] = NA
    }
    
    ## load reference data
    shapeSite <- shape[shape$plot == site,]  
    shapeSite <- spTransform(shapeSite, crs(ortho))
    # shape$species_ID <- shape$species_ID-7
    
    ## plot site
    if(plot) {
      plotRGB(ortho, colNA = "red")
      lines(shapeSite, lwd = 1.5, col = "orange")
      lines(AOIsite, col = "red", lwd = 2)
      # lines(rasShp, col = "yellow", lwd = 1.5)
    }
    
    #### segment images + masks ####
    message(paste0("segmenting images ", j, "/", length(sites)))
    
    ## define kernel size
    kernelSizeX <- tilesize * xres(ortho)
    kernelSizeY <- tilesize * yres(ortho)
    
    ## create sample positions
    xOffset <- (ncol(ortho)/256 - floor(ncol(ortho)/256))/2 * 256 * xres(ortho)
    yOffset <- (nrow(ortho)/256 - floor(nrow(ortho)/256))/2 * 256 * yres(ortho)
    
    x <- seq(extent(ortho)[1] + kernelSizeX/2 + xOffset,
             extent(ortho)[2] - kernelSizeX/2 - xOffset,
             kernelSizeX)
    y <- seq(extent(ortho)[3] + kernelSizeY/2 + yOffset,
             extent(ortho)[4] - kernelSizeY/2 - yOffset,
             kernelSizeY)
    
    XYpos <- expand.grid(x, y)
    XYpos <- SpatialPointsDataFrame(coords = XYpos, proj4string = crs(AOIsite), data = XYpos)
    XYpos <- XYpos[AOIsite,]
    if(plot) points(XYpos, col = "yellow", pch = 3)
    XYpos <- as.data.frame(XYpos)[,c(1,2)]
    XYpos <- cbind(XYpos[,1] - kernelSizeX/2,
                   XYpos[,1] + kernelSizeX/2,
                   XYpos[,2] - kernelSizeY/2,
                   XYpos[,2] + kernelSizeY/2)
    rownames(XYpos) <- paste0("img", sprintf("%04d", 1:nrow(XYpos)), "_", site, "_", tileTag, ".png")
    

    ## crop images and calc percentage cover of endmember
    cl <- makeCluster(19)
    registerDoParallel(cl)
    
    rmXY <- foreach(i = 1:nrow(XYpos), .packages = c("raster", "rgdal", "keras", "magick"), .combine = "c", .inorder = T) %dopar% {
      
      cropExt <- extent(XYpos[i,])
      
      ## crop and write rasters
      orthoCrop <- crop(ortho, cropExt)
      orthoCrop <- crop(orthoCrop, extent(orthoCrop, 1, tilesize, 1, tilesize)) # remove rounding artifacts
      if(aggregation > 1) orthoCrop <- aggregate(orthoCrop, fact = aggregation)
      
      
      ## crop mask
      polyCrop <- crop(shapeSite, cropExt)
      if(length(polyCrop) > 0) { # rasterize shapefile if polygons exist 
        polyCropR  <- rasterize(polyCrop, orthoCrop[[1]], field = polyCrop$species_ID)
        
        NAidx      <- which(is.na(values(polyCropR)))
        # flagPoly0  <- !(0 %in% polyCrop$species_ID) # check if species_ID in data
        flagPolyNA <- length(NAidx) < 2500 # TRUE if NAValues exist AND no less then 2500 (50*50 pixel = 1m2) in crop
        flagOrtho  <- length(which(is.na(values(orthoCrop[[1]])) == TRUE)) == 0 # TRUE if no NA in crop
      } else {
        flagPolyNA <- flagOrtho <- FALSE
      }
      
      
      if(flagOrtho && flagPolyNA) {
        # fill NA values
        if(length(NAidx) > 0) {
          rows <- rowFromCell(polyCropR, NAidx)
          cols <- colFromCell(polyCropR, NAidx)
          
          left <- cols-floor(40/2); left[left < 1] = 1
          top  <- rows-floor(40/2); top[top < 1] = 1
          for(k in 1:length(NAidx)) {
            vals                <- getValuesBlock(polyCropR, row = top[k], nrow = 40, col = left[k], ncol = 40)
            polyCropR[NAidx[k]] <- as.numeric(names(table(vals)[1]))
          } 
        }
        
        extent(orthoCrop) <- extent(0, kernelSizeX, 0, kernelSizeY)
        extent(polyCropR) <- extent(0, kernelSizeX, 0, kernelSizeY)
        
        orthoCrop <- as.array(orthoCrop)
        polyCropR <- as.array(polyCropR)
        orthoCrop <- image_read(orthoCrop / 255)
        polyCropR <- image_read(polyCropR / 255)
        
        filename  <- paste(site, tileTag, sep = "_")
        image_write(orthoCrop, format = "png",
                    path = paste0(imgDir, "img", sprintf("%04d", i), "_", filename, ".png"))
        image_write(polyCropR, format = "png",
                    path = paste0(mskDir, "msk", sprintf("%04d", i), "_", filename, ".png"))
      }

      rm(orthoCrop, polyCrop, polyCropR, cropExt)
      
      # return if tile was exported
      flagOrtho && flagPolyNA
      
    }
    stopCluster(cl)
    
    # export xy positions to a text file
    XYpos           <- as.data.frame(XYpos)
    XYpos           <- XYpos[-which(rmXY == FALSE), ] # remove unexported tiles from list
    colnames(XYpos) <- c("xmin", "xmax", "ymin", "ymax")
    write.csv(XYpos, file = paste0(imgDir, "/metadataXYpos.csv"))
    
    removeTmpFiles(h=0.17)
    gc()
  }
}

segmentImages(files = data[3:(length(data)-4)], useDSM = F, tilesize = 256, aggregation = 1, plot = T, overwrite = T) #outDir = "02_pipeline/softmax/", 


