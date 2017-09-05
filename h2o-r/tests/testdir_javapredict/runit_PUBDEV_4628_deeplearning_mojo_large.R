setwd(normalizePath(dirname(
  R.utils::commandArgs(asValues = TRUE)$"f"
)))
source("../../scripts/h2o-r-test-setup.R")
#----------------------------------------------------------------------
# Purpose:  This Runit test aims to test the correctness of deeplearning
# mojo implementation.  A random neural network is generated and trained
# with a random dataset.  The prediction from h2o predict and mojo
# predict on a test dataset will be compared and they should equal.
#----------------------------------------------------------------------

test.deeplearning.mojo <-
  function() {
    #----------------------------------------------------------------------
    # Parameters for the test.
    #----------------------------------------------------------------------
    allAct = c("Tanh", "TanhWithDropout", "Rectifier", "RectifierWithDropout", "Maxout", "MaxoutWithDropout")
    problemType = c("binomial", "multinomial", "regression")
    missingValues = c('Skip', 'MeanImputation')
    allFactors = c(TRUE, FALSE)
    categoricalEncodings = c("AUTO", "OneHotInternal", "Binary", "Eigen")
    
    problem = problemType[sample(1:length(problemType), replace = F)[1]]
    actFunc = allAct[sample(1:length(allAct), replace = F)[1]]
    missing_values = missingValues[sample(1:length(missingValues), replace =
                                            F)[1]]
    cateEn = categoricalEncodings[sample(1:length(categoricalEncodings), replace =
                                           F)[1]]
    toStandardize = allFactors[sample(1:length(allFactors), replace = F)[1]]
    useAllFactors = allFactors[sample(1:length(allFactors), replace = F)[1]]
    enableAutoEncoder = allFactors[sample(1:length(allFactors), replace = F)[1]]
    
    if (problem == 'regression') {
      loss <- 'Automatic'
    } else {
      loss <- 'CrossEntropy'
    }
  
    numTest = 1000
    training_file <- random_dataset(problem, testrow = numTest)
    ratios <- (h2o.nrow(training_file)-numTest)/h2o.nrow(training_file)
    allFrames <- h2o.splitFrame(training_file, ratios)
    training_frame <- allFrames[[1]]
    test_frame <- allFrames[[2]]
    allNames = h2o.names(training_frame)
    nn_structure <- random_NN(actFunc, 6, 10)
    params                  <- list()
    params$loss             <- loss
    params$use_all_factor_levels <- useAllFactors
    params$activation <- actFunc
    params$standardize <- toStandardize
    params$missing_values_handling <- missing_values
    params$categorical_encoding <- cateEn
    params$hidden <- nn_structure$hidden
    params$training_frame <- training_frame
    if (!params$autoencoder)
      params$y <- "response"
    params$x <- allNames[-which(allNames=="response")]
    params$autoencoder <- enableAutoEncoder
    
    if (length(nn_structure$hiddenDropouts) > 0) {
      params$input_dropout_ratio <- runif(1, 0, 0.1)
      params$hidden_dropout_ratios <- nn_structure$hiddenDropouts
    }
    
    #----------------------------------------------------------------------
    # Run the test
    #----------------------------------------------------------------------
    e <- tryCatch(doJavapredictTest("deeplearning",
                                    'not_used_here',
                                    test_frame,
                                    params,
                                    pojo_model = FALSE), error = function(x) x)
    expect_true(!all(sapply("Prediction mismatch", grepl, e[[1]])))
  }

doTest("GBM test", test.deeplearning.mojo)
