library (neuralnet)

#Check this article: http://www.r-bloggers.com/how-to-perform-a-logistic-regression-in-r/

trainingSampleGenerator <- function( s_dataset, posture, correct_size, varied_size){
# This function generates a training sample for the designated posture
# correct_size: Is the number of samples that are matched with the posture.
# varied_size: Is the number of samples that are mostly not matched with the posture.
# posture: Is the posture used to train the ANN. range = [1,9]
# This function is used to control the cases used to train the neural network.
  
  if(posture<1 || posture > 9){
    stop ("Posture out of bounds!");
  }
    
  list_Indexes <- which(s_dataset[,sprintf("Posture_%d",posture)]==1);
  s_limit_down <- list_Indexes[1];
  s_limit_up   <- list_Indexes[length(list_Indexes)];
  
  posture = posture+10;

  nn_sample_training_0 = s_dataset[sample(s_limit_down:s_limit_up,correct_size),];
  nn_sample_training_0 = nn_sample_training_0[c(1:10,posture)];
  nn_sample_training_1 = s_dataset[sample(1:nrow(s_dataset),varied_size),];
  nn_sample_training_1 = nn_sample_training_1[c(1:10,posture)];
  nn_sample_training   = rbind(nn_sample_training_0,nn_sample_training_1);
  return (sample(nn_sample_training));
}

testSampleGenerator <- function(s_dataset, sample_size){
# Returns a list to train the ANN.
# Sample size: Size of the sample to get.
  nn_sample_test = s_dataset[sample(1:nrow(s_dataset),sample_size),1:19];
  return (nn_sample_test);
}



NeuralNetworkCreation <- function(s_dataset, correct_size, varied_size, hidden_size){
# This function generates a list of 9 different neural networks. 
# Each classify a particular posture.
# If no convergence is reached with the given model, the hidden layer is reduced by one, and the model is re-trained.
  ANN_list <- list();
  
  for (i in 1:9){
    hidden_new = hidden_size;
    training <- trainingSampleGenerator(s_dataset,i, correct_size, varied_size);
    formula  <- paste(names(s_dataset)[10+i], "~", paste(names(s_dataset)[1:10], collapse=" + "));
    nn       <- neuralnet(formula,training,hidden=hidden_new,err.fct = "sse",act.fct = "logistic");
    while (is.null(nn$net.result)==TRUE){
      hidden_new = hidden_new-1;
      nn       <- neuralnet(formula,training,hidden=hidden_new,err.fct = "sse",act.fct = "logistic");
    }
    ANN_list[[i]] <- nn;
  }
  return (ANN_list);
}

errorCalculation <- function(ANN_list, test_sample, error_threshhold){
# This function calculates the classification accuracy  
  error_list = list();
  counter = 0;
  for (i in 1:9){
    ANN_error<-compute(x = ANN_list[[i]], covariate = test_sample[,1:10]);
    error_vector<-ifelse(ANN_error$net.result > error_threshhold,1,0);
    error_calculation_vector<-test_sample[,11:19]
    for(j in 1:nrow(error_calculation_vector)){
      if(error_vector[j]==error_calculation_vector[j,i]){
        counter = counter + 1;
      }
    }
    correct_percentage<-counter / nrow(error_calculation_vector);
    error_list[[i]]<-correct_percentage;
    counter = 0;
  }
  return (error_list)
}

init_ANN <- function(dataset, correct_size, varied_size, hidden_size, test_size, threshhold){
# Neural network creation is defined here.
# First step is to create the 9 neural networks.
# Second step is to generate the test sample.
# Last step is to calculate the error with the given sample  
  
  nn <- NeuralNetworkCreation(dataset,correct_size,varied_size,hidden_size);
  test_sample <- testSampleGenerator(dataset,test_size);
  error_vector <- errorCalculation(nn,test_sample,threshhold);
  nn;
  return (error_vector)
}

Average_error <- function(error_list){
  counter = 0;
  for(i in 1:9){
    counter = counter + error_list[[i]];
  }
  return (counter/9)
}

optimizeNetwork<-function(ANN_list, s_dataset, correct_size, varied_size,threshold ,posture, learning_rate, hidden_size){
  training <- trainingSampleGenerator(s_dataset,posture, correct_size, varied_size);
  formula  <- paste(names(s_dataset)[10+posture], "~", paste(names(s_dataset)[1:10], collapse=" + "));
  nn       <- neuralnet(formula,training,hidden=hidden_size,err.fct = "sse",act.fct = "logistic");
}

main <- function(dataset, correct_size, varied_size, hidden_size, test_size, threshhold){
  iter_vector <- init_ANN(dataset, correct_size, varied_size, hidden_size, test_size, threshhold);
  iter_average <- Average_error(iter_vector);
  return (iter_vector)
}
