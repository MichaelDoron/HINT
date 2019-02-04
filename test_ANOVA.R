
library(reticulate)
library(mgcv)
library(pROC)

source_python("generate_global_functions_for_r.py")
# task = 'hard'
# number_of_variables = 100
# number_of_functions = 1000

# task = 'easy'
# number_of_variables = 10
# number_of_functions = 25

task = 'vin'
number_of_variables = 10

for (number_of_samples in c(1, 10, 100, 1000)) {
    # print(number_of_samples)
    rocs = NULL
    y_trues = NULL
    predictions = NULL
    times = NULL
    iteration = 0
    for (iteration in 0:29) {
        N = number_of_samples * ((number_of_variables ** 2) + number_of_variables) + number_of_samples
        if (task == 'vin') {
            DAT <- read.csv(sprintf("ga2m_synt_%s_%s.train", number_of_samples, iteration), header = FALSE, sep=" ")
            true_pairs <- read.csv(sprintf("r_global_function_vin_true_pairs.csv"))
        } else {
            true_pairs <- read.csv(sprintf("random_function_%s_%s_true_pairs_%s.csv", iteration, number_of_samples, task), header = FALSE, sep=",")
            DAT <- read.csv(sprintf("random_function_%s_%s_%s.train", iteration, number_of_samples, task), header = FALSE, sep=" ")
        }

        start.time <- Sys.time()
        attach(DAT)
        formula <- as.formula(sprintf("V%s ~ %s + (%s)^2", number_of_variables + 1, (paste(names(DAT)[1:length(names(DAT)) - 1], collapse=' + ')), paste(names(DAT)[1:length(names(DAT)) - 1], collapse=' + ')))
        b<-gam(formula, data = DAT)
        p_values <- anova(b)$p.pv
        end.time <- Sys.time()
        interaction_colnames = names(p_values)[(length(names(DAT)) + 1):length(names(p_values))]
        interaction_score = order(p_values[interaction_colnames])
        true_pairs  = true_pairs[,2:3]
        true_pairs_names = NULL
        for (ind in 1:dim(true_pairs)[1]) {
            if ((true_pairs[ind,2] + 1) > true_pairs[ind,1] + 1){
                true_pairs_names <- c(true_pairs_names, sprintf("V%s:V%s", true_pairs[ind,1] + 1, true_pairs[ind,2] + 1))
            }
        }
        y_true = NULL
        for (ind in 1:length(interaction_colnames)) {
            if (interaction_colnames[ind] %in% true_pairs_names) {
                y_true <- c(y_true, 1)
            } else {
                y_true <- c(y_true, 0)
            }
        }
        rocs <- c(rocs, (auc(as.vector(y_true), abs(as.vector(p_values[interaction_colnames])))[1]))
        times <- c(times, end.time - start.time)
        y_trues <- c(y_trues, (as.vector(y_true)))
        predictions <- c(predictions, (as.vector(abs(as.vector(p_values[interaction_colnames])))))
        print(auc(as.vector(y_true), abs(as.vector(p_values[interaction_colnames])))[1])
        detach()
        filename <- sprintf('global_ANOVA_AUC_%s_%s.csv', task, number_of_samples)
        write.csv(c(mean(rocs),sd(rocs)), file = filename)
        filename <- sprintf('global_ANOVA_y_trues_%s_%s_%s.csv', task, number_of_samples, iteration)
        write.csv((as.vector(y_true)), file = filename)
        filename <- sprintf('global_ANOVA_predictions_%s_%s_%s.csv', task, number_of_samples, iteration)
        write.csv((as.vector(abs(as.vector(p_values[interaction_colnames])))), file = filename)
    }
    print("Finished number of samples")
    print(number_of_samples)
    print(mean(rocs))
    filename <- sprintf('global_ANOVA_times_%s_%s.csv', task, number_of_samples)
    write.csv(times, file = filename)    
}

