using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using CSV, DataFrames, MLJ, MLJLinearModels, Random, Distributions, MLJMultivariateStatsInterface, Plots

train_data = CSV.read(joinpath(@__DIR__, "data", "train.csv"), DataFrame); #loading the .CSV file containing the training data
test_data = CSV.read(joinpath(@__DIR__, "data" "test.csv"), DataFrame); #loading the .CSV file containing the test data

dropmissing!(train_data); #removing rows with missing values
coerce!(train_data, :labels => Multiclass); #changing the type of the labels column to Multiclass

all_train_data_output = train_data.labels; #DataFrame containing the labels of the training data
all_train_data_input = select(train_data, Not(:labels)); #DataFrame containing all training data except labels

standard_deviation_columns = std.(eachcol(all_train_data_input)); #computing the standard deviation of each column of the training data
const_columns_indices = [i for i in 1:size(all_train_data_input)[2] if standard_deviation_columns[i]==0]; #storing the indices of the columns for which the standard deviation is equal to 0
all_clean_const_train_input = select(all_train_data_input, Not(const_columns_indices)); #keeping only the columns for which the standard deviation is larger than 0

correlated_columns_indices = findall(≈(1), cor(Matrix(all_clean_const_train_input))) |> idxs -> filter(x -> x[1] > x[2], idxs); #storing all the indices of the correlated columns in the DataFrame
correlated_columns_indices = sort(union(idx.I[1] for idx in correlated_columns_indices)); #sorting the correlated columns' indices and keeping only one index to avoid pairs and removing to many columns
all_clean_uncorrelated_train_input = select(all_clean_const_train_input, Not(correlated_columns_indices)) #keeping uncorrelated columns

mdenoise = fit!(machine(PCA(maxoutdim = 5000), all_clean_uncorrelated_train_input), verbosity = 0); #fitting a PCA machine for denoising the training data
report(mdenoise) #reporting the main characteristics of the PCA denoising machine 
cleaned_train_input_PCA = MLJ.transform(mdenoise, all_clean_uncorrelated_train_input) #keeping only the predictors that explain almost all the variance in the training data


xgb = XGBoostClassifier()  # using the method XGBoostClassifier() for the tuning
tuned_XGB_mod= (TunedModel(model = xgb, resampling = CV(nfolds = 4), measure= MisclassificationRate(), tuning = Grid(goal = 10),range = [range(xgb, :eta,lower = 1e-2, upper = .1, scale = :log), range(xgb, :max_depth, lower = 2, upper = 6), range(xgb, :min_child_weight, lower = 0.5, upper = 1.5)])) # tuning the model for hyperparameters eta, max_depth and min_child_weight
mach_XGB=machine(tuned_XGB_mod, cleaned_train_input_PCA , all_train_data_output) # implementing a machine on the previously cleaned training data
fit!(mach_XGB) # fitting the machine

report(mach_XGB).best_model ##reporting the main characteristics of the machine
evaluate!(machine(report(mach_XGB).best_model, cleaned_train_input_PCA , all_train_data_output), measure = MisclassificationRate()) #evaluating the best model found 

tuned_mach_XGB = machine(XGBoostClassifier(eta = 0.10000000000000002, max_depth = 6, min_child_weight = 0.5), cleaned_train_input_PCA, all_train_data_output);  # #choosing the best value for eta, max_depth and min_child_weight
fit!(tuned_mach_XGB) # fitting the tuned machine

confusion_matrix(predict_mode(tuned_mach_XGB), all_train_data_output) #computation of the confusion matrix
training_auc = auc(predict(tuned_mach_XGB), all_train_data_output) # computation of the AUC
misclassification_rate = mean(predict(tuned_mach_XGB) .!= all_train_data_output) # computation of the misclassification rate 


dropmissing!(test_data); #removing rows with missing values
all_clean_const_test_data = select(test_data, Not(const_columns_indices)); #keeping the same columns as for the training data for which the standard deviation is larger than 0
all_clean_uncorrelated_test_data = select(all_clean_const_test_data, Not(correlated_columns_indices)); #keeping the same uncorrelated columns as for the training data
cleaned_test_data_PCA = MLJ.transform(mdenoise, all_clean_uncorrelated_test_data); #keeping only the same predictors that explain almost all the variance as for the training data

test_predictions = predict_mode(tuned_mach_XGB, cleaned_test_data_PCA); #predicting labels with the tuned model on the test data
df_test_predictions = DataFrame(id = [i for i in 1:length(test_predictions)], prediction = [test_predictions[i] for i in 1:length(test_predictions)]); #creating a DataFrame of the predicted labels
CSV.write(joinpath(@__DIR__, "test_predictions_random_forest_PCA_Final.csv"), df_test_predictions) #saving a .CSV file with all the labels' predictions






