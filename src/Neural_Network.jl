
using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using CSV, DataFrames, MLJ, MLJLinearModels, Random, Distributions, MLJMultivariateStatsInterface, Plots

train_data = CSV.read(joinpath(@__DIR__, "data", "train.csv"), DataFrame); #loading the .CSV file containing the training data
test_data = CSV.read(joinpath(@__DIR__, "test.csv"), DataFrame);

dropmissing!(train_data); #removing rows with missing values
dropmissing!(test_data);
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

#implementation of a neuronal network classifier machine 
neuronal_network_classifier_machine = machine(NeuralNetworkClassifier(builder=MLJFlux.Short(n_hidden=128, dropout=0.1, σ=relu), batch_size=32, epochs=30),cleaned_train_input_PCA, all_train_data_output)|> fit!;

all_clean_const_test_data = select(test_data, Not(const_columns_indices));
all_clean_uncorrelated_test_data = select(all_clean_const_test_data, Not(correlated_columns_indices));

cleaned_test_data_PCA = MLJ.transform(mdenoise, all_clean_uncorrelated_test_data);


prediction_neural_network=String.(predict_mode(neuronal_network_classifier_machine, cleaned_test_data_PCA))


df_neural_network=DataFrame(id=1:3093, prediction=prediction_neural_network)
CSV.write("./test_predictions_NeuralNetworkClassifier_PCA_Final.csv", df_neural_network)


confusion_matrix(predict_mode(neuronal_network_classifier_machine), all_train_data_output) # computation of confusion matrix 

training_auc = auc(predict(neuronal_network_classifier_machine), all_train_data_output)#computation of training AUC

misclassification_rate = mean(predict(neuronal_network_classifier_machine) .!= all_train_data_output) #computation of missclassification rate
