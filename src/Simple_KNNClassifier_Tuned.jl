#loading useful packages
using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using CSV, DataFrames, MLJLinearModels, Random, NearestNeighborModels
using Distances, Distributions, MLJ
Pkg.add("MLJModelInterface")
import MLJModelInterface
const MMI = MLJModelInterface

#functions defining the SimpleKNNClassifier
MMI.@mlj_model mutable struct SimpleKNNClassifier <: MMI.Probabilistic
    K::Int = 5 :: (_ > 0)
    metric::Distances.Metric = Euclidean()
end
function MMI.fit(::SimpleKNNClassifier, verbosity, X, y)
    fitresult = (; X = MMI.matrix(X, transpose = true), y)
    fitresult, nothing, nothing
end
function MMI.predict(model::SimpleKNNClassifier, fitresult, Xnew)
    similarities = pairwise(model.metric,
                            fitresult.X, MMI.matrix(Xnew, transpose = true))
    [Distributions.fit(MLJ.UnivariateFinite, fitresult.y[partialsortperm(col, 1:model.K)])
     for col in eachcol(similarities)]
end
function MMI.predict_mode(model::SimpleKNNClassifier, fitresult, Xnew)
    mode.(predict(model, fitresult, Xnew))
end

train_data = CSV.read(joinpath(@__DIR__, "data", "train.csv"), DataFrame); #loading the .CSV file containing the training data

dropmissing!(train_data); #removing rows with missing values

coerce!(train_data, :labels => Multiclass); #changing the type of the labels column to Multiclass

all_train_data_output = train_data.labels; #DataFrame containing the labels of the training data
all_train_data_input = select(train_data, Not(:labels)); #DataFrame containing all training data except labels

standard_deviation_columns = std.(eachcol(all_train_data_input)); #computing the standard deviation of each column of the training data
const_columns_indices = [i for i in 1:size(all_train_data_input)[2] if standard_deviation_columns[i]==0]; #storing the indices of the columns for which the standard deviation is equal to 0
all_clean_const_train_input = select(all_train_data_input, Not(const_columns_indices)); #keeping only the columns for which the standard deviation is larger than 0

correlated_columns_indices = findall(≈(1), cor(Matrix(all_clean_const_train_input))) |> idxs -> filter(x -> x[1] > x[2], idxs); #storing all the indices of the correlated columns in the DataFrame
correlated_columns_indices = sort(union(idx.I[1] for idx in correlated_columns_indices)); #sorting the correlated columns' indices and keeping only one index to avoid pairs and removing to many columns
all_clean_uncorrelated_train_input = select(all_clean_const_train_input, Not(correlated_columns_indices)) #keeping only one column of the pairs of correlated columns in the DataFrame


model_KNN = SimpleKNNClassifier() #determining the model's type 

self_tuning_model_KNN = TunedModel(model = model_KNN, resampling = CV(nfolds = 4), tuning = Grid(), range = range(model_KNN, :K, values = 2:8), measure = MisclassificationRate()) #self-tuning the K value of the SimpleKNNClassifier model
self_tuning_mach_KNN = machine(self_tuning_model_KNN, all_clean_uncorrelated_train_input, all_train_data_output) |> fit! #fitting the tuned machine

rep_KNN = report(self_tuning_mach_KNN) #reporting the main characteristics of the tuned machine
evaluate!(machine(report(self_tuning_mach_KNN).best_model, all_clean_uncorrelated_train_input, all_train_data_output), measure = MisclassificationRate()) #evaluating the best model found during tuning

tuned_machine_KNN = machine(SimpleKNNClassifier(K=4), all_clean_uncorrelated_train_input, all_train_data_output) #choosing the best value for K in the SimpleKNNClassifier model
fit!(tuned_machine_KNN) #fitting the machine with the best value for K in the SimpleKNNClassifier model

#assessing the model
confusion_matrix(predict_mode(tuned_machine_KNN), all_train_data_output) #confusion matrix
training_auc = auc(predict(tuned_machine_KNN), all_train_data_output) #computing the training auc
misclassification_rate = mean(predict(tuned_machine_KNN) .!= all_train_data_output) #computing the misclassification rate

test_data = CSV.read(joinpath(@__DIR__, "data", "test.csv"), DataFrame); #loading the .CSV file containing the test data

dropmissing!(test_data); #removing rows with missing values
all_clean_const_test_data = select(test_data, Not(const_columns_indices)); #keeping the same columns as for the training data for which the standard deviation is larger than 0
all_clean_uncorrelated_test_data = select(all_clean_const_test_data, Not(correlated_columns_indices)); #keeping the same uncorrelated columns as for the training data

test_predictions = predict_mode(tuned_machine_KNN, all_clean_uncorrelated_test_data); #predicting labels with the tuned model on the test data
df_test_predictions = DataFrame(id = [i for i in 1:length(test_predictions)], prediction = [test_predictions[i] for i in 1:length(test_predictions)]); #creating a DataFrame of the predicted labels
CSV.write(joinpath(@__DIR__, "test_predictions_SimpleKNNClassifier_KTuned_Final.csv"), df_test_predictions) #saving a .CSV file with all the labels' predictions
