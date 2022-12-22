#loading useful packages
using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using CSV, DataFrames, MLJ, MLJLinearModels, Random, Distributions, MLJMultivariateStatsInterface, Plots

train_data = CSV.read(joinpath(@__DIR__, "data", "train.csv"), DataFrame); #loading the .CSV file containing the training data

dropmissing!(train_data); #removing rows with missing values

coerce!(train_data, :labels => Multiclass); #changing the type of the labels column to Multiclass

all_train_data_output = train_data.labels; #DataFrame containing the labels of the training data
all_train_data_input = select(train_data, Not(:labels)); #DataFrame containing all training data except labels

standard_deviation_columns = std.(eachcol(all_train_data_input));
const_columns_indices = [i for i in 1:size(all_train_data_input)[2] if standard_deviation_columns[i]==0];
all_clean_const_train_input = select(all_train_data_input, Not(const_columns_indices)); #keeping only the columns for which the standard deviation is larger than 0

correlated_columns_indices = findall(≈(1), cor(Matrix(all_clean_const_train_input))) |> idxs -> filter(x -> x[1] > x[2], idxs);
correlated_columns_indices = sort(union(idx.I[1] for idx in correlated_columns_indices));
all_clean_uncorrelated_train_input = select(all_clean_const_train_input, Not(correlated_columns_indices))

m_PCA = fit!(machine(PCA(), all_clean_uncorrelated_train_input), verbosity = 0);

vars = report(m_PCA).principalvars ./ report(m_PCA).tvar;
p1 = plot(vars, label = nothing, yscale = :log10,
          xlabel = "component", ylabel = "proportion of variance explained")
p2 = plot(cumsum(vars),
          label = nothing, xlabel = "component",
          ylabel = "cumulative prop. of variance explained")
plot(p1, p2, layout = (1, 2), size = (600, 400))

mdenoise = fit!(machine(PCA(maxoutdim = 5000), all_clean_uncorrelated_train_input), verbosity = 0);
report(mdenoise)
cleaned_train_input_PCA = MLJ.transform(mdenoise, all_clean_uncorrelated_train_input)

model_lc = LogisticClassifier(penalty = :l1); #determining the model's type with a Lasso Regression

self_tuning_model_lc = TunedModel(model = model_lc,resampling = CV(nfolds = 6),tuning = Grid(),
range = range(model_lc, :lambda, values = 1e-8:1e-2:1e-1)); #self-tuning the model
self_tuning_mach_lc = machine(self_tuning_model_lc, cleaned_train_input_PCA, all_train_data_output) |> fit! ; #fitting the tuned machine

rep_lc = report(self_tuning_mach_lc) #report of the tuned machine

tuned_machine_lc = machine(LogisticClassifier(penalty = :l1, lambda = 1e-8), cleaned_train_input_PCA, all_train_data_output);
fit!(tuned_machine_lc);

confusion_matrix(predict_mode(tuned_machine_lc), all_train_data_output) #confusion matrix
training_auc = auc(predict(tuned_machine_lc), all_train_data_output)
misclassification_rate = mean(predict(tuned_machine_lc) .!= all_train_data_output)

test_data = CSV.read(joinpath(@__DIR__, "data", "test.csv"), DataFrame);

dropmissing!(test_data);
all_clean_const_test_data = select(test_data, Not(const_columns_indices));
all_clean_uncorrelated_test_data = select(all_clean_const_test_data, Not(correlated_columns_indices));
cleaned_test_data_PCA = MLJ.transform(mdenoise, all_clean_uncorrelated_test_data);

test_predictions = predict_mode(tuned_machine_lc, cleaned_test_data_PCA);
df_test_predictions = DataFrame(id = [i for i in 1:length(test_predictions)], prediction = [test_predictions[i] for i in 1:length(test_predictions)]);
CSV.write(joinpath(@__DIR__, "test_predictions_logistic_classifier_L1_PCA_Final.csv"), df_test_predictions)
