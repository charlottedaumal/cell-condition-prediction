using Pkg

Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))

using OpenML, MLJ, MLJLinearModels, MLJMultivariateStatsInterface, Distributions, MLJFlux, Flux, Random, DataFrames, CSV, MLCourse, MLJDecisionTreeInterface, MLJXGBoostInterface, Statistics



train_data = CSV.read(joinpath(@__DIR__, "train.csv"), DataFrame);



dropmissing!(train_data);

coerce!(train_data, :labels=>Multiclass);



all_train_data_input = select(train_data, Not(:labels));

all_train_data_output = train_data.labels;



standard_deviation_columns = std.(eachcol(all_train_data_input)); #computing the standard deviation of each column of the training data

const_columns_indices = [i for i in 1:size(all_train_data_input)[2] if standard_deviation_columns[i]==0]; #storing the indices of the columns for which the standard deviation is equal to 0

all_clean_const_train_input = select(all_train_data_input, Not(const_columns_indices)); #keeping only the columns for which the standard deviation is larger than 0



correlated_columns_indices = findall(≈(1), cor(Matrix(all_clean_const_train_input))) |> idxs -> filter(x -> x[1] > x[2], idxs); #storing all the indices of the correlated columns in the DataFrame

correlated_columns_indices = sort(union(idx.I[1] for idx in correlated_columns_indices)); #sorting the correlated columns' indices and keeping only one index to avoid pairs and removing to many columns

all_clean_uncorrelated_train_input = select(all_clean_const_train_input, Not(correlated_columns_indices)) #keeping only one column of the pairs of correlated columns in the DataFrame



mdenoise = fit!(machine(PCA(maxoutdim = 100), all_clean_uncorrelated_train_input), verbosity = 0);

report(mdenoise)

cleaned_train_input_PCA = MLJ.transform(mdenoise, all_clean_uncorrelated_train_input)





xgb = XGBoostClassifier()

tuned_XGB_mod= (TunedModel(model = xgb, resampling = CV(nfolds = 4), measure= MisclassificationRate(), tuning = Grid(goal = 10),

range = [range(xgb, :eta,lower = 1e-2, upper = .1, scale = :log), range(xgb, :max_depth, lower = 2, upper = 6), range(xgb, :min_child_weight, lower = 0.5, upper = 1.5)]))





mach_XGB=machine(tuned_XGB_mod, cleaned_train_input_PCA , all_train_data_output)

fit!(mach_XGB)



report(mach_XGB).best_model

evaluate!(machine(report(mach_XGB).best_model, cleaned_train_input_PCA , all_train_data_output), measure = MisclassificationRate())



tuned_mach_XGB = machine(XGBoostClassifier(eta = 0.10000000000000002, max_depth = 6, min_child_weight = 0.5), cleaned_train_input_PCA, all_train_data_output);

fit!(tuned_mach_XGB)



confusion_matrix(predict_mode(tuned_mach_XGB), all_train_data_output) #confusion matrix

training_auc = auc(predict(tuned_mach_XGB), all_train_data_output)

misclassification_rate = mean(predict(tuned_mach_XGB) .!= all_train_data_output)



test_data = CSV.read(joinpath(@__DIR__, "test.csv"), DataFrame);



dropmissing!(test_data);

all_clean_const_test_data = select(test_data, Not(const_columns_indices));

all_clean_uncorrelated_test_data = select(all_clean_const_test_data, Not(correlated_columns_indices));

cleaned_test_data_PCA = MLJ.transform(mdenoise, all_clean_uncorrelated_test_data);



test_predictions = predict_mode(tuned_mach_XGB, cleaned_test_data_PCA);

df_test_predictions = DataFrame(id = [i for i in 1:length(test_predictions)], prediction = [test_predictions[i] for i in 1:length(test_predictions)]);

CSV.write(joinpath(@__DIR__, "test_predictions_random_forest_PCA_Final.csv"), df_test_predictions)
