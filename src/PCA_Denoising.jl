#loading useful packages
using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using CSV, DataFrames, MLJ, MLJLinearModels, Random, Distributions, MLJMultivariateStatsInterface, Plots

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

m_PCA = fit!(machine(PCA(), all_clean_uncorrelated_train_input), verbosity = 0); #fitting a PCA machine on the previously cleaned training data

#plotting the proportion of the variance and the cumulative proportion of variance explained by the training data's predictors
vars = report(m_PCA).principalvars ./ report(m_PCA).tvar;
p1 = plot(vars, label = nothing, yscale = :log10,
          xlabel = "component", ylabel = "proportion of variance explained")
p2 = plot(cumsum(vars),
          label = nothing, xlabel = "component",
          ylabel = "cumulative prop. of variance explained")
plot(p1, p2, layout = (1, 2), size = (600, 400))
