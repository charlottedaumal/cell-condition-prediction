#loading useful packages
using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using CSV, DataFrames, MLJLinearModels, Random, NearestNeighborModels
using Distances, Distributions, MLJ
Pkg.add("MLJModelInterface")
import MLJModelInterface
const MMI = MLJModelInterface
