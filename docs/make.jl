using Documenter
using DataAssimilationBenchmarks

makedocs(
    sitename = "DataAssimilationBenchmarks",
    format = Documenter.HTML(),
    modules = [DataAssimilationBenchmarks],
    pages = [
             "Home" => "index.md",
             "Models" => Any[
                "L96" => "models/L96.md",
                "IEEE39bus" => "models/IEEE39bus.md"
                ],
             "Methods" => Any[
                "DeSolvers" => "methods/DeSolvers.md",
                "EnsembleKalmanSchemes" => "methods/EnsembleKalmanSchemes.md"
               ],
             "Experiments" => Any[
                "GenerateTimeSeries" => "experiments/GenerateTimeSeries.md",
                "FilterExperiments" => "experiments/FilterExps.md",
               ]
            ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
