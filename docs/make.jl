using Documenter
using DataAssimilationBenchmarks

makedocs(
    sitename = "DataAssimilationBenchmarks",
    format = Documenter.HTML(),
    modules = [DataAssimilationBenchmarks],
    pages = [
             "Home" => "index.md",
             "DataAssimilationBenchmarks" => Any[
               "Introduction" => "home/Introduction.md",
               "Getting Started" => "home/Getting Started.md",
               "Global Types" => "home/DataAssimilationBenchmarks.md",
              ],
             "Submodules" => Any[
               "Models" => Any[
                 "L96" => "submodules/models/L96.md",
                 "IEEE39bus" => "submodules/models/IEEE39bus.md",
                 "ObsOperators" => "submodules/models/ObsOperators.md"
                 ],
               "Methods" => Any[
                  "DeSolvers" => "submodules/methods/DeSolvers.md",
                  "EnsembleKalmanSchemes" => "submodules/methods/EnsembleKalmanSchemes.md",
                  "XdVAR" => "submodules/methods/XdVAR.md"
                 ],
               "Experiments" => Any[
                 "Workflow" => "submodules/experiments/Workflow.md",
                 "GenerateTimeSeries" => "submodules/experiments/GenerateTimeSeries.md",
                 "FilterExps" => "submodules/experiments/FilterExps.md",
                 "SmootherExps" => "submodules/experiments/SmootherExps.md",
                 "SingleExperimentDriver" => "submodules/experiments/SingleExperimentDriver.md",
                 "ParallelExperimentDriver" => "submodules/experiments/ParallelExperimentDriver.md",
                ],
               "Analysis" => Any[
                 "ProcessExperimentData" => "submodules/analysis/ProcessExperimentData.md",
                 "PlotExperimentData" => "submodules/analysis/PlotExperimentData.md",
                ],
              ],
             "Community" => Any[
               "Contributing" => "community/Contributing.md",
               "Contributor Covenant Code of Conduct" => "community/CodeOfConduct.md",
               ],
            ]
)

deploydocs(
    repo = "github.com:cgrudz/DataAssimilationBenchmarks.jl.git",
)
