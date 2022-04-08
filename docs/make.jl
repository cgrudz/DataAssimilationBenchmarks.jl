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
                "Types" => "home/DataAssimilationBenchmarks.md",
                ],
             "Submodules" => Any[
                 "Models" => Any[
                    "L96" => "submodules/models/L96.md",
                    "IEEE39bus" => "submodules/models/IEEE39bus.md"
                    ],
                 "Methods" => Any[
                    "DeSolvers" => "submodules/methods/DeSolvers.md",
                    "EnsembleKalmanSchemes" => "submodules/methods/EnsembleKalmanSchemes.md"
                   ],
                 "Experiments" => Any[
                    "GenerateTimeSeries" => "submodules/experiments/GenerateTimeSeries.md",
                    "FilterExps" => "submodules/experiments/FilterExps.md",
                    "SmootherExps" => "submodules/experiments/SmootherExps.md",
                   ]
                ]
            ]
)
