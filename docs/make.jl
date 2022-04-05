using Documenter
using DataAssimilationBenchmarks

makedocs(
    sitename = "DataAssimilationBenchmarks",
    format = Documenter.HTML(),
    modules = [DataAssimilationBenchmarks]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
