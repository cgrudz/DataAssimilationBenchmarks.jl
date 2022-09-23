##############################################################################################
module plot_var_analysis
##############################################################################################
# imports and exports
using Random, LinearAlgebra
using JLD2, HDF5, Plots
using Measures
using DataAssimilationBenchmarks

##############################################################################################

path = pkgdir(DataAssimilationBenchmarks) * "/src/data/D3-var-bkg-ID/"
file = "bkg-ID_L96_state_seed_0000_diff_0.000_sysD_40_obsD_40_obsU_1.00_gamma_001.0_nanl_03500" *
"_tanl_0.05_h_0.05_stateInfl_0.200.jld2"

ts = load(path * file)::Dict{String,Any}
nanl = ts["nanl"]::Int64
filt_rmse = ts["filt_rmse"]::Array{Float64, 1}
t = collect(1:1:nanl)

plot(t, filt_rmse, label = "Filter", title="Uninformative Prior: Filter RMSE vs. Cycles", 
legend_position = :topright, margin=15mm, size=(800,500), dpi = 600)
xlabel!("Cycles")
ylabel!("Filter RMSE")
savefig("stab_I")

##############################################################################################
# end module

end
