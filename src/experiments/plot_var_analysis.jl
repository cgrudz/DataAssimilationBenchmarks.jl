##############################################################################################
module plot_var_analysis
##############################################################################################
# imports and exports
using Random, Distributions, LinearAlgebra, StatsBase, Statistics
using JLD2, HDF5, Plots
using Measures
using DataAssimilationBenchmarks

##############################################################################################
# define tuning range
tuning_min = 0.01
tuning_step = 0.01
tuning_max = 1.0


t = collect(tuning_min:tuning_step:tuning_max)
stab_fore_rmse = Vector{Float64}(undef, length(t))
stab_filt_rmse = Vector{Float64}(undef, length(t))

loadpath = pkgdir(DataAssimilationBenchmarks) * "/src/data/d3_var_exp/"

for i in 1:length(t)
      time_series = "D3_var_filter_analysis_L96_time_series_seed_0000_gam_1.000_Informed_false" * 
      "_Updated_true_Tuned_" * rpad(t[i], 5, "0") * ".jld2"

      ts = load(loadpath * time_series)::Dict{String,Any}
      nanl = ts["nanl"]::Int64
      fore_rmse = ts["fore_rmse"]::Array{Float64, 1}
      filt_rmse = ts["filt_rmse"]::Array{Float64, 1}

      stab_fore_rmse[i] = sum(fore_rmse[1:(nanl-1)])/(nanl-1)
      stab_filt_rmse[i] = sum(filt_rmse[1:(nanl-1)])/(nanl-1)
end

print("Minimum: ")
print(findmin(stab_filt_rmse))
plot(t, stab_fore_rmse, label = "Forecast", title="Uninformative Prior: Stabilized Analysis RMSE vs. Tuning Parameter", legend_position = :topright, margin=15mm, size=(800,500), dpi = 600)
plot!(t, stab_filt_rmse, label = "Filter", )
plot!([tuning_min, tuning_max], [1, 1], label = "Observation Error Std")
xlabel!("Tuning Parameter")
ylabel!("Stabilized Analysis RMSE")
savefig("stab_I")

##############################################################################################
# end module

end
