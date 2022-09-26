##############################################################################################
module plot_var_analysis
##############################################################################################
# imports and exports
using Random, LinearAlgebra
using JLD2, HDF5, Plots, Statistics
using Measures
using DataAssimilationBenchmarks

##############################################################################################

bkg_covs = ["ID", "clima"]
dims = collect(1, 1, 40)

path = pkgdir(DataAssimilationBenchmarks) * "/src/data/D3-var-bkg-ID/"

stab_rmse_id = Vector{Float64}(undef, 40)
stab_rmse_clima = Vector{Float64}(undef, 40)

for bkg_cov in bkg_covs
    for dim in dims
        file = "bkg-" * bkg_cov * "_L96_state_seed_0000_diff_0.000_sysD_40_obsD_" * lpad(dim, 2, "0") * 
        "_obsU_1.00_gamma_001.0_nanl_03500_tanl_0.05_h_0.05_stateInfl_1.000.jld2"

        ts = load(path * file)::Dict{String,Any}
        nanl = ts["nanl"]::Int64
        filt_rmse = ts["filt_rmse"]::Array{Float64, 1}

        for cycle in cycles
            if cmp(bkg_cov, "ID") == 0
                stab_rmse_id[dim] = mean(filt_rmse[3000:nanl])
            else
                stab_rmse_clima[dim] = mean(filt_rmse[3000:nanl])
            end
        end
    end
end

plot(dims, stab_rmse_id, label = "ID", title="Stabilized Filter RMSE vs. Cycles", legend_position = :topright, margin=15mm, size=(800,500), dpi = 600)
plot!(dims, stab_rmse_clima, label = "ID")
xlabel!("Dimensions")
ylabel!("Stabilized RMSE")
savefig("stab_both")

##############################################################################################
# end module

end
