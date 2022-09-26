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
dims = collect(1:1:40)
s_infls = collect(0.005:0.005:1.0)

stab_rmse_id = Vector{Float64}(undef, 40)
stab_rmse_clima = Vector{Float64}(undef, 40)

for s_infl in s_infls
	for bkg_cov in bkg_covs
    		for dim in dims
        		path = pkgdir(DataAssimilationBenchmarks) * "/src/data/D3-var-bkg-" * bkg_cov * "/"
			file = "bkg-" * bkg_cov * "_L96_state_seed_0123_diff_0.000_sysD_40_obsD_" * lpad(dim, 2, "0") * 
        		"_obsU_1.00_gamma_001.0_nanl_03500_tanl_0.05_h_0.05_stateInfl_" * rpad(round(s_infl, digits=3), 5, "0") * ".jld2"

        		ts = load(path * file)::Dict{String,Any}
        		nanl = ts["nanl"]::Int64
        		filt_rmse = ts["filt_rmse"]::Array{Float64, 1}

        		if cmp(bkg_cov, "ID") == 0
				stab_rmse = mean(filt_rmse[1000:nanl])
				if (stab_rmse < stab_rmse_id[dim])
					stab_rmse_id[dim] = stab_rmse
				end
        		else
				stab_rmse = mean(filt_rmse[1000:nanl])
                                if (stab_rmse < stab_rmse_clima[dim])
                                        stab_rmse_clima[dim] = stab_rmse
                                end
        		end
    		end
	end
end

plot(dims, stab_rmse_id, label = "ID", title="Stabilized Filter RMSE vs. Dimensions", legend_position = :topright, margin=15mm, size=(800,500), dpi = 600)
plot!(dims, stab_rmse_clima, label = "Clima")
xlabel!("Dimensions")
ylabel!("Stabilized RMSE")
savefig("stab_minInfl")

##############################################################################################
# end module

end
