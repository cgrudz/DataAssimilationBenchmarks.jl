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
s_infls = collect(1:1:199)

stab_rmse_id = ones(length(dims), length(s_infls)) * 20
stab_rmse_clima = ones(length(dims), length(s_infls)) * 20


for bkg_cov in bkg_covs
    for dim in dims
        for s_infl in s_infls
            infl = s_infl*0.005;
            path = pkgdir(DataAssimilationBenchmarks) * "/src/data/D3-var-bkg-" * bkg_cov * "/"
            file = "bkg-" * bkg_cov * "_L96_state_seed_0123_diff_0.000_sysD_40_obsD_" * lpad(dim, 2, "0") *
            "_obsU_1.00_gamma_001.0_nanl_03500_tanl_0.05_h_0.05_stateInfl_" * rpad(round(infl, digits=3), 5, "0") * ".jld2"

            ts = load(path * file)::Dict{String,Any}
            nanl = ts["nanl"]::Int64
            filt_rmse = ts["filt_rmse"]::Array{Float64, 1}
            if dim == 19
                display(filt_rmse[1000:nanl])
                print("\n")
            end
            if cmp(bkg_cov, "ID") == 0
                stab_rmse_id[dim,s_infl] = mean(filter(!isinf, filter(!isnan,filt_rmse[1000:nanl])))
            else
                stab_rmse_clima[dim,s_infl] = mean(filter(!isinf, filter(!isnan,filt_rmse[1000:nanl])))
            end
        end
    end
end

infl_vals = s_infls*0.005
clima_t = transpose(stab_rmse_clima)
id_t = transpose(stab_rmse_id)


heatmap(dims,infl_vals,clima_t,clim=(0.001,5.0), title="Stabilized Filter RMSE Clima: Inflation Tuning Parameter vs. Dimensions", margin=15mm, size=(800,500), dpi = 600)
xlabel!("Dimensions")
ylabel!("Inflation Tuning Parameter")
savefig("stab_heat_clima_t")

heatmap(dims,infl_vals,id_t,clim=(0.001,5.0), title="Stabilized Filter RMSE ID: Inflation Tuning Parameter vs. Dimensions", margin=15mm, size=(800,500), dpi = 600)
xlabel!("Dimensions")
ylabel!("Inflation Tuning Parameter")
savefig("stab_heat_id_t")
print("\n" * "ID Bot Left:")

##############################################################################################
# end module

end
