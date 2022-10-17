##############################################################################################
module plot_var_analysis
##############################################################################################
# imports and exports
using Random, LinearAlgebra
using JLD2, HDF5, Plots, Statistics, Measures
using DataAssimilationBenchmarks

##############################################################################################
# parameters of interest
bkg_covs = ["ID", "clima"]
dims = collect(1:1:40)
s_infls = collect(1:1:199)

# pre-allocation for stabilized rmse
stab_rmse_id = ones(length(dims), length(s_infls)) * 20
stab_rmse_clima = ones(length(dims), length(s_infls)) * 20

# iterate through configurations of parameters of interest
for bkg_cov in bkg_covs
    for dim in dims
        for s_infl in s_infls
            infl = s_infl*0.005;
            # import data
            path = pkgdir(DataAssimilationBenchmarks) * "/src/data/D3-var-bkg-" * bkg_cov * "/"
            file = "bkg-" * bkg_cov * "_L96_state_seed_0123_diff_0.000_sysD_40_obsD_" * lpad(dim, 2, "0") *
            "_obsU_1.00_gamma_001.0_nanl_03500_tanl_0.05_h_0.05_stateInfl_" * rpad(round(infl, digits=3), 5, "0") * ".jld2"

            # retrieve statistics of interest
            ts = load(path * file)::Dict{String,Any}
            nanl = ts["nanl"]::Int64
            filt_rmse = ts["filt_rmse"]::Array{Float64, 1}

            # generate statistics based on background covariance
            if cmp(bkg_cov, "ID") == 0
                stab_rmse_id[dim,s_infl] = mean(filter(!isinf, filter(!isnan,filt_rmse[1000:nanl])))
            else
                stab_rmse_clima[dim,s_infl] = mean(filter(!isinf, filter(!isnan,filt_rmse[1000:nanl])))
            end
        end
    end
end

# transform data for plotting
infl_vals = s_infls*0.005
clima_t = transpose(stab_rmse_clima)
id_t = transpose(stab_rmse_id)

# create heatmap for background covariance using climatology
heatmap(dims,infl_vals,clima_t,clim=(0.001,5.0), title="Stabilized Filter RMSE Clima: Inflation Tuning Parameter vs. Dimensions", margin=15mm, size=(800,500), dpi = 600)
xlabel!("Dimensions")
ylabel!("Inflation Tuning Parameter")
savefig("stab_heat_clima_t")

# create heatmap for background covariance using identity
heatmap(dims,infl_vals,id_t,clim=(0.001,5.0), title="Stabilized Filter RMSE ID: Inflation Tuning Parameter vs. Dimensions", margin=15mm, size=(800,500), dpi = 600)
xlabel!("Dimensions")
ylabel!("Inflation Tuning Parameter")
savefig("stab_heat_id_t")

##############################################################################################
# end module

end
