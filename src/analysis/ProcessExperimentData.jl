#######################################################################################################################
module ProcessExperimentData
########################################################################################################################
########################################################################################################################
# imports and exports
using Debugger
using Statistics
using JLD, HDF5
export process_filter_state, process_smoother_state, process_smoother_param, process_filter_nonlinear_obs, 
       process_smoother_nonlinear_obs, process_smoother_versus_shift, process_smoother_versus_tanl,   
       rename_smoother_state

########################################################################################################################
########################################################################################################################
# Scripts for processing experimental output data and writing to JLD and HDF5 to read into matplotlib later
#
# These scripts are designed to try to load every file according to the standard naming conventions
# and if these files cannot be loaded, to save inf as a dummy variable for missing or corrupted data.
# NOTE: these scripts are currently deprecated and there is only a single method provided as an example
# Future plans include merging in methods for processing data that are more consistent.
########################################################################################################################

function process_filter_state()
    # creates an array of the average RMSE and spread for each experiment 
    # ensemble size is increasing from the origin on the horizontal axis
    # inflation is increasing from the origin on the vertical axis
    
    # time the operation
    t1 = time()

    # static parameters that are not varied 
    seed = 0
    tanl = 0.05
    nanl = 20000
    burn = 5000
    diffusion = 0.0
    h = 0.01
    obs_un = 1.0
    obs_dim = 40
    sys_dim = 40
    γ = 1.0
    
    # parameters in ranges that will be used in loops
    analysis_list = [
                     "fore", 
                     "filt",
                    ]
    stat_list = [
                 "rmse",
                 "spread",
                ]
    method_list = [
                   "enkf", 
                   "etkf",
                   "enkf-n-primal", 
                  ]
    ensemble_sizes = 15:2:41 
    total_ensembles = length(ensemble_sizes)
    inflations = LinRange(1.00, 1.10, 11)
    total_inflations = length(inflations)

    # define the storage dictionary here, looping over the method list
    data = Dict{String, Array{Float64}}()
    for method in method_list
        if method == "enkf-n"
            for analysis in analysis_list
                for stat in stat_list
                    # multiplicative inflation parameter should always be one, there is no dimension for this variable
                    data[method * "_" * analysis * "_" * stat] = Array{Float64}(undef, total_ensembles)
                end
            end
        else
            for analysis in analysis_list
                for stat in stat_list
                    # create storage for inflations and ensembles
                    data[method * "_" * analysis * "_" * stat] = Array{Float64}(undef, total_inflations, total_ensembles)
                end
            end
        end
    end

    # auxilliary function to process data, producing rmse and spread averages
    function process_data(fnames::Vector{String}, method::String)
        # loop ensemble size, last axis
        for j in 0:(total_ensembles - 1)
            if method[1:6] == "enkf-n"
                try
                    # attempt to load the file
                    tmp = load(fnames[j+1])
                    
                    # if successful, continue to unpack arrays and store the mean stats over 
                    # the experiment after the burn period for stationary statistics
                    for analysis in analysis_list
                        for stat in stat_list
                            analysis_stat = tmp[analysis * "_" * stat]::Vector{Float64}
                            
                            data[method * "_" * analyis * "_" * stat][j+1] = 
                            mean(analysis_stat[burn+1: nanl+burn])
                        end
                    end
                catch
                    # file is missing or corrupted, load infinity to represent an incomplete or unstable experiment
                    for analysis in analysis_list
                        for stat in stat_list
                            analysis_stat = tmp[analysis * "_" * stat]::Vector{Float64}
                            data[method * "_" * analyis * "_" * stat][j+1] = inf
                        end
                    end
                end
            else
                # loop inflations, first axis
                for i in 1:total_inflations
                    try
                        # attempt to load the file
                        tmp = load(fnames[i + j*total_inflations])
                        
                        # if successful, continue to unpack arrays and store the mean stats over 
                        # the experiment after the burn period for stationary statistics
                        for analysis in analysis_list
                            for stat in stat_list
                                analysis_stat = tmp[analysis * "_" * stat]::Vector{Float64}
                                
                                data[method * "_" * analyis * "_" * stat][total_inflations + 1 - i, j + 1] =
                                mean(analysis_stat[burn+1: nanl+burn])
                            end
                        end
                    catch
                        # file is missing or corrupted, load infinity to represent an incomplete or unstable experiment
                        for analysis in analysis_list
                            for stat in stat_list
                                analysis_stat = tmp[analysis * "_" * stat]::Vector{Float64}
                                data[method * "_" * analyis * "_" * stat][total_inflations + 1 - i, j + 1] = inf
                            end
                        end
                    end
                end
            end
        end
    end

    # define path to data on server
    fpath = "/x/capa/scratch/cgrudzien/final_experiment_data/all_ens/"
    
    # generate the range of experiments, storing file names as a list
    for method in method_list
        fnames = [] 
        for N_ens in ensemble_sizes
            if method[1:6] == "enkf-n"
                
                # inflation is a static value of 1.0
                name = method * 
                        "_L96_state_seed_" * lpad(seed, 4, "0") *
                        "_diffusion_" * rpad(diffusion, 4, "0") * 
                        "_sys_dim_" * lpad(sys_dim, 2, "0") * 
                        "_obs_dim_" * lpad(obs_dim, 2, "0") * 
                        "_obs_un_" * rpad(obs_un, 4, "0") *
                        "_gamma_" * lpad(γ, 5, "0") *
                        "_nanl_" * lpad(nanl + burn, 5, "0") * 
                        "_tanl_" * rpad(tanl, 4, "0") * 
                        "_h_" * rpad(h, 4, "0") *
                        "_N_ens_" * lpad(N_ens, 3,"0") * 
                        "_state_inflation_" * rpad(round(1.0, digits=2), 4, "0") * 
                        ".jld"
                push!(fnames, fpath * method * "/diffusion_" * rpad(diffusion, 4, "0") * "/" * name)

            else
                # loop inflations
                for infl in inflations
                    name = method * 
                            "_L96_state_seed_" * lpad(seed, 4, "0") *
                            "_diffusion_" * rpad(diffusion, 4, "0") * 
                            "_sys_dim_" * lpad(sys_dim, 2, "0") * 
                            "_obs_dim_" * lpad(obs_dim, 2, "0") * 
                            "_obs_un_" * rpad(obs_un, 4, "0") *
                            "_gamma_" * lpad(γ, 5, "0") *
                            "_nanl_" * lpad(nanl + burn, 5, "0") * 
                            "_tanl_" * rpad(tanl, 4, "0") * 
                            "_h_" * rpad(h, 4, "0") *
                            "_N_ens_" * lpad(N_ens, 3,"0") * 
                            "_state_inflation_" * rpad(round(infl, digits=2), 4, "0") * 
                            ".jld"
                    push!(fnames, fpath * method * "/diffusion_" * rpad(diffusion, 4, "0") * "/" * name)
                end
            end
        end

        # turn fnames into a string array, use this as the argument in process_data
        fnames = Array{String}(fnames)
        process_data(fnames, method)

    end

    # create jld file name with relevant parameters
    jlname = "processed_filter_state" * 
             "_diffusion_" * rpad(diffusion, 4, "0") *
             "_tanl_" * rpad(tanl, 4, "0") * 
             "_nanl_" * lpad(nanl, 5, "0") * 
             "_burn_" * lpad(burn, 5, "0") * 
             ".jld"

    # create hdf5 file name with relevant parameters
    h5name = "processed_filter_state" * 
             "_diffusion_" * rpad(diffusion, 4, "0") *
             "_tanl_" * rpad(tanl, 4, "0") * 
             "_nanl_" * lpad(nanl, 5, "0") * 
             "_burn_" * lpad(burn, 5, "0") * 
             ".h5"


    # write out file in jld
    save(jlname, data)

    # write out file in hdf5
    h5open(h5name, "w") do file
        for key in keys(data)
            h5write(h5name, key, data[key])
        end
    end
    print("Runtime " * string(round((time() - t1)  / 60.0, digits=4))  * " minutes\n")
end


########################################################################################################################

end
