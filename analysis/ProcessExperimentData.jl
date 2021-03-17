#######################################################################################################################
module ProcessExperimentData
########################################################################################################################
########################################################################################################################
# imports and exports
using Revise
using Debugger
using Random, Distributions, Statistics
using JLD
using LinearAlgebra
using HDF5
using Glob
export process_filter_state_glob, process_filter_state_strings, process_filter_param, process_classic_smoother_state, 
       process_classic_smoother_param, process_hybrid_smoother_state, process_all_smoother_state

########################################################################################################################
########################################################################################################################
# Scripts for processing experimental output data and writing to HDF5 to read into matplotlib later
########################################################################################################################

function process_filter_state_glob()
    # will create an array of the average RMSE and spread for each experiment, sorted by analysis or forecast step
    # ensemble size is increasing from the origin on the horizontal axis
    # inflation is increasing from the origin on the vertical axis
    
    # parameters for the file names and separating out experiments
    tanl = 0.10
    nanl = 40000
    burn = 5000
    diffusion = 0.0
    method_list = ["enkf-n", "enkf", "etkf"]
    ensemble_size = 28
    total_inflation = 21

    # define the storage dictionary here
    data = Dict{String, Array{Float64}}()
    for method in method_list
        if method == "enkf-n"
            data[method * "_anal_rmse"] = Array{Float64}(undef, ensemble_size)
            data[method * "_fore_rmse"] = Array{Float64}(undef, ensemble_size) 
            data[method * "_anal_spread"] = Array{Float64}(undef, ensemble_size)
            data[method * "_fore_spread"] = Array{Float64}(undef, ensemble_size)
        else
            data[method * "_anal_rmse"] = Array{Float64}(undef, total_inflation, ensemble_size)
            data[method * "_fore_rmse"] = Array{Float64}(undef, total_inflation, ensemble_size) 
            data[method * "_anal_spread"] = Array{Float64}(undef, total_inflation, ensemble_size)
            data[method * "_fore_spread"] = Array{Float64}(undef, total_inflation, ensemble_size)
        end
    end

    # auxilliary function to process data
    function process_data(fnames::Vector{String}, method::String)
        # loop columns
        for j in 0:(ensemble_size - 1)
            if method == "enkf-n"
                try
                    tmp = load(fnames[j+1])
                    
                    ana_rmse = tmp["anal_rmse"]::Vector{Float64}
                    ana_spread = tmp["anal_spread"]::Vector{Float64}

                    for_rmse = tmp["fore_rmse"]::Vector{Float64}
                    for_spread = tmp["fore_spread"]::Vector{Float64}

                    data[method * "_anal_rmse"][j+1] = mean(ana_rmse[burn+1: nanl+burn])
                    data[method * "_fore_rmse"][j+1] = mean(for_rmse[burn+1: nanl+burn])

                    data[method * "_fore_spread"][j+1] = mean(for_spread[burn+1: nanl+burn])
                    data[method * "_anal_spread"][j+1] = mean(ana_spread[burn+1: nanl+burn])
                catch
                    data[method * "_anal_rmse"][j+1] = Inf 
                    data[method * "_fore_rmse"][j+1] = Inf

                    data[method * "_fore_spread"][j+1] = Inf
                    data[method * "_anal_spread"][j+1] = Inf
                end
            else
                #loop rows
                for i in 1:total_inflation
                    try
                        tmp = load(fnames[i + j*total_inflation])
                        
                        ana_rmse = tmp["anal_rmse"]::Vector{Float64}
                        ana_spread = tmp["anal_spread"]::Vector{Float64}

                        for_rmse = tmp["fore_rmse"]::Vector{Float64}
                        for_spread = tmp["fore_spread"]::Vector{Float64}

                        data[method * "_anal_rmse"][total_inflation + 1 - i, j+1] = mean(ana_rmse[burn+1: nanl+burn])
                        data[method * "_anal_spread"][total_inflation + 1 - i, j+1] = mean(ana_spread[burn+1: nanl+burn])

                        data[method * "_fore_rmse"][total_inflation + 1 - i, j+1] = mean(for_rmse[burn+1: nanl+burn])
                        data[method * "_fore_spread"][total_inflation + 1 - i, j+1] = mean(for_spread[burn+1: nanl+burn])

                    catch
                        data[method * "_anal_rmse"][total_inflation + 1 - i, j+1] = Inf 
                        data[method * "_anal_spread"][total_inflation + 1 - i, j+1] = Inf

                        data[method * "_fore_rmse"][total_inflation + 1 - i, j+1] = Inf
                        data[method * "_fore_spread"][total_inflation + 1 - i, j+1] = Inf
                    end
                end
            end
        end
    end

    # for each DA method in the experiment, process the data, loading into the dictionary
    for method in method_list
        fnames = Glob.glob("../storage/filter_state/" * method * 
                           "/diffusion_" * rpad(diffusion, 4, "0") *
                            "/*_nanl_" * lpad(nanl + burn, 5, "0")  *  
                            "*_tanl_" * rpad(tanl, 4, "0") * 
                            "*" )

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

end


########################################################################################################################

function process_filter_state_strings()
    # will create an array of the average RMSE and spread for each experiment, sorted by analysis or forecast step
    # ensemble size is increasing from the origin on the horizontal axis
    # inflation is increasing from the origin on the vertical axis
    
    # parameters for the file names and separating out experiments
    tanl = 0.05
    nanl = 40000
    burn = 5000
    diffusion = 0.0
    h = 0.01
    obs_un = 1.0
    obs_dim = 40
    sys_dim = 40
    method_list = ["enkf-n", "enkf", "etkf"]
    ensemble_sizes = 14:41 
    ensemble_size = length(ensemble_sizes)
    total_inflations = LinRange(1.00, 1.20, 21)
    total_inflation = length(total_inflations)

    # define the storage dictionary here
    data = Dict{String, Array{Float64}}()
    for method in method_list
        if method == "enkf-n"
            data[method * "_anal_rmse"] = Array{Float64}(undef, ensemble_size)
            data[method * "_fore_rmse"] = Array{Float64}(undef, ensemble_size) 
            data[method * "_anal_spread"] = Array{Float64}(undef, ensemble_size)
            data[method * "_fore_spread"] = Array{Float64}(undef, ensemble_size)
        else
            data[method * "_anal_rmse"] = Array{Float64}(undef, total_inflation, ensemble_size)
            data[method * "_fore_rmse"] = Array{Float64}(undef, total_inflation, ensemble_size) 
            data[method * "_anal_spread"] = Array{Float64}(undef, total_inflation, ensemble_size)
            data[method * "_fore_spread"] = Array{Float64}(undef, total_inflation, ensemble_size)
        end
    end

    # auxilliary function to process data
    function process_data(fnames::Vector{String}, method::String)
        # loop columns
        for j in 0:(ensemble_size - 1)
            if method == "enkf-n"
                try
                    tmp = load(fnames[j+1])
                    
                    ana_rmse = tmp["anal_rmse"]::Vector{Float64}
                    ana_spread = tmp["anal_spread"]::Vector{Float64}

                    for_rmse = tmp["fore_rmse"]::Vector{Float64}
                    for_spread = tmp["fore_spread"]::Vector{Float64}

                    data[method * "_anal_rmse"][j+1] = mean(ana_rmse[burn+1: nanl+burn])
                    data[method * "_fore_rmse"][j+1] = mean(for_rmse[burn+1: nanl+burn])

                    data[method * "_fore_spread"][j+1] = mean(for_spread[burn+1: nanl+burn])
                    data[method * "_anal_spread"][j+1] = mean(ana_spread[burn+1: nanl+burn])
                catch
                    data[method * "_anal_rmse"][j+1] = Inf 
                    data[method * "_fore_rmse"][j+1] = Inf

                    data[method * "_fore_spread"][j+1] = Inf
                    data[method * "_anal_spread"][j+1] = Inf
                end
            else
                #loop rows
                for i in 1:total_inflation
                    try
                        tmp = load(fnames[i + j*total_inflation])
                        
                        ana_rmse = tmp["anal_rmse"]::Vector{Float64}
                        ana_spread = tmp["anal_spread"]::Vector{Float64}

                        for_rmse = tmp["fore_rmse"]::Vector{Float64}
                        for_spread = tmp["fore_spread"]::Vector{Float64}

                        data[method * "_anal_rmse"][total_inflation + 1 - i, j+1] = mean(ana_rmse[burn+1: nanl+burn])
                        data[method * "_anal_spread"][total_inflation + 1 - i, j+1] = mean(ana_spread[burn+1: nanl+burn])

                        data[method * "_fore_rmse"][total_inflation + 1 - i, j+1] = mean(for_rmse[burn+1: nanl+burn])
                        data[method * "_fore_spread"][total_inflation + 1 - i, j+1] = mean(for_spread[burn+1: nanl+burn])

                    catch
                        data[method * "_anal_rmse"][total_inflation + 1 - i, j+1] = Inf 
                        data[method * "_anal_spread"][total_inflation + 1 - i, j+1] = Inf

                        data[method * "_fore_rmse"][total_inflation + 1 - i, j+1] = Inf
                        data[method * "_fore_spread"][total_inflation + 1 - i, j+1] = Inf
                    end
                end
            end
        end
    end

    # for each DA method in the experiment, process the data, loading into the dictionary
    fpath = "/x/capc/cgrudzien/da_benchmark/storage/filter_state/"
    for method in method_list
        fnames = [] 
        for N_ens in ensemble_sizes
            if method == "enkf-n"
                name = method * "_filter_l96_state_benchmark_seed_0000_diffusion_" * rpad(diffusion, 4, "0") * 
                        "_sys_dim_" * lpad(sys_dim, 2, "0") * "_obs_dim_" * lpad(obs_dim, 2, "0") * "_obs_un_" * rpad(obs_un, 4, "0") *
                        "_nanl_" * lpad(nanl + burn, 5, "0") * "_tanl_" * rpad(tanl, 4, "0") * "_h_" * rpad(h, 4, "0") *
                        "_N_ens_" * lpad(N_ens, 3,"0") * "_state_inflation_" * rpad(round(1.0, digits=2), 4, "0") * ".jld"
                push!(fnames, fpath * method * "/diffusion_" * rpad(diffusion, 4, "0") * "/" * name)

            else
                for infl in total_inflations
                    name = method * "_filter_l96_state_benchmark_seed_0000_diffusion_" * rpad(diffusion, 4, "0") * 
                            "_sys_dim_" * lpad(sys_dim, 2, "0") * "_obs_dim_" * lpad(obs_dim, 2, "0") * "_obs_un_" * rpad(obs_un, 4, "0") *
                            "_nanl_" * lpad(nanl + burn, 5, "0") * "_tanl_" * rpad(tanl, 4, "0") * "_h_" * rpad(h, 4, "0") *
                            "_N_ens_" * lpad(N_ens, 3,"0") * "_state_inflation_" * rpad(round(infl, digits=2), 4, "0") * ".jld"
                    push!(fnames, fpath * method * "/diffusion_" * rpad(diffusion, 4, "0") * "/" * name)
                end
            end
        end
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

end


########################################################################################################################

function process_filter_param()
    # will create an array of the average RMSE and spread for each experiment, sorted by analysis or forecast step
    # ensemble size is increasing from the origin on the horizontal axis
    # inflation is increasing from the origin on the vertical axis

    # parameters for the file names and separating out experiments
    tanl = 0.05
    nanl = 40000
    burn = 5000
    diffusion = 0.00
    wlk = 0.0001
    method_list = ["enkf", "etkf"]
    ensemble_size = 28
    total_inflation = 21
    
    # define the storage dictionary here
    data = Dict{String, Array{Float64,2}}(
                "enkf_anal_rmse"   => Array{Float64}(undef, total_inflation, ensemble_size),
                "enkf_anal_spread" => Array{Float64}(undef, total_inflation, ensemble_size),
                "etkf_anal_rmse"   => Array{Float64}(undef, total_inflation, ensemble_size),
                "etkf_anal_spread" => Array{Float64}(undef, total_inflation, ensemble_size),
                "enkf_fore_rmse"   => Array{Float64}(undef, total_inflation, ensemble_size),
                "enkf_fore_spread" => Array{Float64}(undef, total_inflation, ensemble_size),
                "etkf_fore_rmse"   => Array{Float64}(undef, total_inflation, ensemble_size),
                "etkf_fore_spread" => Array{Float64}(undef, total_inflation, ensemble_size),
                "enkf_para_rmse"   => Array{Float64}(undef, total_inflation, ensemble_size),
                "enkf_para_spread" => Array{Float64}(undef, total_inflation, ensemble_size),
                "etkf_para_rmse"   => Array{Float64}(undef, total_inflation, ensemble_size),
                "etkf_para_spread" => Array{Float64}(undef, total_inflation, ensemble_size)
               )

    # auxilliary function to process data
    function process_data(fnames::Vector{String}, method::String)
        # loop columns
        for j in 0:27 
            #loop rows
            for i in 1:total_inflation
                tmp = load(fnames[i + j*total_inflation])
                
                ana_rmse = tmp["anal_rmse"]::Vector{Float64}
                ana_spread = tmp["anal_spread"]::Vector{Float64}

                for_rmse = tmp["fore_rmse"]::Vector{Float64}
                for_spread = tmp["fore_spread"]::Vector{Float64}

                par_rmse = tmp["param_rmse"]::Vector{Float64}
                par_spread = tmp["param_spread"]::Vector{Float64}

                data[method * "_anal_rmse"][total_inflation + 1 - i, j+1] = mean(ana_rmse[burn+1: nanl+burn])
                data[method * "_anal_spread"][total_inflation + 1 - i, j+1] = mean(ana_spread[burn+1: nanl+burn])

                data[method * "_fore_rmse"][total_inflation + 1 - i, j+1] = mean(for_rmse[burn+1: nanl+burn])
                data[method * "_fore_spread"][total_inflation + 1 - i, j+1] = mean(for_spread[burn+1: nanl+burn])

                data[method * "_para_rmse"][total_inflation + 1 - i, j+1] = mean(par_rmse[burn+1: nanl+burn])
                data[method * "_para_spread"][total_inflation + 1 - i, j+1] = mean(par_spread[burn+1: nanl+burn])
            end
        end
    end

    # for each DA method in the experiment, process the data, loading into the dictionary
    for method in method_list
        fnames = Glob.glob("../storage/filter_param/" * method * 
                           "/diffusion_" * rpad(diffusion, 4, "0") *
                           "/*_wlk_" * rpad(wlk, 6, "0") *
                           "*_nanl_" * lpad(nanl + burn, 5, "0")  *  
                           "*_tanl_" * rpad(tanl, 4, "0") * 
                           "*" )

        process_data(fnames, method)

    end
    
    # create jld file name with relevant parameters
    jlname = "processed_filter_param" * 
             "_diffusion_" * rpad(diffusion, 4, "0") * 
             "_wlk_" * rpad(wlk, 6, "0") *
             "_tanl_" * rpad(tanl, 4, "0") * 
             "_nanl_" * lpad(nanl, 5, "0") * 
             "_burn_" * lpad(burn, 5, "0") * 
             ".jld"
    
    # create hdf5 file name with relevant parameters
    h5name = "processed_filter_param" * 
             "_diffusion_" * rpad(diffusion, 4, "0") * 
             "_wlk_" * rpad(wlk, 6, "0") *
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

end


########################################################################################################################

function process_classic_smoother_state()
    # will create an array of the average RMSE and spread for each experiment, sorted by analysis or forecast step
    # ensemble size is increasing from the origin on the horizontal axis
    # inflation is increasing from the origin on the vertical axis
    
    # parameters for the file names and separating out experiments
    tanl = 0.05
    nanl = 40000
    burn = 5000
    diffusion = 0.0
    method_list = ["etks"]
    ensemble_size = 15
    total_lag = 18
    total_inflation = 11

    # define the storage dictionary here
    data = Dict{String, Array{Float64,3}}(
                #"enks_anal_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                #"enks_anal_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                #"enks_filt_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                #"enks_filt_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                #"enks_fore_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                #"enks_fore_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "etks_anal_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "etks_anal_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "etks_filt_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "etks_filt_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "etks_fore_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "etks_fore_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size)
               )

    # auxilliary function to process data
    function process_data(fnames::Vector{String}, method::String)
        # loop lag
        for k in 0:10
            # loop ensemble size 
            for j in 0:27 
                #loop inflation
                for i in 1:total_inflation
                    tmp = load(fnames[i + j*total_inflation + k*ensemble_size*total_inflation])
                    
                    ana_rmse = tmp["anal_rmse"]::Vector{Float64}
                    ana_spread = tmp["anal_spread"]::Vector{Float64}

                    for_rmse = tmp["fore_rmse"]::Vector{Float64}
                    for_spread = tmp["fore_spread"]::Vector{Float64}

                    fil_rmse = tmp["filt_rmse"]::Vector{Float64}
                    fil_spread = tmp["filt_spread"]::Vector{Float64}

                    data[method * "_anal_rmse"][total_lag - k, total_inflation + 1 - i, j+1] = mean(ana_rmse[burn+1: nanl+burn])
                    data[method * "_anal_spread"][total_lag - k, total_inflation + 1 - i, j+1] = mean(ana_spread[burn+1: nanl+burn])

                    data[method * "_fore_rmse"][total_lag - k, total_inflation + 1 - i, j+1] = mean(for_rmse[burn+1: nanl+burn])
                    data[method * "_fore_spread"][total_lag - k, total_inflation + 1 - i, j+1] = mean(for_spread[burn+1: nanl+burn])

                    data[method * "_filt_rmse"][total_lag - k, total_inflation + 1 - i, j+1] = mean(fil_rmse[burn+1: nanl+burn])
                    data[method * "_filt_spread"][total_lag - k, total_inflation + 1 - i, j+1] = mean(fil_spread[burn+1: nanl+burn])
                end
            end
        end
    end

    # for each DA method in the experiment, process the data, loading into the dictionary
    for method in method_list
        fnames = Glob.glob("../storage/smoother_state/" * method * "_classic" *
                           "/diffusion_" * rpad(diffusion, 4, "0") *
                            "/*_nanl_" * lpad(nanl + burn, 5, "0")  *  
                            "*_tanl_" * rpad(tanl, 4, "0") * 
                            "*" )

        process_data(fnames, method)

    end

    # create jld file name with relevant parameters
    jlname = "processed_classic_smoother_state" * 
             "_diffusion_" * rpad(diffusion, 4, "0") *
             "_tanl_" * rpad(tanl, 4, "0") * 
             "_nanl_" * lpad(nanl, 5, "0") * 
             "_burn_" * lpad(burn, 5, "0") * 
             ".jld"

    # create hdf5 file name with relevant parameters
    h5name = "processed_classic_smoother_state" * 
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

end


########################################################################################################################

function process_classic_smoother_param()
    # will create an array of the average RMSE and spread for each experiment, sorted by analysis or forecast step
    # ensemble size is increasing from the origin on the horizontal axis
    # inflation is increasing from the origin on the vertical axis
    
    # parameters for the file names and separating out experiments
    tanl = 0.05
    nanl = 40000
    burn = 5000
    diffusion = 0.0
    wlk = 0.0010
    method_list = ["enks", "etks"]
    ensemble_size = 28
    total_inflation = 21
    total_lag = 11

    # define the storage dictionary here
    data = Dict{String, Array{Float64,3}}(
                #"enks_anal_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                #"enks_anal_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                #"enks_para_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                #"enks_para_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                #"enks_filt_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                #"enks_filt_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                #"enks_fore_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                #"enks_fore_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "etks_anal_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "etks_anal_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "etks_para_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "etks_para_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "etks_filt_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "etks_filt_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "etks_fore_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "etks_fore_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size)
               )

    # auxilliary function to process data
    function process_data(fnames::Vector{String}, method::String)
        # loop lag
        for k in 0:total_lag - 1
            # loop ensemble size 
            for j in 0:ensemble_size - 1
                #loop inflation
                for i in 1:total_inflation
                    tmp = load(fnames[i + j*total_inflation + k*ensemble_size*total_inflation])
                    
                    ana_rmse = tmp["anal_rmse"]::Vector{Float64}
                    ana_spread = tmp["anal_spread"]::Vector{Float64}

                    par_rmse = tmp["param_rmse"]::Vector{Float64}
                    par_spread = tmp["param_spread"]::Vector{Float64}

                    for_rmse = tmp["fore_rmse"]::Vector{Float64}
                    for_spread = tmp["fore_spread"]::Vector{Float64}

                    fil_rmse = tmp["filt_rmse"]::Vector{Float64}
                    fil_spread = tmp["filt_spread"]::Vector{Float64}

                    data[method * "_anal_rmse"][total_lag - k, total_inflation + 1 - i, j+1] = mean(ana_rmse[burn+1: nanl+burn])
                    data[method * "_anal_spread"][total_lag - k, total_inflation + 1 - i, j+1] = mean(ana_spread[burn+1: nanl+burn])

                    data[method * "_para_rmse"][total_lag - k, total_inflation + 1 - i, j+1] = mean(par_rmse[burn+1: nanl+burn])
                    data[method * "_para_spread"][total_lag - k, total_inflation + 1 - i, j+1] = mean(par_spread[burn+1: nanl+burn])

                    data[method * "_fore_rmse"][total_lag - k, total_inflation + 1 - i, j+1] = mean(for_rmse[burn+1: nanl+burn])
                    data[method * "_fore_spread"][total_lag - k, total_inflation + 1 - i, j+1] = mean(for_spread[burn+1: nanl+burn])

                    data[method * "_filt_rmse"][total_lag - k, total_inflation + 1 - i, j+1] = mean(fil_rmse[burn+1: nanl+burn])
                    data[method * "_filt_spread"][total_lag - k, total_inflation + 1 - i, j+1] = mean(fil_spread[burn+1: nanl+burn])
                end
            end
        end
    end

    # for each DA method in the experiment, process the data, loading into the dictionary
    for method in method_list
        fnames = Glob.glob("../storage/smoother_param/" * method * "_classic" *
                           "/diffusion_" * rpad(diffusion, 4, "0") *
                            "/*_wlk_" * rpad(wlk, 6, "0") *
                            "*_nanl_" * lpad(nanl + burn, 5, "0")  *  
                            "*_tanl_" * rpad(tanl, 4, "0") * 
                            "*" )

        process_data(fnames, method)

    end

    # create jld file name with relevant parameters
    jlname = "processed_classic_smoother_param" * 
             "_diffusion_" * rpad(diffusion, 4, "0") *
             "_wlk_" * rpad(wlk, 6, "0") *
             "_tanl_" * rpad(tanl, 4, "0") * 
             "_nanl_" * lpad(nanl, 5, "0") * 
             "_burn_" * lpad(burn, 5, "0") * 
             ".jld"

    # create hdf5 file name with relevant parameters
    h5name = "processed_classic_smoother_param" * 
             "_diffusion_" * rpad(diffusion, 4, "0") *
             "_wlk_" * rpad(wlk, 6, "0") *
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

end


########################################################################################################################

function process_hybrid_smoother_state()
    # will create an array of the average RMSE and spread for each experiment, sorted by analysis or forecast step
    # ensemble size is increasing from the origin on the horizontal axis
    # inflation is increasing from the origin on the vertical axis
    
    # parameters for the file names and separating out experiments
    tanl = 0.05
    nanl = 20000
    burn = 5000
    diffusion = 0.0
    mda = false 
    method_list = ["etks"]
    total_inflation = 11
    ensemble_size = 15
    total_lag = 18

    # define the storage dictionary here
    data = Dict{String, Array{Float64,3}}(
                #"enks_anal_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                #"enks_anal_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                #"enks_filt_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                #"enks_filt_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                #"enks_fore_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                #"enks_fore_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "etks_anal_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "etks_anal_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "etks_filt_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "etks_filt_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "etks_fore_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "etks_fore_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size)
               )

    # auxilliary function to process data
    function process_data(fnames::Vector{String}, method::String)
        # loop lag
        for k in 0:total_lag - 1
            # loop ensemble size 
            for j in 0:ensemble_size - 1
                #loop inflation
                for i in 1:total_inflation
                    tmp = load(fnames[i + j*total_inflation + k*ensemble_size*total_inflation])
                    
                    ana_rmse = tmp["anal_rmse"]::Vector{Float64}
                    ana_spread = tmp["anal_spread"]::Vector{Float64}

                    for_rmse = tmp["fore_rmse"]::Vector{Float64}
                    for_spread = tmp["fore_spread"]::Vector{Float64}

                    fil_rmse = tmp["filt_rmse"]::Vector{Float64}
                    fil_spread = tmp["filt_spread"]::Vector{Float64}

                    data[method * "_anal_rmse"][total_lag - k, total_inflation + 1- i, j+1] = mean(ana_rmse[burn+1: nanl+burn])
                    data[method * "_anal_spread"][total_lag - k, total_inflation + 1 - i, j+1] = mean(ana_spread[burn+1: nanl+burn])

                    data[method * "_fore_rmse"][total_lag - k, total_inflation + 1 - i, j+1] = mean(for_rmse[burn+1: nanl+burn])
                    data[method * "_fore_spread"][total_lag - k, total_inflation + 1 - i, j+1] = mean(for_spread[burn+1: nanl+burn])

                    data[method * "_filt_rmse"][total_lag - k, total_inflation + 1 - i, j+1] = mean(fil_rmse[burn+1: nanl+burn])
                    data[method * "_filt_spread"][total_lag - k, total_inflation + 1 - i, j+1] = mean(fil_spread[burn+1: nanl+burn])
                end
            end
        end
    end

    # for each DA method in the experiment, process the data, loading into the dictionary
    for method in method_list
        fnames = Glob.glob("../storage/smoother_state/" * method * "_hybrid" *
                           "/diffusion_" * rpad(diffusion, 4, "0") *
                            "/*_nanl_" * lpad(nanl + burn, 5, "0")  *  
                            "*_tanl_" * rpad(tanl, 4, "0") * 
                            "*_mda_" * string(mda) * 
                            "*" )

        process_data(fnames, method)

    end

    # create jld file name with relevant parameters
    jlname = "processed_hybrid_smoother_state" * 
             "_diffusion_" * rpad(diffusion, 4, "0") *
             "_tanl_" * rpad(tanl, 4, "0") * 
             "_nanl_" * lpad(nanl, 5, "0") * 
             "_burn_" * lpad(burn, 5, "0") * 
             "_mda_" * string(mda) * 
             ".jld"

    # create hdf5 file name with relevant parameters
    h5name = "processed_hybrid_smoother_state" * 
             "_diffusion_" * rpad(diffusion, 4, "0") *
             "_tanl_" * rpad(tanl, 4, "0") * 
             "_nanl_" * lpad(nanl, 5, "0") * 
             "_burn_" * lpad(burn, 5, "0") * 
             "_mda_" * string(mda) * 
             ".h5"


    # write out file in jld
    save(jlname, data)

    # write out file in hdf5
    h5open(h5name, "w") do file
        for key in keys(data)
            h5write(h5name, key, data[key])
        end
    end

end


########################################################################################################################

function process_iterative_smoother_state()
    # will create an array of the average RMSE and spread for each experiment, sorted by analysis or forecast step
    # ensemble size is increasing from the origin on the horizontal axis
    # inflation is increasing from the origin on the vertical axis
    
    # parameters for the file names and separating out experiments
    tanl = 0.05
    nanl = 40
    burn = 5
    diffusion = 0.0
    mda = false
    method_list = ["ienks-transform"]
    total_inflation = 11
    ensemble_size = 15
    total_lag = 11

    # define the storage dictionary here
    data = Dict{String, Array{Float64,3}}(
                #"ienks-transform_anal_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                #"ienks-transform_anal_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                #"ienks-transform_filt_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                #"ienks-transform_filt_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                #"ienks-transform_fore_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                #"ienks-transform_fore_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size)
                "ienks-bundle_anal_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "ienks-bundle_anal_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "ienks-bundle_filt_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "ienks-bundle_filt_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "ienks-bundle_fore_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "ienks-bundle_fore_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size)
               )

    # auxilliary function to process data
    function process_data(fnames::Vector{String}, method::String)
        # loop lag
        for k in 0:total_lag - 1
            # loop ensemble size 
            for j in 0:ensemble_size - 1
                #loop inflation
                for i in 1:total_inflation
                    tmp = load(fnames[i + j*total_inflation + k*ensemble_size*total_inflation])
                    
                    ana_rmse = tmp["anal_rmse"]::Vector{Float64}
                    ana_spread = tmp["anal_spread"]::Vector{Float64}

                    for_rmse = tmp["fore_rmse"]::Vector{Float64}
                    for_spread = tmp["fore_spread"]::Vector{Float64}

                    fil_rmse = tmp["filt_rmse"]::Vector{Float64}
                    fil_spread = tmp["filt_spread"]::Vector{Float64}

                    data[method * "_anal_rmse"][total_lag - k, total_inflation + 1- i, j+1] = mean(ana_rmse[burn+1: nanl+burn])
                    data[method * "_anal_spread"][total_lag - k, total_inflation + 1 - i, j+1] = mean(ana_spread[burn+1: nanl+burn])

                    data[method * "_fore_rmse"][total_lag - k, total_inflation + 1 - i, j+1] = mean(for_rmse[burn+1: nanl+burn])
                    data[method * "_fore_spread"][total_lag - k, total_inflation + 1 - i, j+1] = mean(for_spread[burn+1: nanl+burn])

                    data[method * "_filt_rmse"][total_lag - k, total_inflation + 1 - i, j+1] = mean(fil_rmse[burn+1: nanl+burn])
                    data[method * "_filt_spread"][total_lag - k, total_inflation + 1 - i, j+1] = mean(fil_spread[burn+1: nanl+burn])
                end
            end
        end
    end

    # for each DA method in the experiment, process the data, loading into the dictionary
    for method in method_list
        fnames = Glob.glob("../storage/smoother_state/" * method *
                           "/diffusion_" * rpad(diffusion, 4, "0") *
                            "/*_nanl_" * lpad(nanl + burn, 5, "0")  *  
                            "*_tanl_" * rpad(tanl, 4, "0") * 
                            "*_mda_" * string(mda) * 
                            "*" )

        process_data(fnames, method)

    end

    # create jld file name with relevant parameters
    jlname = "processed_iterative_smoother_state" * 
             "_diffusion_" * rpad(diffusion, 4, "0") *
             "_tanl_" * rpad(tanl, 4, "0") * 
             "_nanl_" * lpad(nanl, 5, "0") * 
             "_burn_" * lpad(burn, 5, "0") * 
             "_mda_" * string(mda) * 
             ".jld"

    # create hdf5 file name with relevant parameters
    h5name = "processed_iterative_smoother_state" * 
             "_diffusion_" * rpad(diffusion, 4, "0") *
             "_tanl_" * rpad(tanl, 4, "0") * 
             "_nanl_" * lpad(nanl, 5, "0") * 
             "_burn_" * lpad(burn, 5, "0") * 
             "_mda_" * string(mda) * 
             ".h5"


    # write out file in jld
    save(jlname, data)

    # write out file in hdf5
    h5open(h5name, "w") do file
        for key in keys(data)
            h5write(h5name, key, data[key])
        end
    end

end


########################################################################################################################

function process_all_smoother_state()
    # will create an array of the average RMSE and spread for each experiment, sorted by analysis filter or 
    # forecast step ensemble size is increasing from the origin on the horizontal axis
    # inflation is increasing from the origin in the middle axis, lag is increasing from the origin
    # on the left axis
    
    # parameters for the file names and separating out experiments
    t1 = time()
    tanl = 0.05
    h = 0.01
    obs_un = 1.0
    obs_dim = 40
    sys_dim = 40
    nanl = 20000
    burn = 5000
    shift = 1
    mda = false
    diffusion = 0.00
    method_list = [
                   #"enks-n_hybrid", 
                   "etks_adaptive_hybrid", 
                   #"etks_hybrid", 
                   #"etks_classic", 
                   #"ienks-bundle", 
                   #"ienks-transform"
                  ]
    ensemble_sizes = 15:2:43 
    ensemble_size = length(ensemble_sizes)
    total_inflations = LinRange(1.00, 1.10, 11)
    total_inflation = length(total_inflations)
    total_lags = 1:3:52
    total_lag = length(total_lags)

    # define the storage dictionary here
    data = Dict{String, Array{Float64}}()
    for method in method_list
        if method == "enks-n_hybrid" || method == "etks_adaptive_hybrid"
            data[method * "_anal_rmse"] = Array{Float64}(undef, total_lag, ensemble_size)
            data[method * "_filt_rmse"] = Array{Float64}(undef, total_lag, ensemble_size)
            data[method * "_fore_rmse"] = Array{Float64}(undef, total_lag, ensemble_size) 
            data[method * "_anal_spread"] = Array{Float64}(undef, total_lag, ensemble_size)
            data[method * "_filt_spread"] = Array{Float64}(undef, total_lag, ensemble_size)
            data[method * "_fore_spread"] = Array{Float64}(undef, total_lag, ensemble_size)
        else
            data[method * "_anal_rmse"] = Array{Float64}(undef, total_lag, total_inflation, ensemble_size)
            data[method * "_filt_rmse"] = Array{Float64}(undef, total_lag, total_inflation, ensemble_size) 
            data[method * "_fore_rmse"] = Array{Float64}(undef, total_lag, total_inflation, ensemble_size) 
            data[method * "_anal_spread"] = Array{Float64}(undef, total_lag, total_inflation, ensemble_size)
            data[method * "_filt_spread"] = Array{Float64}(undef, total_lag, total_inflation, ensemble_size)
            data[method * "_fore_spread"] = Array{Float64}(undef, total_lag, total_inflation, ensemble_size)
        end
    end

    # auxilliary function to process data
    function process_data(fnames::Vector{String}, method::String)
        # loop lag
        for k in 0:total_lag - 1
            # loop ensemble size 
            for j in 0:ensemble_size - 1
                if method == "enks-n_hybrid" || method == "etks_adaptive_hybrid"
                    try
                        tmp = load(fnames[1+j+k*ensemble_size])
                        
                        ana_rmse = tmp["anal_rmse"]::Vector{Float64}
                        ana_spread = tmp["anal_spread"]::Vector{Float64}

                        fil_rmse = tmp["filt_rmse"]::Vector{Float64}
                        fil_spread = tmp["filt_spread"]::Vector{Float64}

                        for_rmse = tmp["fore_rmse"]::Vector{Float64}
                        for_spread = tmp["fore_spread"]::Vector{Float64}

                        data[method * "_anal_rmse"][total_lag - k, j+1] = mean(ana_rmse[burn+1: nanl+burn])
                        data[method * "_filt_rmse"][total_lag - k, j+1] = mean(fil_rmse[burn+1: nanl+burn])
                        data[method * "_fore_rmse"][total_lag - k, j+1] = mean(for_rmse[burn+1: nanl+burn])

                        data[method * "_anal_spread"][total_lag - k, j+1] = mean(ana_spread[burn+1: nanl+burn])
                        data[method * "_filt_spread"][total_lag - k, j+1] = mean(fil_spread[burn+1: nanl+burn])
                        data[method * "_fore_spread"][total_lag - k, j+1] = mean(for_spread[burn+1: nanl+burn])
                    catch
                        data[method * "_anal_rmse"][total_lag - k, j+1] = Inf 
                        data[method * "_filt_rmse"][total_lag - k, j+1] = Inf
                        data[method * "_fore_rmse"][total_lag - k, j+1] = Inf

                        data[method * "_anal_spread"][total_lag - k, j+1] = Inf
                        data[method * "_filt_spread"][total_lag - k, j+1] = Inf
                        data[method * "_fore_spread"][total_lag - k, j+1] = Inf
                    end
                else
                    #loop inflation
                    for i in 1:total_inflation
                        try
                            tmp = load(fnames[i + j*total_inflation + k*ensemble_size*total_inflation])
                            
                            ana_rmse = tmp["anal_rmse"]::Vector{Float64}
                            ana_spread = tmp["anal_spread"]::Vector{Float64}

                            for_rmse = tmp["fore_rmse"]::Vector{Float64}
                            for_spread = tmp["fore_spread"]::Vector{Float64}

                            fil_rmse = tmp["filt_rmse"]::Vector{Float64}
                            fil_spread = tmp["filt_spread"]::Vector{Float64}

                            data[method * "_anal_rmse"][total_lag - k, total_inflation + 1- i, j+1] = mean(ana_rmse[burn+1: nanl+burn])
                            data[method * "_anal_spread"][total_lag - k, total_inflation + 1 - i, j+1] = mean(ana_spread[burn+1: nanl+burn])

                            data[method * "_fore_rmse"][total_lag - k, total_inflation + 1 - i, j+1] = mean(for_rmse[burn+1: nanl+burn])
                            data[method * "_fore_spread"][total_lag - k, total_inflation + 1 - i, j+1] = mean(for_spread[burn+1: nanl+burn])

                            data[method * "_filt_rmse"][total_lag - k, total_inflation + 1 - i, j+1] = mean(fil_rmse[burn+1: nanl+burn])
                            data[method * "_filt_spread"][total_lag - k, total_inflation + 1 - i, j+1] = mean(fil_spread[burn+1: nanl+burn])
                        catch
                            data[method * "_anal_rmse"][total_lag - k, total_inflation + 1- i, j+1] = Inf 
                            data[method * "_anal_spread"][total_lag - k, total_inflation + 1 - i, j+1] = Inf

                            data[method * "_fore_rmse"][total_lag - k, total_inflation + 1 - i, j+1] = Inf
                            data[method * "_fore_spread"][total_lag - k, total_inflation + 1 - i, j+1] = Inf

                            data[method * "_filt_rmse"][total_lag - k, total_inflation + 1 - i, j+1] = Inf
                            data[method * "_filt_spread"][total_lag - k, total_inflation + 1 - i, j+1] = Inf
                        end
                    end
                end
            end
        end
    end

    # for each DA method in the experiment, process the data, loading into the dictionary
    fpath = "/x/capc/cgrudzien/da_benchmark/storage/smoother_state/"
    for method in method_list
        fnames = []
        for lag in total_lags
            for N_ens in ensemble_sizes
                if method == "enks-n_hybrid" || method == "etks_adaptive_hybrid"
                    name = method * 
                            "_smoother_l96_state_benchmark_seed_0000"  *
                            "_sys_dim_" * lpad(sys_dim, 2, "0") * 
                            "_obs_dim_" * lpad(obs_dim, 2, "0") * 
                            "_obs_un_" * rpad(obs_un, 4, "0") *
                            "_nanl_" * lpad(nanl + burn, 5, "0") * 
                            "_tanl_" * rpad(tanl, 4, "0") * 
                            "_h_" * rpad(h, 4, "0") *
                            "_lag_" * lpad(lag, 3, "0") * 
                            "_shift_" * lpad(shift, 3, "0") * 
                            "_mda_" * string(mda) *
                            "_N_ens_" * lpad(N_ens, 3,"0") * 
                            "_state_inflation_" * rpad(round(1.00, digits=2), 4, "0") * 
                            ".jld"

                    push!(fnames, fpath * method * "/diffusion_" * rpad(diffusion, 4, "0") * "/" * name)
                else
                    for infl in total_inflations
                        if method[end-6:end] == "classic"
                            name = method * 
                                    "_smoother_l96_state_benchmark_seed_0000" *
                                    "_sys_dim_" * lpad(sys_dim, 2, "0") * 
                                    "_obs_dim_" * lpad(obs_dim, 2, "0") * 
                                    "_obs_un_" * rpad(obs_un, 4, "0") *
                                    "_nanl_" * lpad(nanl + burn, 5, "0") * 
                                    "_tanl_" * rpad(tanl, 4, "0") * 
                                    "_h_" * rpad(h, 4, "0") *
                                    "_lag_" * lpad(lag, 3, "0") * 
                                    "_shift_" * lpad(shift, 3, "0") *
                                    "_N_ens_" * lpad(N_ens, 3,"0") * 
                                    "_state_inflation_" * rpad(round(infl, digits=2), 4, "0") * 
                                    ".jld"

                            push!(fnames, fpath * method * "/diffusion_" * rpad(diffusion, 4, "0") * "/" * name)
                        elseif method[end-5:end] == "hybrid"
                            name = method * 
                                    "_smoother_l96_state_benchmark_seed_0000" *
                                    "_sys_dim_" * lpad(sys_dim, 2, "0") * 
                                    "_obs_dim_" * lpad(obs_dim, 2, "0") * 
                                    "_obs_un_" * rpad(obs_un, 4, "0") *
                                    "_nanl_" * lpad(nanl + burn, 5, "0") * 
                                    "_tanl_" * rpad(tanl, 4, "0") * 
                                    "_h_" * rpad(h, 4, "0") *
                                    "_lag_" * lpad(lag, 3, "0") * 
                                    "_shift_" * lpad(shift, 3, "0") * 
                                    "_mda_" * string(mda) *
                                    "_N_ens_" * lpad(N_ens, 3,"0") * 
                                    "_state_inflation_" * rpad(round(infl, digits=2), 4, "0") * 
                                    ".jld"
                            
                            push!(fnames, fpath * method * "/diffusion_" * rpad(diffusion, 4, "0") * "/" * name)
                        else
                            name = method * 
                                    "_l96_state_benchmark_seed_0000" *
                                    "_sys_dim_" * lpad(sys_dim, 2, "0") * 
                                    "_obs_dim_" * lpad(obs_dim, 2, "0") * 
                                    "_obs_un_" * rpad(obs_un, 4, "0") *
                                    "_nanl_" * lpad(nanl + burn, 5, "0") * 
                                    "_tanl_" * rpad(tanl, 4, "0") * 
                                    "_h_" * rpad(h, 4, "0") *
                                    "_lag_" * lpad(lag, 3, "0") * 
                                    "_shift_" * lpad(shift, 3, "0") * 
                                    "_mda_" * string(mda) *
                                    "_N_ens_" * lpad(N_ens, 3,"0") * 
                                    "_state_inflation_" * rpad(round(infl, digits=2), 4, "0") * 
                                    ".jld"
                            
                            push!(fnames, fpath * method * "/diffusion_" * rpad(diffusion, 4, "0") * "/" * name)
                        end
                    end
                end
            end
        end
        fnames = Array{String}(fnames)
        process_data(fnames, method)
    end

    # create jld file name with relevant parameters
    jlname = "processed_smoother_state" * 
             "_diffusion_" * rpad(diffusion, 4, "0") *
             "_tanl_" * rpad(tanl, 4, "0") * 
             "_nanl_" * lpad(nanl, 5, "0") * 
             "_burn_" * lpad(burn, 5, "0") * 
             "_mda_" * string(mda) * 
             ".jld"

    # create hdf5 file name with relevant parameters
    h5name = "processed_smoother_state" * 
             "_diffusion_" * rpad(diffusion, 4, "0") *
             "_tanl_" * rpad(tanl, 4, "0") * 
             "_nanl_" * lpad(nanl, 5, "0") * 
             "_burn_" * lpad(burn, 5, "0") * 
             "_mda_" * string(mda) * 
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
process_all_smoother_state()

end
