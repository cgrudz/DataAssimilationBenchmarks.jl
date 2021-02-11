#######################################################################################################################
module ProcessExperimentData
########################################################################################################################
########################################################################################################################
# imports and exports
using Debugger
using Random, Distributions, Statistics
using JLD
using LinearAlgebra
using HDF5
using Glob
export process_filter_state, process_filter_param, process_classic_smoother_state, process_classic_smoother_param,
       process_hybrid_smoother_state

########################################################################################################################
########################################################################################################################
# Scripts for processing experimental output data and writing to HDF5 to read into matplotlib later
########################################################################################################################
function process_filter_state()
    # will create an array of the average RMSE and spread for each experiment, sorted by analysis or forecast step
    # ensemble size is increasing from the origin on the horizontal axis
    # inflation is increasing from the origin on the vertical axis
    
    # parameters for the file names and separating out experiments
    tanl = 0.05
    nanl = 40000
    burn = 5000
    diffusion = 0.0
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
                "etkf_fore_spread" => Array{Float64}(undef, total_inflation, ensemble_size)
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

                data[method * "_anal_rmse"][total_inflation + 1 - i, j+1] = mean(ana_rmse[burn+1: nanl+burn])
                data[method * "_anal_spread"][total_inflation + 1 - i, j+1] = mean(ana_spread[burn+1: nanl+burn])

                data[method * "_fore_rmse"][total_inflation + 1 - i, j+1] = mean(for_rmse[burn+1: nanl+burn])
                data[method * "_fore_spread"][total_inflation + 1 - i, j+1] = mean(for_spread[burn+1: nanl+burn])
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
    method_list = ["enks", "etks"]
    ensemble_size = 28
    total_inflation = 21

    # define the storage dictionary here
    data = Dict{String, Array{Float64,3}}(
                "enks_anal_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "enks_anal_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "etks_anal_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "etks_anal_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "enks_filt_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "enks_filt_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "etks_filt_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "etks_filt_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "enks_fore_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "enks_fore_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
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
                "enks_anal_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "enks_anal_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "etks_anal_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "etks_anal_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "enks_para_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "enks_para_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "etks_para_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "etks_para_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "enks_filt_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "enks_filt_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "etks_filt_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "etks_filt_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "enks_fore_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "enks_fore_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
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
    mda = true
    method_list = ["etks"]
    total_inflation = 11
    ensemble_size = 15
    total_lag = 11
    @bp

    # define the storage dictionary here
    data = Dict{String, Array{Float64,3}}(
                "enks_anal_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "enks_anal_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "etks_anal_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "etks_anal_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "enks_filt_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "enks_filt_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "etks_filt_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "etks_filt_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "enks_fore_rmse"   => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
                "enks_fore_spread" => Array{Float64}(undef, total_lag, total_inflation, ensemble_size),
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
process_hybrid_smoother_state()

end
