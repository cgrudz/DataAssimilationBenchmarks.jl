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
export process_filter_state

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

    # define the storage dictionary here
    data = Dict{String, Array{Float64,2}}(
                "enkf_anal_rmse"   => Array{Float64}(undef, 21, 28),
                "enkf_anal_spread" => Array{Float64}(undef, 21, 28),
                "etkf_anal_rmse"   => Array{Float64}(undef, 21, 28),
                "etkf_anal_spread" => Array{Float64}(undef, 21, 28),
                "enkf_fore_rmse"   => Array{Float64}(undef, 21, 28),
                "enkf_fore_spread" => Array{Float64}(undef, 21, 28),
                "etkf_fore_rmse"   => Array{Float64}(undef, 21, 28),
                "etkf_fore_spread" => Array{Float64}(undef, 21, 28)
               )

    # auxilliary function to process data
    function process_data(fnames::Vector{String}, method::String)
        # loop columns
        for j in 0:27 
            #loop rows
            for i in 1:21
                tmp = load(fnames[i + j*21])
                
                ana_rmse = tmp["anal_rmse"]::Vector{Float64}
                ana_spread = tmp["anal_spread"]::Vector{Float64}

                for_rmse = tmp["fore_rmse"]::Vector{Float64}
                for_spread = tmp["fore_spread"]::Vector{Float64}

                data[method * "_anal_rmse"][22 - i, j+1] = mean(ana_rmse[burn+1: nanl+burn])
                data[method * "_anal_spread"][22 - i, j+1] = mean(ana_spread[burn+1: nanl+burn])

                data[method * "_fore_rmse"][22 - i, j+1] = mean(for_rmse[burn+1: nanl+burn])
                data[method * "_fore_spread"][22 - i, j+1] = mean(for_spread[burn+1: nanl+burn])
            end
        end
    end

    # for each DA method in the experiment, process the data, loading into the dictionary
    for method in method_list
        fnames = Glob.glob("../storage/filter_state/" * method * 
                           "/*_diffusion_" * rpad(diffusion, 4, "0") *
                            "*_nanl_" * lpad(nanl + burn, 5, "0")  *  
                            "_tanl_" * rpad(tanl, 4, "0") * 
                            "*" )

        process_data(fnames, method)

    end

    # create file name with relevant parameters
    fname = "processed_filter_state_" * 
            "_diffusion_" * rpad(diffusion, 4, "0") *
            "_tanl_" * rpad(tanl, 4, "0") * 
            "_nanl_" * lpad(nanl, 5, "0") * 
            "_burn_" * rpad(tanl, 5, "0") * 
            ".h5"


    # write out file in hdf5
    h5open(fname, "w") do file
        for key in keys(data)
            h5write(fname, key, data[key])
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
    diffusion = 0.0
    wlk = 0.001
    method_list = ["enkf", "etkf"]
    
    # define the storage dictionary here
    data = Dict{String, Array{Float64,2}}(
                "enkf_anal_rmse"   => Array{Float64}(undef, 21, 28),
                "enkf_anal_spread" => Array{Float64}(undef, 21, 28),
                "etkf_anal_rmse"   => Array{Float64}(undef, 21, 28),
                "etkf_anal_spread" => Array{Float64}(undef, 21, 28),
                "enkf_fore_rmse"   => Array{Float64}(undef, 21, 28),
                "enkf_fore_spread" => Array{Float64}(undef, 21, 28),
                "etkf_fore_rmse"   => Array{Float64}(undef, 21, 28),
                "etkf_fore_spread" => Array{Float64}(undef, 21, 28)
                "enkf_para_rmse"   => Array{Float64}(undef, 21, 28),
                "enkf_para_spread" => Array{Float64}(undef, 21, 28),
                "etkf_para_rmse"   => Array{Float64}(undef, 21, 28),
                "etkf_para_spread" => Array{Float64}(undef, 21, 28)
               )

    # auxilliary function to process data
    function process_data(fnames::Vector{String}, method::String)
        # loop columns
        for j in 0:27 
            #loop rows
            for i in 1:21
                tmp = load(fnames[i + j*21])
                
                ana_rmse = tmp["anal_rmse"]::Vector{Float64}
                ana_spread = tmp["anal_spread"]::Vector{Float64}

                for_rmse = tmp["fore_rmse"]::Vector{Float64}
                for_spread = tmp["fore_spread"]::Vector{Float64}

                data[method * "_anal_rmse"][22 - i, j+1] = mean(ana_rmse[burn+1: nanl+burn])
                data[method * "_anal_spread"][22 - i, j+1] = mean(ana_spread[burn+1: nanl+burn])

                data[method * "_fore_rmse"][22 - i, j+1] = mean(for_rmse[burn+1: nanl+burn])
                data[method * "_fore_spread"][22 - i, j+1] = mean(for_spread[burn+1: nanl+burn])
            end
        end
    end

    # for each DA method in the experiment, process the data, loading into the dictionary
    for method in method_list
        fnames = Glob.glob("../storage/filter_param/" * method * 
                           "/*_diffusion_" * rpad(diffusion, 4, "0") *
                           "_wlk_" * rpad(param_wlk, 6, "0") *
                           "*_nanl_" * lpad(nanl + burn, 5, "0")  *  
                           "_tanl_" * rpad(tanl, 4, "0") * 
                           "*" )

        process_data(fnames, method)

    end
    
    # create file name with relevant parameters
    fname = "processed_filter_param_" * 
            "_diffusion_" * rpad(diffusion, 4, "0") * 
            "_wlk_" * rpad(param_wlk, 6, "0") *
            "_tanl_" * rpad(tanl, 4, "0") * 
            "_nanl_" * lpad(nanl, 5, "0") * 
            "_burn_" * rpad(tanl, 5, "0") * 
            ".h5"
    
    # write out file in hdf5
    h5open(fname, "w") do file
        for key in keys(data)
            h5write(fname, key, data[key])
        end
    end

end

end
