#######################################################################################################################
module DataAssimilationBenchmarks
########################################################################################################################
########################################################################################################################
# imports and exports
include("../methods/DeSolvers.jl")
include("../methods/EnsembleKalmanSchemes.jl")
include("../models/L96.jl")

########################################################################################################################
########################################################################################################################

function __init__()

    print("  _____        _                         ")
    printstyled("_",color=9)
    print("           ")
    printstyled("_",color=2)
    print(" _       _   ")
    printstyled("_              \n",color=13) 
    print(" |  __ \\      | |          /\\           ")
    printstyled("(_)",color=9)
    printstyled("         (_)",color=2)
    print(" |     | | ")
    printstyled("(_)             \n",color=13)  
    print(" | |  | | __ _| |_ __ _   /  \\   ___ ___ _ _ __ ___  _| | __ _| |_ _  ___  _ __   \n")
    print(" | |  | |/ _` | __/ _` | / /\\ \\ / __/ __| | '_ ` _ \\| | |/ _` | __| |/ _ \\| '_ \\  \n")
    print(" | |__| | (_| | || (_| |/ ____ \\\\__ \\__ \\ | | | | | | | | (_| | |_| | (_) | | | | \n")
    print(" |_____/ \\__,_|\\__\\__,_/_/    \\_\\___/___/_|_| |_| |_|_|_|\\__,_|\\__|_|\\___/|_| |_| \n")
    print("\n")
    print("  ____                  _                          _         ")
    printstyled(" _ ", color=12)
    print("_\n")
    print(" |  _ \\                | |                        | |        ")
    printstyled("(_)",color=12)
    print(" |                \n")
    print(" | |_) | ___ _ __   ___| |__  _ __ ___   __ _ _ __| | _____   _| |                \n")
    print(" |  _ < / _ \\ '_ \\ / __| '_ \\| '_ ` _ \\ / _` | '__| |/ / __| | | |                \n")
    print(" | |_) |  __/ | | | (__| | | | | | | | | (_| | |  |   <\\__ \\_| | |                \n")
    print(" |____/ \\___|_| |_|\\___|_| |_|_| |_| |_|\\__,_|_|  |_|\\_\\___(_) |_|                \n")
    print("                                                            _/ |                  \n")
    print("                                                           |__/                   \n")

    print("\n")
    printstyled(" Welcome to DataAssimilationBenchmarks!\n", bold=true)
    print(" This is a wrapper library including the core numerical solvers for ordinary and stochastic differential \n")
    print(" equations, solvers for data assimilation routines and the core process model code for running twin\n")
    print(" experiments with benchmark models. These methods can be run stand-alone in other programs by calling\n") 
    print(" these functions from the DeSolvers, EnsembleKalmanSchemes and L96 sub-modules from this library.\n")
    print(" Future solvers and models will be added as sub-modules in the methods and models directories respectively.\n")
    print("\n")
    print(" In order to get the full functionality of this package you you will need to install the dev version.\n")
    print(" This provides the access to edit all of the outer-loop routines for setting up twin experiments. \n")
    print(" These routines are defined in the modules in the \"experiments\" directory.  The \"slurm_submit_scripts\"\n")
    print(" directory includes routines for parallel submission of experiments in Slurm.  Data processing scripts\n")
    print(" and visualization scripts (written in Python with Matplotlib and Seaborn) are included in the \"analysis\"\n")
    print(" directory. \n")
    print(" \n")
    print(" Instructions on how to install the dev version of this package are included in the README.md")

    nothing

end

########################################################################################################################

end
