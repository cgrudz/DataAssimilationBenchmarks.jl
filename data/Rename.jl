#######################################################################################################################
module Rename 
########################################################################################################################
# imports and exports
using Debugger
using Glob
using JLD

export rename_files

########################################################################################################################

function rename_files()
    fnames = Glob.glob("./ienks-transform/*")
    for name in fnames
        split_name = split(name, "_")
        tmp = [split_name[1:end-7]; ["nens"]; split_name[end-4:end]]
        rename = ""
        string_len = (length(tmp)-1)
        for i in 1:string_len
            rename *= tmp[i] * "_"
        end
        rename *= tmp[end]
        @bp
        print(rename)
        my_command = `mv $name $rename`
        run(my_command)
    end
 end
 
########################################################################################################################

end
