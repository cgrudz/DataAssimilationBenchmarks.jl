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
    fnames = Glob.glob("./mlef-ls-transform/*")
    for name in fnames
        split_name = split(name, "_")
        @bp
        tmp = [split_name[1:18]; [lpad(parse(Float64, split_name[19]), 5, "0")]; split_name[20:end]]
        rename = ""
        string_len = (length(tmp)-1)
        for i in 1:string_len
            rename *= tmp[i] * "_"
        end
        rename *= tmp[end]
        @bp
        #print(rename)
        my_command = `mv $name $rename`
        run(my_command)
    end
 end
 
########################################################################################################################

end
