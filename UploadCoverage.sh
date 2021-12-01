#!/bin/bash
CODECOV_TOKEN=fad961cc-8e22-49d6-8f6f-06663265bd57 julia -e 'using Pkg; using Coverage; Codecov.submit_local(process_folder())'

