# Contributing

## What to expect

This is currenlty a small project maintained by a small community of researchers and
developers, focused on academic research in data assimilation and machine learning
methodology. This software is not intended for use in an operational data assimilation
environment, or even for use with real data, as there are a variety of existing alternatives
for that scope.

We will gladly work with others who want to join our community by using this software for
their own research, whether that means simply using an existing functionality or contributing
new source code to the main code base.  However, please note that support for community
development is primarily a volunteer effort, and it may take some time to get a response.

Anyone participating in this community is expected to follow the 
[Contributor Covenant Code of Conduct](@ref).

## How to get support or ask a question

The preferred means to get support is to use the
[Github discussions](https://github.com/cgrudz/DataAssimilationBenchmarks.jl/discussions). 
to ask a question or to make an introduction as a user of this software.
If you cannot find what you are looking for in the
[main documentation](https://cgrudz.github.io/DataAssimilationBenchmarks.jl/dev/),
please feel free to post a question in the 
[Github discussions](https://github.com/cgrudz/DataAssimilationBenchmarks.jl/discussions). 
and this may be migrated to the main documentation as frequently asked questions develop.

## Bug reports with existing code

Bug reports go in the 
[Github issues](https://github.com/cgrudz/DataAssimilationBenchmarks.jl/issues)
for the project.  Please follow the template and provide any relevant details of the
bug you have encountered that will allow the community to reproduce and solve the issue.
Reproducibility is key, and if there are not sufficient details to reproduce an issue,
the issue will be sent back for more details.

## How to contribute new methods, models or other core core

The best way to contribute new code is to reach out to the community first, as this code
base is in an early and active state of development and will occassionally face breaking
changes in order to accomodate more generality and new features.  Please start with an
introduction of yourself in the 
[Github discussions](https://github.com/cgrudz/DataAssimilationBenchmarks.jl/discussions)
followed by a detailed feature request in the
[Github issues](https://github.com/cgrudz/DataAssimilationBenchmarks.jl/issues),
covering your use-case and what new functionality you are proposing. This will help
the community anticipate your needs and the backend changes that might need to be implemented
in order to accomodate new functionality. There is not currently a general system for
how new data assmiliation methods or models are to be implemented, and it is therefore
critical to bring up your use-case to the community so that how this new feature is
incorporated can be planned into the development.  Once the issue can be evaluated and
discussed by the development community, the strategy is usually to create a fork of the
main code base where new modules and features can be prototyped.  Once the new code
development is ready for a review, a pull request can be made where the new functionality
may be merged, and possibly further refactored for general consistency and consolidation
of codes.

Ideally, any new data assimilation method incorporated into this code should come with
a hyper-parameter configuration built into the [SingleExperimentDriver](@ref) module,
a selected benchmark model in which the learning scheme is to be utilized and a
corresponding test case that demonstrates and verifies an expected behavior.  As much
as possible, conventions with arguments should try to match
existing conventions in, e.g., [EnsembleKalmanSchemes](@ref) and [XdVAR](@ref), though
it is understood that not all data assimilation methods need follow these conventions
or even have analogous arguments and sub-routines. Please discuss your approach with
the community in advanced so that the framework can be made as consistent (and
therefore extendable and user-friendly) as possible.
