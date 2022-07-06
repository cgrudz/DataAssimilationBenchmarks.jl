# Introduction

## Statement of purpose

The purpose of this package is to provide a research framework for the theoretical
development and empirical validation of novel data assimilation techniques.
While analytical proofs can be derived for classical methods such as the Kalman filter
in linear-Gaussian dynamics, most currently developed DA
techniques are designed for estimation in nonlinear, non-Gaussian models where no
analytical solution may exist.  Novel data assimilation methods,
therefore, must be studied with rigorous numerical simulation in standard test-cases
to demonstrate the effectiveness and computational performance.
Pursuant to proposing a novel DA method, one should likewise compare the performance
of a proposed scheme with other standard methods within the same class of estimators.

This package implements a variety of standard data assimilation algorithms, including
widely used performance modifications that are used in practice to tune these estimators.
Standard libraries exist for full-scale DA system research and development, e.g.,
the [Data Assimilation Research Testbed (DART)](https://dart.ucar.edu/), but
there are fewer standard options for theoretical research and algorithm development in
simple test systems. DataAssimilationBenchmarks.jl provides one framework for studying
ensemble-based filters and sequential smoothers that are commonly used in online,
geoscientific prediction settings.

## Validated methods currently in use

For a discussion of the below methods and benchmarks for their validation, please
see the manuscript
[A fast, single-iteration ensemble Kalman smoother for sequential data
assimilation](https://gmd.copernicus.org/preprints/gmd-2021-306/).

```@raw html
<table>
<tr>
	<th>Estimator / implemented techniques</th>
	<th>Tuned multiplicative inflation</th>
	<th>Adaptive inflation, finite-size formalism (perfect model dual / primal)</th>
	<th>Adaptive inflation, finite-size formalism (imperfect model)</th>
	<th>Linesearch</th>
	<th>Localization / Hybridization</th>
	<th>Multiple data assimilation (general shift and lag)</th>
</tr>
<tr>
  <td> ETKF </td>
	<td> X  </td>
	<td> X  </td>
	<td>    </td>
	<td> NA </td>
	<td>    </td>
	<td> NA </td>
</tr>
<tr>
  <td> MLEF, transform / bundle variants</td>
	<td> X  </td>
	<td> X  </td>
	<td>    </td>
	<td> X  </td>
	<td>    </td>
	<td> NA </td>
</tr>
<tr>
  <td> ETKS</td>
	<td> X  </td>
	<td> X  </td>
	<td>    </td>
	<td> NA </td>
	<td>    </td>
	<td> NA </td>
</tr>
<tr>
  <td> MLES, transform / bundle variants</td>
	<td> X  </td>
	<td> X  </td>
	<td>    </td>
	<td> X  </td>
	<td>    </td>
	<td> NA </td>
</tr>
<tr>
  <td>SIEnKS, perturbed obs / ETKF / MLEF variants</td>
	<td> X </td>
	<td> X </td>
	<td>   </td>
	<td> X </td>
	<td>   </td>
	<td> X </td>
</tr>
<tr>
  <td>Gauss-Newton IEnKS, transform / bundle variants</td>
	<td> X </td>
	<td> X </td>
	<td>   </td>
	<td>   </td>
	<td>   </td>
	<td> X </td>
</tr>
</table>
```

