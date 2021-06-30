# Scripts for Approximate Bayesian Computation in population genomics  
Refactoring of abc_scripts.
My goal is to build a customizable set of scripts to allow demographic inference under ABC. Given some of the known affects of linked selection (BGS and Positive) on patterns of polymorphism, I will implement these options in Slim. I have also added a model of sequencing error (Jay et al. 2019) and the option to use the MMC in msprime.

**TODO**
* add support for SLiM and pySlim. 
* option to use MMC, beta or dirac in msprime 

## Stats: 
1. 2-locus stats of [momentsLD](https://bitbucket.org/simongravel/moments/src/LD/)
2. Stats implemented in [Jay et al. 2019](https://doi.org/10.1093/molbev/msz038)

### run_sims.py
 * python > 3  
 * scikit-allel   
 * numpy  
 * pandas  
 * msprime v1.0.1
 * [discoal](https://github.com/kr-colab/discoal) *(optional)*
 * [scrm](https://github.com/scrm/scrm) *(optional)*

### run_stats.py
 * python > 3
 * sk-allel
 * numpy
 * pandas  
 * [momentsLD](https://bitbucket.org/simongravel/moments/src/LD/moments/) *(optional)*

### examples
**test model**  
`run_sims.py -cfg examples/model.example.cfg -m examples/model.2test.txt -i 5000 --out msp_test --dryrun`  
`run_sims.py -cfg examples/model.example.cfg -m examples/model.2test.txt -i 5000 --out msp_test --ms scrm --dryrun`  

**sims and stats**  
`run_sims.py -cfg examples/model.example.cfg -m examples/model.2test.txt -i 5000 --out msp_test --stats_cfg examples/stats.example.cfg`  
`run_sims.py -cfg examples/model.example.cfg -m examples/model.2test.txt -i 5000 --out scrm_test --ms scrm --stats_cfg examples/stats.example.cfg`  
`run_sims.py -cfg examples/model.example.cfg -m examples/model.2test.txt -i 5000 --out discoal_test --ms discoal --stats_cfg examples/stats.example.cfg`  

**write sims to file, then calc stats from file**  
`run_sims.py -cfg examples/model.example.cfg -m examples/model.2test.txt -i 5000 --out scrm.sims.msout --ms scrm`  
`run_stats.py sims scrm.sims.msout -cfg examples/stats_example.cfg --outfile scrm.sims.out --ms scrm`  

**get obs stats**  
`utils/make_coordsfile.py 3 998670 10000 1000 --gff tests/test.gff --gff_filter intergenic`  
`run_stats.py obs 3 --pops_file examples/obs.40.csv -cfg docs/examples/example.stats.cfg --coords_bed docs/examples/test.bed --zarr_path docs/examples/test --outfile testobsstats`  

**ABC for param inference**  
[abcrf](https://cran.r-project.org/web/packages/abcrf/index.html) (for importance rankings on parameter inference and inference)       
[abc](https://cran.r-project.org/web/packages/abc/vignettes/abcvignette.pdf) (for parameter inference)  

### notes on multiple merger coalescence (MMC) and deterministic wright-fisher model (dtwf)
[MMC primer](https://pubmed.ncbi.nlm.nih.gov/24750385/)  
[MMC-ABC](https://pubmed.ncbi.nlm.nih.gov/30651284/)  
[MMC Selection](https://pubmed.ncbi.nlm.nih.gov/32396636/)  
[Wright-Fisher in msprime](https://www.biorxiv.org/content/10.1101/674440v1)

