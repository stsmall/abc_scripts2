# abc_scripts2
Refactoring of abc_scripts.
My goal is to build a customizable set of scripts to allow demographic inference under ABC. Given some of the known affects of linked selection (BGS and Positive) on patterns of polymorphism, I had added the msbgs program. I have also added a model of sequencing error (Jay et al. 2019) and the option to use the MMC in msprime.

**TODO**
* add support for SLiM and pySlim

## NEW in sims: 
1. msprime as default
2. msbgs (for background selection)

## NEW in stats: 
1. 2-locus stats of [momentsLD](https://bitbucket.org/simongravel/moments/src/LD/)
2. Stats implemented in [Jay et al. 2019](https://doi.org/10.1093/molbev/msz038)

### run_sims.py
 * python > 3  
 * sk-allel   
 * numpy  
 * pandas  
 * msprime  
 * [msbgs](https://zeng-lab.group.shef.ac.uk/wordpress/?page_id=28) *(optional)*
 * [discoal](https://github.com/kr-colab/discoal) *(optional)*

### run_stats.py
 * python > 3
 * sk-allel
 * numpy
 * pandas  
 * momentsLD and moments *(optional)*

### recommended workflow
**make sims**  
run_sims.py -cfg examples/model.example.cfg -m examples/model.2test.txt -i 50000 --out msp_test --dryrun  
run_sims.py -cfg examples/model.example.cfg -m examples/model.2test.txt -i 50000 --out msp_test --stats_cfg examples/stats.example.cfg  
python run_sims.py -cfg examples/model.example.cfg -m examples/model.2test.txt -i 5000 --out discoal_test --ms discoal  
python run_sims.py -cfg examples/model.example.cfg -m examples/model.2test.txt -i 5000 --out msbgs_test --ms msbgs  

**get sim stats**  
run_stats.py sims -cfg examples/stats_example.cfg --file discoal.sims.out --format ms  
run_stats.py sims -cfg examples/stats_example.cfg --file msbgs.sims.out --format ms  

**get obs stats**  
run_stats.py obs -cfg examples/stats_example.cfg --file example.vcf --pops example.meta.csv  

**perform analyses**  
[abcrf](https://cran.r-project.org/web/packages/abcrf/index.html) (for importance rankings on parameter inference)       
[abc](https://cran.r-project.org/web/packages/abc/vignettes/abcvignette.pdf) (for parameter inference)   

### notes on multiple merger coalescence (MMC) and deterministic wright-fisher model (dtwf)
(TODO:option to use MMC, beta or dirac)  
[MMC primer](https://pubmed.ncbi.nlm.nih.gov/24750385/)  
[MMC-ABC](https://pubmed.ncbi.nlm.nih.gov/30651284/)  
[MMC Selection](https://pubmed.ncbi.nlm.nih.gov/32396636/)  

