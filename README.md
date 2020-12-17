# abc_scripts2
Refactoring of abc_scripts.
My goal is to build a customizable set of scripts to allow demographic inference under ABC. Given some of the known affects of linked selection (BGS and Positive) on patterns of polymorphism, I had added the msbgs program. I have also added a model of sequencing error (Jay et al. 2019) and the option to use the MMC in msprime.

## NEW in sims: 
1. msprime as default
2. msbgs (for background selection)

## NEW in stats: 
1. 2-locus stats of [momentsLD](https://bitbucket.org/simongravel/moments/src/LD/)
2. Stats implemented in [Jay et al. 2019](https://doi.org/10.1093/molbev/msz038)

### run_sims.py 
 *required*
 * python > 3
 * sk-allel
 * numpy
 * pandas  
 *optional*
 * msprime + tskit
 * [msbgs](https://zeng-lab.group.shef.ac.uk/wordpress/?page_id=28)
 * [discoal](https://github.com/kr-colab/discoal)

### run_stats.py
 *required*
 * python > 3
 * sk-allel
 * numpy
 * pandas
 *optional*  
 * [twoPopnStats_forML](https://github.com/kr-colab/FILET)
 * momentsLD

### recommended workflow
run_sims.py --ms msprime (consider using MMC if training set for selection)
run_sims.py --ms msbgs (adds sites with background selection)
run_stats.py  
[abcrf](https://cran.r-project.org/web/packages/abcrf/index.html) (for importance rankings on parameter inference)  
[abc](https://cran.r-project.org/web/packages/abc/vignettes/abcvignette.pdf)  

[MMC primer](https://pubmed.ncbi.nlm.nih.gov/24750385/)  
[MMC-ABC](https://pubmed.ncbi.nlm.nih.gov/30651284/)  
[MMC Selection](https://pubmed.ncbi.nlm.nih.gov/32396636/)  

