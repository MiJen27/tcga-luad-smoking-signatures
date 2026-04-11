from SigProfilerMatrixGenerator.scripts import SigProfilerMatrixGeneratorFunc

##project = "TCGA_LUAD"
##path_to_maf = "../data/maf/"

matrices = SigProfilerMatrixGeneratorFunc.SigProfilerMatrixGeneratorFunc(
    project="LUAD",
    reference_genome="GRCh38",
    path_to_input_files="../data/maf/",
    plot=False,
)



