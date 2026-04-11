from SigProfilerAssignment import Analyzer as Analyze

project = "../results/LUAD_sig_output"
input_type = "matrix"
input_data = "../data/maf/output/SBS/TCGA_LUAD.SBS96.all"

Analyze.cosmic_fit(
    samples=input_data,
    output=project,
    input_type=input_type,
    context_type="96",
    collapse_to_SBS96=True,
    cosmic_version=3.4
)