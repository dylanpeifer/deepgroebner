-* Generate CSV file of returns for new strategies.

This is just a quick copy of make_stats.m2 with the listed strategies changed.
The two files should be combined eventually.

Call this script as

    M2 --script make_stats2.m2 n d s consts degs homog pure samples

where n, d, s, and samples are integers, consts, homog, and pure are 0 or 1
representing false or true, and degs is one of {uniform, weighted, maximum}.

Output is stored in file data/stats/<params>.csv. If the file already exists
it will be appended to rather than replaced.
*-

needsPackage "Ideals"
needsPackage "SelectionStrategies"

capitalize = method()
capitalize String := s -> toUpper s#0 | substring(1, #s, s)

setupFile = method(Options => {
	Constants => false,
	Degrees => "Uniform",
	Homogeneous => false,
	Pure => false})
setupFile(ZZ, ZZ, ZZ) := String => opts -> (n, d, s) -> (
    fname = "../data/stats/";
    if not isDirectory fname then makeDirectory fname;
    fname = fname | "new-" | concatenate between("-", {n, d, s}/toString);
    if opts.Constants then fname = fname | "-consts";
    fname = fname | "-" | toLower opts.Degrees;
    if opts.Homogeneous then fname = fname | "-homog";
    if opts.Pure then fname = fname | "-pure";
    fname = fname | ".csv";
    if not fileExists fname then (
	F := openOutAppend fname;
	for sel in {"Degree", "TrueDegree", "MonomialDegree", "MonomialTrueDegree", "MonomialTrueDegreeDegree"} do
	    for i from 1 to 4 do
		F << sel << ",";
	F << "Difficulty,Difficulty" << endl;
	for sel in {"Degree", "TrueDegree", "MonomialDegree", "MonomialTrueDegree", "MonomialTrueDegreeDegree"} do
	    F << "zeroReductions,nonzeroReductions,polynomialAdditions,monomialAdditions,";
	F << "time (seconds),generators" << endl;
	close F;
	);
    fname
    )

writeToFile = method()
writeToFile(Ideal, String) := (I, fname) -> (
    stats := {};
    for sel in {"Degree", "TrueDegree", "MonomialDegree", "MonomialTrueDegree", "MonomialTrueDegreeDegree"} do (
	(G, s) := buchberger(I,
                             SelectionStrategy => sel,
                             Minimalize => false,
                             Interreduce => false);
	stats = stats | {s#"zeroReductions",
	                 s#"nonzeroReductions",
	                 s#"polynomialAdditions",
	                 s#"monomialAdditions"}
	);
    (t, G) := toSequence timing gb I;
    stats = stats | {t, length flatten entries gens G};
    F := openOutAppend fname;
    F << concatenate between(",", stats/toString) << endl;
    close F
    )

n = value(scriptCommandLine#1);
d = value(scriptCommandLine#2);
s = value(scriptCommandLine#3);
consts = (value(scriptCommandLine#4) == 1);
degs = capitalize scriptCommandLine#5;
homog = (value(scriptCommandLine#6) == 1);
pure = (value(scriptCommandLine#7) == 1);
samples = value(scriptCommandLine#8);

R = ZZ/32003[vars(52..52+n-1)];
opts = new OptionTable from {Constants => consts,
                             Degrees => degs,
                             Homogeneous => homog,
                             Pure => pure};
fname = setupFile(n, d, s, opts);

for i from 1 to samples do (
    I = randomBinomialIdeal(R, d, s, opts);
    writeToFile(I, fname);
    );
