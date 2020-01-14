-* Generate binned files of random ideals.

Call this script as

    M2 --script make_bins.m2 n d s consts degs homog pure samples

where n, d, s, and samples are integers, consts, homog, and pure are 0 or 1
representing false or true, and degs is one of {uniform, weighted, maximum}.

Output is stored in files data/bins/<parameters>/bin<bin>.txt. If files already
exist they will be appended to rather than replaced. These files are in the
format that FromFileIdealGenerator reads.

Ideals are binned by the log size of their reduced Groebner basis, which is a
rough measure of how difficult the Groebner basis is to compute.
*-

needsPackage "Ideals"

bin = method()
bin ZZ := n -> floor(log(n * 1.0) / log(1.1))

binomialToString = method()
binomialToString RingElement := f -> (
    tms := terms f;
    es := tms/exponents/first//flatten/toString;
    c := if #tms == 1 then 0 else lift(leadCoefficient tms_1, ZZ);
    str := ("  "|c|" "|concatenate between(" ", es));
    str
    )

capitalize = method()
capitalize String := s -> toUpper s#0 | substring(1, #s, s)

setupDirectory = method(Options => {
	Constants => false,
	Degrees => "Uniform",
	Homogeneous => false,
	Pure => false})
setupDirectory(ZZ, ZZ, ZZ) := String => opts -> (n, d, s) -> (
    dir = "../data/bins/";
    if not isDirectory dir then makeDirectory dir;
    dir = dir | concatenate between("-", {n, d, s}/toString);
    if opts.Constants then dir = dir | "-consts";
    dir = dir | "-" | toLower opts.Degrees;
    if opts.Homogeneous then dir = dir | "-homog";
    if opts.Pure then dir = dir | "-pure";
    if not isDirectory dir then makeDirectory dir;
    dir
    )

writeToFile = method()
writeToFile(Ideal, String) := (I, dir) -> (
    J := flatten entries gens gb I;
    ngb := #J;
    b := bin ngb;
    F := openOutAppend(dir|"/bin"|b|".txt");
    F << numgens I << " " << ngb << endl;
    for f in I_* do F << binomialToString f << endl;
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
dir = setupDirectory(n, d, s, opts);

for i from 1 to samples do (
    I = randomBinomialIdeal(R, d, s, opts);
    writeToFile(I, dir);
    );
