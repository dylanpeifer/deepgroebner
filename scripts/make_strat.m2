-* Generate CSV file of strategy performance for sampled ideals.

Call this script as

    M2 --script scripts/make_strat.m2 <distribution> <strategy> <seed>

Input is taken from

    data/stats/<distribution>/<distribution>.csv

and output is stored in the file

    data/stats/<distribution>/<distribution>_<strategy>.csv

If the input file does not exist or the output file already exists an error
will be raised.
*-

needsPackage("SelectionStrategies", FileName => "m2/SelectionStrategies.m2")

capitalize = method()
capitalize String := s -> toUpper s#0 | substring(1, #s, s)

parseIdealDist = method()
parseIdealDist String := HashTable => dist -> (
    -- Return HashTable with parameters for ideal distribution.
    args := separate("-", dist);
    if member(args#3, {"uniform", "weighted", "maximum"}) then (
	L := {"n" => value(args#0),
	      "d" => value(args#1),
	      "s" => value(args#2),
	      "degs" => args#3,
	      "consts" => member("consts", args),
	      "homog" => member("homog", args),
	      "pure" => member("pure", args)};
    ) else (
    	error("must be a binomial ideal distribution");
	);
    hashTable L
    )

setupInFile = method()
setupInFile String := String => dist -> (
    -- Setup input file and return its name.
    inFile := "data/stats/" | dist | "/" | dist | ".csv";
    if not fileExists inFile then
        error("No distribution file found. Run scripts/make_dist.m2 first.");
    inFile
    )

setupOutFile = method()
setupOutFile(String, String) := String => (dist, strategy) -> (
    -- Setup output file and return its name.
    outFile := "data/stats/" | dist | "/" | dist | "_" | strategy | ".csv";
    if fileExists outFile then
        error("Output file " | outFile | " already exists. Delete or move it first.");
    F := openOut outFile;
    F << "ZeroReductions,NonzeroReductions,PolynomialAdditions,MonomialAdditions" << endl;
    close F;
    outFile
    )

writePerformanceToFile = method()
writePerformanceToFile(Ideal, String, String) := (I, strategy, fname) -> (
    -- Append a line for strategy performance on I to fname.
    (G, stats) := buchberger(I, SelectionStrategy => strategy);
    F := openOutAppend fname;
    F << stats#"zeroReductions" << ","
      << stats#"nonzeroReductions" << ","
      << stats#"polynomialAdditions" << ","
      << stats#"monomialAdditions" << endl;
    close F
    )

dist = scriptCommandLine#1;
strategy = scriptCommandLine#2;
if #scriptCommandLine == 4 then setRandomSeed(value(scriptCommandLine#3));

inFile = setupInFile dist;
outFile = setupOutFile(dist, strategy);
H = parseIdealDist dist;
R = ZZ/32003[vars(0..(H#"n" - 1))];
ideals = apply(drop(lines get inFile, 1), s -> ideal value replace("\\|", ",", s));

for I in ideals do (
    writePerformanceToFile(I, capitalize strategy, outFile);
    );
