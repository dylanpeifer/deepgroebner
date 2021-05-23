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
    params := {};
    if args#0 == "poly" then (
	params = {"kind" => "poly",
	          "n" => value(args#1),
	          "d" => value(args#2),
	          "s" => value(args#3),
		  "lambda" => value(args#4),
	          "degs" => args#5,
	          "consts" => member("consts", args)};
	)
    else if args#0 == "toric" then (
	params = {"kind" => "toric",
	          "n" => value(args#1),
	          "L" => value(args#2),
		  "U" => value(args#3),
		  "M" => value(args#4)};
	)
    else (
	params = {"kind" => "binom",
	          "n" => value(args#0),
	          "d" => value(args#1),
	          "s" => value(args#2),
	          "degs" => args#3,
	          "consts" => member("consts", args),
	          "homog" => member("homog", args),
	          "pure" => member("pure", args)};
        );
    hashTable params
    )

writePerformanceToFile = method()
writePerformanceToFile(Ideal, String, String) := (I, strategy, fname) -> (
    -- Append a line for strategy performance on I to fname.
    (G, stats) := buchberger(I, SelectionStrategy => strategy, Minimalize => false, Interreduce => false);
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

inFile := "data/stats/" | dist | "/" | dist | ".csv";
if not fileExists inFile then error("No distribution file found. Run scripts/make_dist.m2 first.");

outFile := "data/stats/" | dist | "/" | dist | "_" | strategy | ".csv";
if #scriptCommandLine == 4 and strategy == "random" then (
    outFile = "data/stats/" | dist | "/" | dist | "_" | strategy | "_" | scriptCommandLine#3 | ".csv";
    );
if fileExists outFile then error("Output file " | outFile | " already exists. Delete or move it first.");
F := openOut outFile;
F << "ZeroReductions,NonzeroReductions,PolynomialAdditions,MonomialAdditions" << endl;
close F;

H = parseIdealDist dist;
R = if H#"kind" == "toric" then ZZ/32003[vars(0..(H#"M" - 1))] else ZZ/32003[vars(0..(H#"n" - 1))];
ideals = apply(drop(lines get inFile, 1), s -> ideal value replace("\\|", ",", s));

for I in ideals do (
    writePerformanceToFile(I, capitalize strategy, outFile);
    );
