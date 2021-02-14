-* Generate CSV file of ideals sampled from distribution.

Call this script as

    M2 --script scripts/make_dist.m2 <distribution> <samples> <seed>

Output is stored in file

    data/stats/<distribution>/<distribution>.csv

If this file already exists an error will be raised.
*-

needsPackage("Ideals", FileName => "m2/Ideals.m2")

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

setupOutFile = method()
setupOutFile String := String => dist -> (
    -- Setup output file and return its name.
    directory := "data/stats/" | dist | "/";
    if not isDirectory directory then makeDirectory directory;
    outFile := directory | dist | ".csv";
    if fileExists outFile then
        error("Output file " | outFile | " already exists. Delete or move it first.");
    F = openOut outFile;
    F << "Ideal" << endl;
    close F;
    outFile
    )

writeIdealToFile = method()
writeIdealToFile(Ideal, String) := (I, fname) -> (
    -- Append a line for ideal I to fname.
    s := toString first entries gens I;
    s = replace("{|}", "", s);
    s = replace(", ", "|", s);
    F := openOutAppend fname;
    F << s << endl;
    close F
    )
    
dist = scriptCommandLine#1;
samples = value(scriptCommandLine#2);
if #scriptCommandLine == 4 then setRandomSeed(value(scriptCommandLine#3));

outFile = setupOutFile dist;
H = parseIdealDist dist;

if H#"kind" == "binom" then (
    R = ZZ/32003[vars(0..(H#"n" - 1))];
    opts = new OptionTable from {Constants => H#"consts",
                                 Degrees => capitalize H#"degs",
                                 Homogeneous => H#"homog",
                                 Pure => H#"pure"};
    for sample from 1 to samples do (
        I = randomBinomialIdeal(R, H#"d", H#"s", opts);
        writeIdealToFile(I, outFile);
        );
) else if H#"kind" == "poly" then (
    R = ZZ/32003[vars(0..(H#"n" - 1))];
    opts = new OptionTable from {Constants => H#"consts",
	                         Degrees => capitalize H#"degs"};
    for sample from 1 to samples do (
        I = randomPolynomialIdeal(R, H#"d", H#"s", H#"lambda", opts);
        writeIdealToFile(I, outFile);
        );
) else (
    for sample from 1 to samples do (
        I = randomToricIdeal(H#"n", H#"L", H#"U", H#"M");
        writeIdealToFile(I, outFile);
        );
);
    
    
