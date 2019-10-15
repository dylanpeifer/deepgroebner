-- bins.m2
-- Dylan Peifer
-- 11 Oct 2019

-- Generate binned sets of random ideals.

needsPackage "Ideals"

n = value(scriptCommandLine#1);
R = ZZ/32003[vars(0..n-1)];
d = value(scriptCommandLine#2);
s = value(scriptCommandLine#3);

bin = method()
bin ZZ := n -> floor(log(n * 1.0) / log(1.1))

binomialToString = method()
binomialToString RingElement := f -> (
    tms := terms f;
    es := tms/exponents/first//flatten/toString;
    c := if #tms == 1 then 0 else lift(leadCoefficient tms_1, ZZ);
    str := (" "|c|" "|concatenate between(" ", es));
    str)

writeToFile = method()
writeToFile Ideal := I -> (
    J := flatten entries gens gb I;
    ngb := #J;
    b := bin ngb;
    F := openOutAppend("bin"|b);
    F << numgens I << " " << ngb << endl;
    for f in I_* do F << binomialToString f << endl;
    close F
    )

-- generates one million examples
for j from 1 to 1000000 do writeToFile randomBinomialIdeal(R, d, s);
