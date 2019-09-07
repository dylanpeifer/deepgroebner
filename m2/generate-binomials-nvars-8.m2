-- Generate random ideals over ZZ/32003 of 2*n binomials each of (inhom) degree <= n
setRandomSeed "hi there"
R = ZZ/32003[vars(0..7)]
B = flatten entries basis(0,5, R);
randBinomial = (R,B) -> (
    kk := coefficientRing R;
    first flatten entries gens gb ideal (B#(random(#B)) + (random kk) * B#(random(#B)))
    )
randIdeal1 = (R,B,n) -> ideal for i to n-1 list randBinomial(R,B)
randIdeal = (n) -> randIdeal1(R,B,n)

bin = (nelems) -> floor(log(nelems * 1.0) / log(1.1))

binomialToString = method()
binomialToString RingElement := f -> (
    tms := terms f;
    es := tms/exponents/first//flatten/toString;
    c := if #tms == 1 then 0 else lift(leadCoefficient tms_1, ZZ);
    str := (" "|c|" "|concatenate between(" ", es));
    str)

writeToFile = method()
writeToFile Ideal := (I) -> (
    J := flatten entries gens gb I;
    ngb := #J;
    b := bin ngb;
    F := openOutAppend("bin"|b);
    F << numgens I << " " << ngb << endl;
    for f in I_* do F << binomialToString f << endl;
    close F
    )

end--
restart
load "generate-binomials-nvars-8.m2"

elapsedTime for i from 0 to 1000 list (I := randIdeal 10; # flatten entries gens gb I)

for i from 0 to 100 do writeToFile randIdeal 10

-- Generates one million examples
for i from 0 to 999 do (
    for j from 0 to 999 do writeToFile randIdeal 10;
    << "done set " << i << endl;
    )
    
