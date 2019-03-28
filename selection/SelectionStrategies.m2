newPackage(
        "SelectionStrategies",
        Version => "0.1", 
        Date => "March 24, 2019",
        Authors => {{Name => "Dylan Peifer", 
                     Email => "djp282@cornell.edu", 
                     HomePage => "https://www.math.cornell.edu/~djp282"}},
        Headline => "Test pair selection strategies in Buchberger's algorithm",
        DebuggingMode => true
        )

export {"SugarPolynomial", "sugarPolynomial", "polynomial", "sugar",
    "reduce", "minimalize", "interreduce",
    "spoly",
    "buchberger", "SelectionStrategy", "EliminationStrategy", "ReductionStrategy", "Homogenize", "Minimalize"}


-------------------------------------------------------------------------------
--- utility functions
-------------------------------------------------------------------------------
argmax = method()
argmax(List) := ZZ => (x) -> (
    -- x = a list
    -- returns the index of the max element of x (or the first index if there are multiple)
    if #x === 0 then null else first fold((i,j) -> if i#1 >= j#1 then i else j, pairs x)
    )
argmax(List, Function) := ZZ => (x, f) -> (
    argmax(apply(x, f))
    )

argmin = method()
argmin(List) := ZZ => (x) -> (
    -- x = a list
    -- returns the index of the min element of x (or the first index if there are multiple)
    if #x === 0 then null else first fold((i,j) -> if i#1 <= j#1 then i else j, pairs x)
    )
argmin(List, Function) := ZZ => (x, f) -> (
    argmin(apply(x, f))
    )

-------------------------------------------------------------------------------
--- sugar polynomials
-------------------------------------------------------------------------------
SugarPolynomial = new Type of BasicList

sugarPolynomial = method(TypicalValue => SugarPolynomial)
sugarPolynomial(ZZ, SugarPolynomial) :=
sugarPolynomial(ZZ, RingElement)     :=
sugarPolynomial(ZZ, Number)          := (z, f) -> new SugarPolynomial from {z, polynomial f}
sugarPolynomial SugarPolynomial      :=
sugarPolynomial RingElement          :=
sugarPolynomial Number               := f -> sugarPolynomial(sugar f, f)

sugar = method(TypicalValue => ZZ)
sugar SugarPolynomial := f -> f#0
sugar RingElement     := f -> if #(degree f) == 0 then 0 else first degree f
sugar Number          := f -> 0

polynomial = method(TypicalValue => RingElement)
polynomial SugarPolynomial := f -> f#1
polynomial RingElement     := f -> f
polynomial Number          := f -> f

leadTerm SugarPolynomial        := f -> leadTerm polynomial f
leadMonomial SugarPolynomial    := f -> leadMonomial polynomial f
leadCoefficient SugarPolynomial := f -> leadCoefficient polynomial f

SugarPolynomial + SugarPolynomial := SugarPolynomial => (f, g) -> (
    sugarPolynomial(max(sugar f, sugar g), polynomial f + polynomial g)
    )
SugarPolynomial + RingElement :=
SugarPolynomial + Number      :=
RingElement + SugarPolynomial :=
Number + SugarPolynomial      := SugarPolynomial => (f, g) -> (
    sugarPolynomial f + sugarPolynomial g
    )

SugarPolynomial - SugarPolynomial := SugarPolynomial => (f, g) -> (
    sugarPolynomial(max(sugar f, sugar g), polynomial f - polynomial g)
    )
SugarPolynomial - RingElement :=
SugarPolynomial - Number      :=
RingElement - SugarPolynomial :=
Number - SugarPolynomial      := SugarPolynomial => (f, g) -> (
    sugarPolynomial f - sugarPolynomial g
    )

SugarPolynomial * SugarPolynomial := SugarPolynomial => (f, g) -> (
    sugarPolynomial(sugar f + sugar g, polynomial f * polynomial g)
    )
SugarPolynomial * RingElement :=
SugarPolynomial * Number      :=
RingElement * SugarPolynomial :=
Number * SugarPolynomial      := SugarPolynomial => (f, g) -> (
    sugarPolynomial f * sugarPolynomial g
    )

SugarPolynomial == SugarPolynomial := Boolean => (f, g) -> (
    polynomial f == polynomial g
    )
SugarPolynomial == RingElement :=
SugarPolynomial == Number      :=
RingElement == SugarPolynomial :=
Number == SugarPolynomial      := SugarPolynomial => (f, g) -> (
    sugarPolynomial f == sugarPolynomial g
    )

SugarPolynomial ? SugarPolynomial := (f, g) -> polynomial f ? polynomial g

-------------------------------------------------------------------------------
--- reduction
-------------------------------------------------------------------------------
reduce = method(Options => {Strategy => "Regular", Reduce => "Full"})
reduce(RingElement, List) := RingElement =>
reduce(SugarPolynomial, List) := SugarPolynomial => opts -> (g, F) -> (
    -- g = a polynomial
    -- F = a list of polynomials
    -- returns a remainder when g is divided by F

    r := 0; -- stores the computed remainder while g stores the tail

    -- if only tail reducing then put lead term on remainder here
    if opts.Reduce === "Tail" then (
	r = r + leadTerm g;
	g = g - leadTerm g;
	);
    
    doubleSugar := opts.Strategy === "DoubleSugar" or opts.Strategy === "Saccharine";

    while g != 0 do (
	lg := leadTerm g;
	foundDivisor := false;

	-- try to remove lead term of g by some f, don't increase sugar if doubleSugar
	for f in F do (
	    lf := leadTerm f;
	    if lg % lf == 0 then (
		reducer := (lg//lf) * f;
		if doubleSugar and sugar reducer > sugar g then (
		    continue;
		    )
		else (
		    g = g - reducer;
		    foundDivisor = true;
		    break;
		    );
	        );
	    );

    	-- if first pass didn't find a divisor then we
	--    try again without doubleSugar if using Saccharine
	--    are done if only head reducing
	--    otherwise remove lead term of g and add it to the remainder
	if not foundDivisor then (
	    if opts.Strategy === "Saccharine" and doubleSugar then (
		doubleSugar = false;
		continue;
		)
	    else if opts.Reduce === "Head" then (
		break;
		)
	    else (
		r = r + lg;
		g = g - lg;
	        );
	    );

    	-- make sure doubleSugar is back on if Saccharine might have turned it off
	if opts.Strategy === "Saccharine" then doubleSugar = true;
	);

    -- r holds remainder and g holds sugar degree and/or unreduced tail terms
    r + g
    )

minimalize = method()
minimalize(List) := List => (F) -> (
    -- F = a list of polynomials forming a Groebner basis
    -- returns a minimal Groebner basis from F

    G := {};
    for f in sort F do (
	if all(G, g -> leadTerm f % leadTerm g != 0) then (
	    G = append(G, f);
	    );
	);
    G
    )

interreduce = method()
interreduce(List) := List => (F) -> (
    -- F = a list of polynomials forming a minimal Groebner basis
    -- returns the reduced Groebner basis from F

    G := {};
    for f in F do (
	g := reduce(f, F, Reduce => "Tail");
	G = append(G, 1/(leadCoefficient g) * g);
	);
    G
    )

-------------------------------------------------------------------------------
--- s-pairs and s-polynomials
-------------------------------------------------------------------------------
SPair = new Type of BasicList

spair = method(TypicalValue => SPair)
spair(Sequence, List) := (S, F) -> (
    f := F#(S#0);
    g := F#(S#1);
    gamma := lcm(leadMonomial f, leadMonomial g);
    sug := max(sugar f + sugar (gamma // leadMonomial f),
	       sugar g + sugar (gamma // leadMonomial g));
    new SPair from {S, gamma, sug}
    )

indices SPair := Sequence    => p -> p#0
lcm     SPair := RingElement => p -> p#1
sugar   SPair := ZZ          => p -> p#2
degree  SPair := ZZ          => p -> first degree lcm p

spoly = method()
spoly(RingElement, RingElement) := RingElement =>
spoly(SugarPolynomial, SugarPolynomial) := SugarPolynomial => (f, g) -> (
    -- f = a polynomial
    -- g = a polynomial
    -- returns the s-polynomial of f and g

    gamma := lcm(leadMonomial f, leadMonomial g);
    (gamma // leadTerm f) * f - (gamma // leadTerm g) * g
    )
spoly(SPair, List) := (p, F) -> (
    -- p = an SPair
    -- F = a list of polynomials
    -- returns the s-polynomial given by p

    (i, j) := indices p;
    spoly(F#i, F#j)
    )

lcmCriterion = method()
lcmCriterion(SPair, List) := Boolean => (p, F) -> (
    -- p = an SPair
    -- F = the corresponding list of polynomials
    -- returns true if the pair satisfies the lcm criterion
    
    (i, j) := indices p;
    lcm p == leadMonomial F#i * leadMonomial F#j
    )

-------------------------------------------------------------------------------
--- s-pair selection and updates
-------------------------------------------------------------------------------
PairList := new Type of MutableList

selectPair = method(Options => {Strategy => "First", Sort => false})
selectPair(List) := SPair => opts -> (P) -> (
    -- P = a list of SPairs in Buchberger's algorithm
    -- returns the next pair to process

    p := 0;

    if opts.Strategy === "First" then (
	p = P#0
	)
    else if opts.Strategy === "Random" then (
	p = P#(random(#P))
	)
    else if opts.Strategy === "Degree" then (
	p = P#(argmin(P, degree))
	)
    else if opts.Strategy === "Normal" then (
	p = P#(argmin(P, lcm))
	)
    else if opts.Strategy === "Sugar" then (
	p = P#(argmin(P, p -> {sugar p, lcm p}))
	);

    P = delete(p, P);
    (p, P)
    )

updatePairs = method(Options => {Strategy => "None"})
updatePairs(List, List, RingElement) := List =>
updatePairs(List, List, SugarPolynomial) := List => opts -> (P, F, f) -> (
    -- P = a list of SPairs
    -- F = the corresponding list of polynomials
    -- f = a new polynomial
    -- returns a sequence (P, F) containing the the new list of pairs P and new
    --     polynomials F obtained after adding f

    F = append(F, f);
    P' := apply(#F-1, i -> spair((i, #F-1), F));

    if opts.Strategy === "LCM" then (
	P' = select(P', p -> not lcmCriterion(p, F));
	)
    else if opts.Strategy === "Sugar" or opts.Strategy === "GebauerMoeller" then (
	-- eliminate from old list
	lf := leadMonomial f;
	P = select(P, p -> (
		(i, j) := indices p;
	    	lcm p % lf != 0 or
		lcm(leadMonomial F#i, lf) == lcm p or
	    	lcm(leadMonomial F#j, lf) == lcm p));

    	-- sugar paper eliminates LCM early
	if opts.Strategy === "Sugar" then (
	    P' = select(P', p -> not lcmCriterion(p, F));
	);

    	-- eliminate if strictly divisible
    	P' = select(P', p -> all(P', p' -> lcm p % lcm p' != 0 or lcm p == lcm p'));

    	-- keep 1 of each equivalence class and remove any with lcm criterion
    	classes := partition(lcm, P');
	P'' := {};
	for m in keys classes do (
	    if any(classes#m, p -> lcmCriterion(p, F)) then continue;
	    P'' = append(P'', classes#m#0);
	    );
	P' = P'';
	);

    (P | P', F)
    )

-------------------------------------------------------------------------------
--- main algorithm
-------------------------------------------------------------------------------
BuchbergerHistory = new Type of HashTable;

buchberger = method(Options => {
	SelectionStrategy => "Sugar",
	EliminationStrategy => "GebauerMoeller",
	ReductionStrategy => "Regular",
	Homogenize => false,
	Minimalize => true
	})
buchberger(Ideal) := BuchbergerHistory => opts -> (I) -> (
    -- I = an ideal in a polynomial ring
    -- returns number of pairs processed in computing a Groebner basis of I

    F := first entries gens I;
    if opts.SelectionStrategy === "Sugar" then F = apply(F, sugarPolynomial);

    -- initialize pairs P and polynomials G
    P := {};
    G := {};
    for f in F do (
	(P, G) = updatePairs(P, G, f, Strategy => opts.EliminationStrategy);
	);

    zeroReductions := 0;
    nonzeroReductions := 0;
    p := 0;

    while #P > 0 do (
	(p, P) = selectPair(P, Strategy => opts.SelectionStrategy);
	s := spoly(p, G);
	r := reduce(s, G, Strategy => opts.ReductionStrategy);
	if r != 0 then (
	    (P, G) = updatePairs(P, G, r, Strategy => opts.EliminationStrategy);
	    nonzeroReductions = nonzeroReductions + 1;
	    )
	else (
	    zeroReductions = zeroReductions + 1;
	    );
	);

    if opts.SelectionStrategy === "Sugar" then G = apply(G, polynomial);
    if opts.Minimalize then G = interreduce(minimalize(G));

    (zeroReductions, nonzeroReductions, G)
    )

-------------------------------------------------------------------------------
--- example ideals
-------------------------------------------------------------------------------
cyclic = method(Options => {CoefficientRing => ZZ/32003, MonomialOrder => GRevLex})
cyclic ZZ := Ideal => opts -> n -> (
    R := (opts.CoefficientRing)[vars(0..n-1), MonomialOrder => opts.MonomialOrder];
    F := toList apply(1..n-1, d -> sum(0..n-1, i -> product(d, k -> R_((i+k)%n))))
         | {product gens R - 1};
    ideal F
    )

hcyclic = method(Options => {CoefficientRing => ZZ/32003, MonomialOrder => GRevLex})
hcyclic ZZ := Ideal => opts -> n -> (
    R := (opts.CoefficientRing)[vars(0..n), MonomialOrder => opts.MonomialOrder];
    F := toList apply(1..n-1, d -> sum(0..n-1, i -> product(d, k -> R_((i+k)%n))))
         | {product(n, i -> R_i) - R_n^n};
    ideal F
    )

ecyclic = method(Options => {CoefficientRing => ZZ/32003, MonomialOrder => GRevLex})
ecyclic ZZ := Ideal => opts -> n -> (
    R := (opts.CoefficientRing)[vars(0..n), MonomialOrder => opts.MonomialOrder];
    F := toList apply(1..n-1, d -> R_0^d + sum(0..n-1, i -> product(d, k -> R_((i+k)%n+1))))
         | {product(1..n, i -> R_i) - 1, R_0^n + 1};
    ideal F
    )

rcyclic = method(Options => {CoefficientRing => ZZ/32003, MonomialOrder => GRevLex})
rcyclic ZZ := Ideal => opts -> n -> (
    R := (opts.CoefficientRing)[vars(0..n), MonomialOrder => opts.MonomialOrder];
    F := toList apply(1..n-1, d -> sum(0..n-1, i -> product(d, k -> R_((i+k)%n))));
    F = apply(F, f -> sub(f, R_(n-1) => 1));
    F = append(F, sub(product(n, i -> R_i), R_(n-1) => R_(n-1)^n) - 1);
    ideal F
    )

elemsym = method(Options => {CoefficientRing => ZZ/32003, MonomialOrder => GRevLex})
elemsym ZZ := Ideal => opts -> n -> (
    R := (opts.CoefficientRing)[vars(0..n-1), MonomialOrder => opts.MonomialOrder];
    F := apply(1..n, d -> sum(apply(subsets(gens R, d), product)));
    ideal F
    )

eco = method(Options => {CoefficientRing => ZZ/32003, MonomialOrder => GRevLex})
eco ZZ := Ideal => opts -> n -> (
    R := (opts.CoefficientRing)[vars(0..n-1), MonomialOrder => opts.MonomialOrder];
    F := toList apply(0..n-2, k -> R_(n-1) * (R_k + sum(0..n-k-3, i -> R_i * R_(i+k+1))) - k - 1)
         | {sum(0..n-2, i -> R_i) + 1};
    ideal F
    )

noon = method(Options => {CoefficientRing => ZZ/32003, MonomialOrder => GRevLex})
noon ZZ := Ideal => opts -> n -> (
    R := (opts.CoefficientRing)[vars(0..n-1), MonomialOrder => opts.MonomialOrder];
    F := apply(0..n-1, i -> 10 * R_i * (sum(0..n-1, j -> R_j^2) - R_i^2) - 11 * R_i + 10);
    ideal F
    )

reimer = method(Options => {CoefficientRing => ZZ/32003, MonomialOrder => GRevLex})
reimer ZZ := Ideal => opts -> n -> (
    R := (opts.CoefficientRing)[vars(0..n-1), MonomialOrder => opts.MonomialOrder];
    F := apply(2..n+1, d -> sum(0..n-1, i -> (-1)^i * 2 * R_i^d) - 1);
    ideal F
    )

katsura = method(Options => {CoefficientRing => ZZ/32003, MonomialOrder => GRevLex})
katsura ZZ := Ideal => opts -> n -> (
    n = n - 1;
    R := (opts.CoefficientRing)[vars(0..n), MonomialOrder => opts.MonomialOrder];
    u := i -> (
	 if i < 0 then i = -i;
	 if i <= n then R_i else 0_R
	 );
    f1 := -1 + sum for i from -n to n list u i;
    F := toList prepend(f1, apply(0..n-1, i -> - u i + sum(-n..n, j -> (u j) * (u (i-j)))));
    ideal F
    )

jason210 = method(Options => {CoefficientRing => ZZ/32003, MonomialOrder => GRevLex})
installMethod(jason210, Ideal => opts -> () -> (
    R := (opts.CoefficientRing)[vars(0..7), MonomialOrder => opts.MonomialOrder];
    ideal "a6,
           b6,
           a2c4+b2d4+abc2e2+abd2f2+abcdeg+abcdfh"
    ))

lichtblau = method(Options => {CoefficientRing => ZZ/32003, MonomialOrder => GRevLex})
installMethod(lichtblau, Ideal => opts -> () -> (
    R := (opts.CoefficientRing)[vars(0..2), MonomialOrder => opts.MonomialOrder];
    ideal "374a11-2189a10+5555a9-8085a8+7590a7-5082a6+2772a5-1320a4+495a3-110a2+b,
           -22a11-88a10+550a9-1650a8+3300a7-3696a6+1848a5-330a3+110a2-22a+c"
    ))

virasoro = method(Options => {CoefficientRing => ZZ/32003, MonomialOrder => GRevLex})
installMethod(virasoro, Ideal => opts -> () -> (
    R := (opts.CoefficientRing)[vars(0..7), MonomialOrder => opts.MonomialOrder];
    ideal "8a2+8ab+8ac+2ad+2ae+2af+2ag-a-8bc-2dg-2ef,
           8ab-8ac+8b2+8bc+2bd+2be+2bf+2bg-b-2df-2eg,
	   -8ab+8ac+8bc+8c2+2cd+2ce+2cf+2cg-c-2de-2fg,
	   2ad-2ag+2bd-2bf+2cd-2ce+8d2+8de+2df+2dg+6dh-d-6eh,
	   2ae-2af+2be-2bg-2cd+2ce+8de-6dh+8e2+2ef+2eg+6eh-e,
	   -2ae+2af-2bd+2bf+2cf-2cg+2df+2ef+8f2+8fg+6fh-f-6gh,
	   -2ad+2ag-2be+2bg-2cf+2cg+2dg+2eg+8fg-6fh+8g2+6gh-g,
	   -6de+6dh+6eh-6fg+6fh+6gh+8h2-h"
    ))

chemkin = method(Options => {CoefficientRing => ZZ/32003, MonomialOrder => GRevLex})
installMethod(chemkin, Ideal => opts -> () -> (
    R := (opts.CoefficientRing)[vars(0..10), MonomialOrder => opts.MonomialOrder];
    ideal "-4ad+9d2+h,
           b2+e2+i2-1,
	   c2+f2+j2-1,
	   9g2+9k2-8,
	   -6abd+3b+3de+3hi-1,
	   3bc+3ef+3ij-1,
	   c+3fg+3jk-1,
	   -6a+3b+3c+8,
	   9d+9e+9f+9g+8,
	   h+i+j+k,
	   a2-2"
    ))

beginDocumentation()

TEST /// -- SugarPolynomial (construct with given sugar)
R = ZZ/32003[x,y]
f = sugarPolynomial(5, x^2 + x*y)
assert(polynomial f == x^2 + x*y)
assert(sugar f == 5)
///

TEST /// -- SugarPolynomial (construct without given sugar)
R = ZZ/32003[x,y]
f = sugarPolynomial(x^2 + x*y)
assert(polynomial f == x^2 + x*y)
assert(sugar f == 2)
///

TEST /// -- SugarPolynomial (construct without given sugar, sugar degree not in lead term)
R = ZZ/32003[x,y,z, MonomialOrder => Lex]
f = sugarPolynomial(x^2*y + y*z^10)
assert(polynomial f == x^2*y + y*z^10)
assert(sugar f == 11)
///

TEST /// -- reduce (Full/Head/Tail with regular polynomials)
R = QQ[x,y,z, MonomialOrder => Lex]
g = x^5*z + x^3*y + x^2*y^2 + x*y^2 + x
F = {x^2*z - x, x*y^2 + z^5, 4*x*z + z^3}
assert(reduce(g, F) == x^4 + x^3*y + x + (1/4)*z^7 - z^5)
assert(reduce(g, F, Reduce => "Head") == x^4 + x^3*y + x^2*y^2 + x*y^2 + x)
assert(reduce(g, F, Reduce => "Tail") == x^5*z + x^3*y + x + (1/4)*z^7 - z^5)
///

TEST /// -- reduce (Full/Head/Tail with sugar polynomials)
R = QQ[x,y,z, MonomialOrder => Lex]
g = sugarPolynomial(x^5*z + x^3*y + x^2*y^2 + x*y^2 + x)
F = {sugarPolynomial(x^2*z - x),
     sugarPolynomial(x*y^2 + z^5),
     sugarPolynomial(4*x*z + z^3)}
g1 = reduce(g, F)
g2 = reduce(g, F, Reduce => "Head")
g3 = reduce(g, F, Reduce => "Tail")
assert(polynomial g1 == x^4 + x^3*y + x + (1/4)*z^7 - z^5)
assert(sugar g1 == 7)
assert(polynomial g2 == x^4 + x^3*y + x^2*y^2 + x*y^2 + x)
assert(sugar g2 == 6)
assert(polynomial g3 == x^5*z + x^3*y + x + (1/4)*z^7 - z^5)
assert(sugar g3 == 7)
///

TEST /// -- reduce (DoubleSugar and Saccharine)
R = QQ[x,y,z, MonomialOrder => Lex]
g = sugarPolynomial(x^3*y*z^2 + x^2*z)
F = {sugarPolynomial(6, x^2 + y),
     sugarPolynomial(10, x*y*z + z),
     sugarPolynomial(3, x*z^2 + y^2)}
g1 = reduce(g, F)
g2 = reduce(g, F, Strategy => "DoubleSugar")
g3 = reduce(g, F, Strategy => "Saccharine")
assert(polynomial g1 == y*z^2 - y*z)
assert(sugar g1 == 12)
assert(polynomial g2 == -x^2*y^3 + x^2*z)
assert(sugar g2 == 6)
assert(polynomial g3 == y^4 - y*z)
assert(sugar g3 == 9)
///

TEST /// -- minimalize and interreduce (regular polynomials)
R = QQ[x,y,z, MonomialOrder => Lex]
G = {x*y^2 + z, x*z + 3*y, x^2 + y*z, -3*y^3 + z^2, -3*y - (1/3)*z^3, (1/243)*z^8 + z}
G' = minimalize(G)
G'' = interreduce(G')
assert(G' == {(1/243)*z^8 + z, -3*y - (1/3)*z^3, x*z + 3*y, x^2 + y*z})
assert(G'' == {z^8 + 243*z, y + (1/9)*z^3, x*z - (1/3)*z^3, x^2 - (1/9)*z^4})
///

TEST /// -- minimalize and interreduce (sugar polynomials)
R = QQ[x,y,z, MonomialOrder => Lex]
G = {x*y^2 + z, x*z + 3*y, x^2 + y*z, -3*y^3 + z^2, -3*y - (1/3)*z^3, (1/243)*z^8 + z}
G = apply(G, sugarPolynomial)
G' = minimalize(G)
G'' = interreduce(G')
assert(apply(G', polynomial) == {(1/243)*z^8 + z, -3*y - (1/3)*z^3, x*z + 3*y, x^2 + y*z})
assert(apply(G', sugar) == {8, 3, 2, 2})
assert(apply(G'', polynomial) == {z^8 + 243*z, y + (1/9)*z^3, x*z - (1/3)*z^3, x^2 - (1/9)*z^4})
assert(apply(G'', sugar) == {8, 3, 3, 4})
///

TEST /// -- spoly (basic example)
R = QQ[x,y]
f = x^2 + x*y
g = y^2 + x*y
assert(spoly(f, g) == 0)
fs = sugarPolynomial f
gs = sugarPolynomial g
assert(sugar spoly(fs, gs) == 3)
assert(polynomial spoly(fs, gs) == 0)
///

TEST /// -- spoly (division by lead coefficient over rationals)
R = QQ[x,y]
f = x^3*y^2 - x^2*y^3
g = 3*x^4*y + y^2
assert(spoly(f, g) == -x^3*y^3 - (1/3)*y^3)
fs = sugarPolynomial f
gs = sugarPolynomial g
assert(sugar spoly(fs, gs) == 6)
assert(polynomial spoly(fs, gs) == -x^3*y^3 - (1/3)*y^3)
///

TEST /// -- spoly (division by lead coefficient over finite field)
R = ZZ/32003[x,y]
f = x^3*y^2 - x^2*y^3
g = 3*x^4*y + y^2
assert(spoly(f, g) == -x^3*y^3 - (1/3)*y^3)
fs = sugarPolynomial f
gs = sugarPolynomial g
assert(sugar spoly(fs, gs) == 6)
assert(polynomial spoly(fs, gs) == -x^3*y^3 - (1/3)*y^3)
///

TEST /// -- spoly (lex order)
R = ZZ/32003[x,y, MonomialOrder => Lex]
f = x^2 - y^3
g = x*y^2 + x
assert(spoly(f, g) == -y^5 - x^2)
fs = sugarPolynomial f
gs = sugarPolynomial g
assert(sugar spoly(fs, gs) == 5)
assert(polynomial spoly(fs, gs) == -y^5 - x^2)
///

TEST /// -- full Groebner basis (LCM elimination)
I = katsura 4
(i, j, G1) = buchberger(I,
    SelectionStrategy => "First", EliminationStrategy => "LCM")
(i, j, G2) = buchberger(I,
    SelectionStrategy => "Random", EliminationStrategy => "LCM")
(i, j, G3) = buchberger(I,
    SelectionStrategy => "Normal", EliminationStrategy => "LCM")
(i, j, G4) = buchberger(I,
    SelectionStrategy => "Sugar", EliminationStrategy => "LCM")
G5 = first entries gens gb I
assert(isSubset(G1, G5) and isSubset(G5, G1))
assert(isSubset(G2, G5) and isSubset(G5, G2))
assert(isSubset(G3, G5) and isSubset(G5, G3))
assert(isSubset(G4, G5) and isSubset(G5, G4))
///

TEST /// -- full Groebner basis (GebauerMoeller elimination)
I = cyclic 5
(i, j, G1) = buchberger(I,
    SelectionStrategy => "First", EliminationStrategy => "GebauerMoeller")
(i, j, G2) = buchberger(I,
    SelectionStrategy => "Random", EliminationStrategy => "GebauerMoeller")
(i, j, G3) = buchberger(I,
    SelectionStrategy => "Normal", EliminationStrategy => "GebauerMoeller")
(i, j, G4) = buchberger(I,
    SelectionStrategy => "Sugar", EliminationStrategy => "GebauerMoeller")
G5 = first entries gens gb I
assert(isSubset(G1, G5) and isSubset(G5, G1))
assert(isSubset(G2, G5) and isSubset(G5, G2))
assert(isSubset(G3, G5) and isSubset(G5, G3))
assert(isSubset(G4, G5) and isSubset(G5, G4))
///

TEST /// -- full Groebner basis
R = QQ[x,y,z,t]
I = ideal(x^31 - x^6 - x - y, x^8 - z, x^10 - t)
(i, j, G1) = buchberger(I,
    SelectionStrategy => "Sugar", EliminationStrategy => "GebauerMoeller")
G2 = first entries gens gb I
assert(isSubset(G1, G2) and isSubset(G2, G1))
///

end--

restart
needsPackage "SelectionStrategies"
check SelectionStrategies

-- TODO: add random binomials to examples
GB = (I) -> flatten entries gens gb I
genquadbinom = (R) -> (
    n := numgens R;
    a := random n;
    b := random n;
    c := random n;
    d := random n;
    f := R_a * R_b - (random coefficientRing R) * R_c * R_d;
    if size f <= 1 then genquadbinom R else f
    )
genid = (R, nelems) -> ideal for i from 1 to nelems list genquadbinom R
R = ZZ/101[a,b,c,d]

-- cyclic roots
I = cyclic(5, MonomialOrder => Lex)
(i1, j1, G) = buchberger I
(i2, j2, G) = buchberger(I, ReductionStrategy => "DoubleSugar")
(i3, j3, G) = buchberger(I, Homogenize => true)
(i3, j2, G) = buchberger(I, SelectionStrategy => "Degree")
(i3, j3, G) = buchberger(I, SelectionStrategy => "Normal")

I = cyclic 5
(i1, j1, G) = buchberger I
(i1, j1, G) = buchberger(I, Homogenize => true)
(i2, j2, G) = buchberger(I, ReductionStrategy => "DoubleSugar")
(i3, j3, G) = buchberger(I, SelectionStrategy => "Normal")

-- parametric curves
R = QQ[x,y,z,t, MonomialOrder => Lex]
I = ideal(x^31 - x^6 - x - y, x^8 - z, x^10 - t)
(i1, j1, G) = buchberger I
(i2, j2, G) = buchberger(I, ReductionStrategy => "DoubleSugar")

R = QQ[x,y,z,t]
I = ideal(x^31 - x^6 - x - y, x^8 - z, x^10 - t)
(i1, j1, G) = buchberger I
(i2, j2, G) = buchberger(I, ReductionStrategy => "DoubleSugar")
(i3, j3, G) = buchberger(I, SelectionStrategy => "Normal")

-- trig example
needsPackage "NumericalAlgebraicGeometry"
needsPackage "NautyGraphs"
needsPackage "Visualize"
needsPackage "MinimalPrimes"
needsPackage "Bertini"
installMinprimes()

trig = method()
trig Ring := (R) -> (
    n := numgens R // 2;
    ideal for i from 0 to n - 1 list R_i^2 + R_(n+i)^2 - 1
    )

oscSystem = method()
oscSystem(Graph, Ring) := (G, R) -> (
    n := # vertices G - 1;  -- R should have == 2n variables.
    assert(n == (numgens R // 2));
    sine := (i) -> if i === 0 then 0_R else R_(n+i-1);
    cosine := (i) -> if i === 0 then 1_R else R_(i-1);
    I := ideal for i from 0 to n-1 list (
        N := toList neighbors(G,i);
        sum for j in N list (
            sine j * cosine i - sine i * cosine j
            )
        );
    I + trig R
    )

Gstrings = generateGraphs(6, OnlyConnected => true)
G = stringToGraph(Gstrings_10)
R = ZZ/32003[x_1..x_5,y_1..y_5]
I = oscSystem(G, R)
(i, j, g) = buchberger(I)

G = stringToGraph(Gstrings_100)
R = ZZ/32003[x_1..x_5,y_1..y_5, MonomialOrder => Lex]
I = oscSystem(G, R)
(i, j, g) = buchberger(I)
elapsedTime (i, j, g) = buchberger(I, ReductionStrategy => "DoubleSugar")
#Gstrings
