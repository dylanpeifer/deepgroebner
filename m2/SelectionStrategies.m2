newPackage(
        "SelectionStrategies",
        Version => "0.0.1", 
        Date => "January 14, 2020",
        Authors => {{Name => "Dylan Peifer", 
                     Email => "djp282@cornell.edu", 
                     HomePage => "https://www.math.cornell.edu/~djp282"}},
        Headline => "Test pair selection strategies in Buchberger's algorithm",
        DebuggingMode => true
        )

export {"SugarPolynomial", "sugarPolynomial", "polynomial", "sugar",
        "reduce", "minimalize", "interreduce", "spoly",
        "buchberger", "SelectionStrategy", "EliminationStrategy", "SortReducers",
	"ReductionStrategy", "Homogenize", "Minimalize", "Interreduce", "SortInput",
	"Lineages"}

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

terms SugarPolynomial           := f -> terms polynomial f
leadTerm SugarPolynomial        := f -> leadTerm polynomial f
leadMonomial SugarPolynomial    := f -> leadMonomial polynomial f
leadCoefficient SugarPolynomial := f -> leadCoefficient polynomial f
degree SugarPolynomial          := f -> degree polynomial f

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
reduce = method(Options => {
	Strategy => "Regular",
	Reduce => "Full",
	SortReducers => false})
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
    
    if opts.SortReducers then F = sort F;

    doubleSugar := opts.Strategy === "DoubleSugar" or opts.Strategy === "Saccharine";

    polynomialAdditions := 0;
    monomialAdditions := 0;

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
		    polynomialAdditions = polynomialAdditions + 1;
		    monomialAdditions = monomialAdditions + length terms reducer;
		    foundDivisor = true;
		    break;
		    );
	        );
	    );

    	-- if first pass didn't find a divisor then we:
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
    
    stats := hashTable {"polynomialAdditions" => polynomialAdditions,
	                "monomialAdditions" => monomialAdditions};

    -- r holds remainder and g holds sugar degree and/or unreduced tail terms
    (r + g, stats)
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
	(g, stats) := reduce(f, F, Reduce => "Tail");
	G = append(G, 1/(leadCoefficient g) * g);
	);
    G
    )

-------------------------------------------------------------------------------
--- utility functions
-------------------------------------------------------------------------------
argmax = method()
argmax(List) := ZZ => (x) -> (
    -- x = a list
    -- returns the index of the max element of x (or the last index if there are multiple)
    if #x === 0 then null else first fold((i,j) -> if i#1 > j#1 then i else j, pairs x)
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
    s := (spoly(f, g))#0;
    td := first degree s;
    im := (if # terms s == 1 then 0 else 1);  -- sort key that favors monomials
    new SPair from {S, gamma, sug, td, im}
    )

indices SPair := Sequence    => p -> p#0
lcm     SPair := RingElement => p -> p#1
sugar   SPair := ZZ          => p -> p#2
degree  SPair := ZZ          => p -> first degree lcm p

trueDegree = method()
trueDegree SPair := ZZ => p -> p#3

isMonomial = method()
isMonomial SPair := ZZ => p -> p#4

SPair ? SPair := (s1, s2) -> indices s1 ? indices s2

spoly = method()
spoly(RingElement, RingElement) := Sequence =>
spoly(SugarPolynomial, SugarPolynomial) := Sequence => (f, g) -> (
    -- f = a polynomial
    -- g = a polynomial
    -- returns the s-polynomial of f and g and stats

    gamma := lcm(leadMonomial f, leadMonomial g);
    ((gamma // leadTerm f) * f - (gamma // leadTerm g) * g, min({f, g}/terms/length))
    )
spoly(SPair, List) := (p, F) -> (
    -- p = an SPair
    -- F = a list of polynomials
    -- returns the s-polynomial given by p and stats

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
	p = P#0;
	)
    else if opts.Strategy === "Degree" then (
	p = P#(argmin(P, degree));
	)
    else if opts.Strategy === "Normal" then (
	p = P#(argmin(P, lcm));
	)
    else if opts.Strategy === "Sugar" then (
	p = P#(argmin(P, p -> {sugar p, lcm p}));
	)
    else if opts.Strategy === "Random" then (
	p = P#(random(#P));
	)
    else if opts.Strategy === "TrueDegree" then (
	p = P#(argmin(P, trueDegree));
	)
    else if opts.Strategy === "MonomialDegree" then (
	p = P#(argmin(P, p -> {isMonomial p, degree p}));
	)
    else if opts.Strategy === "MonomialTrueDegree" then (
	p = P#(argmin(P, p -> {isMonomial p, trueDegree p}));
	)
    else if opts.Strategy === "MonomialTrueDegreeDegree" then (
	p = P#(argmin(P, p -> {isMonomial p, trueDegree p, degree p}));
	)
    else if opts.Strategy === "Last" then (
	p = P#(#P-1);
	)
    else if opts.Strategy === "Codegree" then (
	p = P#(argmax(P, degree));
	)
    else if opts.Strategy === "Strange" then (
	p = P#(argmax(P, lcm));
	)
    else if opts.Strategy === "Spice" then (
	p = P#(argmax(P, p -> {sugar p, lcm p}));
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
	P' = sort(P'');
	);

    (P | P', F)
    )

-------------------------------------------------------------------------------
--- main algorithm
-------------------------------------------------------------------------------
buchberger = method(Options => {
	SelectionStrategy => "Sugar",
	EliminationStrategy => "GebauerMoeller",
	ReductionStrategy => "Regular",
	SortInput => false,
	SortReducers => true,
	Homogenize => false,
	Minimalize => true,
	Interreduce => true
	})
buchberger(Ideal) := Sequence => opts -> I -> (
    -- I = an ideal in a polynomial ring
    -- returns number of pairs processed in computing a Groebner basis of I

    F := first entries gens I;
    if opts.SortInput then F = sort F;
    if opts.SelectionStrategy === "Sugar" or opts.SelectionStrategy === "Spice" then (
	F = apply(F, sugarPolynomial);
	);

    P := {};
    G := {};
    for f in F do (
	(P, G) = updatePairs(P, G, f, Strategy => opts.EliminationStrategy);
	);

    reducers := G;
    if opts.SortReducers then reducers = sort reducers;

    zeroReductions := 0;
    nonzeroReductions := 0;
    polynomialAdditions := 0;
    monomialAdditions := 0;
    p := 0;
    s := 0;
    adds := 0;

    while #P > 0 do (
	(p, P) = selectPair(P, Strategy => opts.SelectionStrategy);
	(s, adds) = spoly(p, G);
	(r, stats) := reduce(s, reducers, Strategy => opts.ReductionStrategy);
	polynomialAdditions = polynomialAdditions + stats#"polynomialAdditions" + 1;
	monomialAdditions = monomialAdditions + stats#"monomialAdditions" + adds;

	if r != 0 then (
	    (P, G) = updatePairs(P, G, r, Strategy => opts.EliminationStrategy);
	    nonzeroReductions = nonzeroReductions + 1;
	    reducers = G;
	    if opts.SortReducers then reducers = sort reducers;
	    )
	else (
	    zeroReductions = zeroReductions + 1;
	    );
	);

    if opts.SelectionStrategy === "Sugar" or opts.SelectionStrategy === "Spice" then (
	G = apply(G, polynomial);
	);
    if opts.Minimalize then G = minimalize(G);
    if opts.Interreduce then G = interreduce(G);

    stats = hashTable {"zeroReductions" => zeroReductions,
	               "nonzeroReductions" => nonzeroReductions,
		       "polynomialAdditions" => polynomialAdditions,
		       "monomialAdditions" => monomialAdditions};
    (G, stats)
    )

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
(r1, stats) = reduce(g, F)
(r2, stats) = reduce(g, F, Reduce => "Head")
(r3, stats) = reduce(g, F, Reduce => "Tail")
assert(r1 == x^4 + x^3*y + x + (1/4)*z^7 - z^5)
assert(r2 == x^4 + x^3*y + x^2*y^2 + x*y^2 + x)
assert(r3 == x^5*z + x^3*y + x + (1/4)*z^7 - z^5)
///

TEST /// -- reduce (Full/Head/Tail with sugar polynomials)
R = QQ[x,y,z, MonomialOrder => Lex]
g = sugarPolynomial(x^5*z + x^3*y + x^2*y^2 + x*y^2 + x)
F = {sugarPolynomial(x^2*z - x),
     sugarPolynomial(x*y^2 + z^5),
     sugarPolynomial(4*x*z + z^3)}
(g1, stats) = reduce(g, F)
(g2, stats) = reduce(g, F, Reduce => "Head")
(g3, stats) = reduce(g, F, Reduce => "Tail")
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
(g1, stats) = reduce(g, F)
(g2, stats) = reduce(g, F, Strategy => "DoubleSugar")
(g3, stats) = reduce(g, F, Strategy => "Saccharine")
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
assert(spoly(f, g) == (0, 2))
fs = sugarPolynomial f
gs = sugarPolynomial g
(s, adds) = spoly(fs, gs)
assert(sugar s == 3)
assert(polynomial s == 0)
assert(adds == 2)
///

TEST /// -- spoly (division by lead coefficient over rationals)
R = QQ[x,y]
f = x^3*y^2 - x^2*y^3
g = 3*x^4*y + y^2
assert(spoly(f, g) == (-x^3*y^3 - (1/3)*y^3, 2))
fs = sugarPolynomial f
gs = sugarPolynomial g
(s, adds) = spoly(fs, gs)
assert(sugar s == 6)
assert(polynomial s == -x^3*y^3 - (1/3)*y^3)
assert(adds == 2)
///

TEST /// -- spoly (division by lead coefficient over finite field)
R = ZZ/32003[x,y]
f = x^3*y^2 - x^2*y^3
g = 3*x^4*y + y^2
assert(spoly(f, g) == (-x^3*y^3 - (1/3)*y^3, 2))
fs = sugarPolynomial f
gs = sugarPolynomial g
(s, adds) = spoly(fs, gs)
assert(sugar s == 6)
assert(polynomial s == -x^3*y^3 - (1/3)*y^3)
assert(adds == 2)
///

TEST /// -- spoly (lex order)
R = ZZ/32003[x,y, MonomialOrder => Lex]
f = x^2 - y^3
g = x*y^2 + x
assert(spoly(f, g) == (-y^5 - x^2, 2))
fs = sugarPolynomial f
gs = sugarPolynomial g
(s, adds) = spoly(fs, gs)
assert(sugar s == 5)
assert(polynomial s == -y^5 - x^2)
assert(adds == 2)
///

TEST /// -- Groebner basis of katsura4 (LCM elimination)
R = ZZ/32003[a,b,c,d]
I = ideal(a + 2*b + 2*c + 2*d - 1,
          a^2 + 2*b^2 + 2*c^2 + 2*d^2 - a,
	  2*a*b + 2*b*c + 2*c*d - b,
	  b^2 + 2*a*c + 2*b*d - c)
(G1, stats) = buchberger(I,
    SelectionStrategy => "First", EliminationStrategy => "LCM")
(G2, stats) = buchberger(I,
    SelectionStrategy => "Random", EliminationStrategy => "LCM")
(G3, stats) = buchberger(I,
    SelectionStrategy => "Normal", EliminationStrategy => "LCM")
(G4, stats) = buchberger(I,
    SelectionStrategy => "Sugar", EliminationStrategy => "LCM")
G5 = first entries gens gb I
assert(isSubset(G1, G5) and isSubset(G5, G1))
assert(isSubset(G2, G5) and isSubset(G5, G2))
assert(isSubset(G3, G5) and isSubset(G5, G3))
assert(isSubset(G4, G5) and isSubset(G5, G4))
///

TEST /// -- Groebner basis of cyclic5 (GebauerMoeller elimination)
R = ZZ/32003[a,b,c,d,e]
I = ideal(a + b + c + d + e,
          a*b + b*c + c*d + a*e + d*e,
          a*b*c + b*c*d + a*b*e + a*d*e + c*d*e,
	  a*b*c*d + a*b*c*e + a*b*d*e + a*c*d*e + b*c*d*e,
	  a*b*c*d*e - 1)
(G1, stats) = buchberger(I,
    SelectionStrategy => "First", EliminationStrategy => "GebauerMoeller")
(G2, stats) = buchberger(I,
    SelectionStrategy => "Random", EliminationStrategy => "GebauerMoeller")
(G3, stats) = buchberger(I,
    SelectionStrategy => "Normal", EliminationStrategy => "GebauerMoeller")
(G4, stats) = buchberger(I,
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
(G1, stats) = buchberger(I,
    SelectionStrategy => "Sugar", EliminationStrategy => "GebauerMoeller")
G2 = first entries gens gb I
assert(isSubset(G1, G2) and isSubset(G2, G1))
///

end
