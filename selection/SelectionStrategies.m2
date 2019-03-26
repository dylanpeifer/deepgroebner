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

export {"SugarPolynomial", "polynomial", "sugar", "spoly", "reduce",
    "minimalize", "interreduce", "cyclic", "hcyclic", "katsura",
    "buchberger", "SelectionStrategy", "EliminationStrategy",
    "ReductionStrategy", "Homogenize"}

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
SugarPolynomial = new Type of List;

new SugarPolynomial from RingElement := (S, f) -> (
    {first degree f, f}
    )

sugar = method()
sugar(SugarPolynomial) := ZZ => (f) -> (
    f#0
    )
sugar(RingElement) := ZZ => (f) -> (
    first degree f
    )

polynomial = method()
polynomial(SugarPolynomial) := RingElement => (f) -> (
    f#1
    )

leadTerm(SugarPolynomial) := RingElement => (f) -> (
    leadTerm polynomial f
    )

leadMonomial(SugarPolynomial) := RingElement => (f) -> (
    leadMonomial polynomial f
    )

leadCoefficient(SugarPolynomial) := RingElement => (f) -> (
    leadCoefficient polynomial f
    )

SugarPolynomial + SugarPolynomial := SugarPolynomial => (f, g) -> (
    new SugarPolynomial from {max(sugar f, sugar g), (polynomial f) + (polynomial g)}
    )

RingElement + SugarPolynomial := SugarPolynomial => (f, g) -> (
    new SugarPolynomial from {max(first degree f, sugar g), f + (polynomial g)}
    )

SugarPolynomial + RingElement := SugarPolynomial => (f, g) -> (
    new SugarPolynomial from {max(sugar f, first degree g), (polynomial f) + g}
    )

ZZ + SugarPolynomial := SugarPolynomial => (f, g) -> (
    new SugarPolynomial from {sugar g, f + (polynomial g)}
    )

SugarPolynomial - SugarPolynomial := SugarPolynomial => (f, g) -> (
    new SugarPolynomial from {max(sugar f, sugar g), (polynomial f) - (polynomial g)}
    )

RingElement - SugarPolynomial := SugarPolynomial => (f, g) -> (
    new SugarPolynomial from {max(first degree f, sugar g), f - (polynomial g)}
    )

SugarPolynomial - RingElement := SugarPolynomial => (f, g) -> (
    new SugarPolynomial from {max(sugar f, first degree g), (polynomial f) - g}
    )

SugarPolynomial * SugarPolynomial := SugarPolynomial => (f, g) -> (
    new SugarPolynomial from {(sugar f) + (sugar g), (polynomial f) * (polynomial g)}
    )

RingElement * SugarPolynomial := SugarPolynomial => (f, g) -> (
    new SugarPolynomial from {(first degree f) + (sugar g), f * (polynomial g)}
    )

SugarPolynomial * RingElement := SugarPolynomial => (f, g) -> (
    new SugarPolynomial from {(sugar f) + (first degree g), (polynomial f) * g}
    )

QQ * SugarPolynomial := SugarPolynomial => (f, g) -> (
    new SugarPolynomial from {sugar g, f * (polynomial g)}
    )

SugarPolynomial * QQ := SugarPolynomial => (f, g) -> (
    new SugarPolynomial from {sugar f, (polynomial f) * g}
    )

SugarPolynomial == SugarPolynomial := Boolean => (f, g) -> (
    (polynomial f) == (polynomial g)
    )

SugarPolynomial == RingElement := Boolean => (f, g) -> (
    (polynomial f) == g
    )

RingElement == SugarPolynomial := Boolean => (f, g) -> (
    f == (polynomial g)
    )

SugarPolynomial == ZZ := Boolean => (f, g) -> (
    (polynomial f) == g
    )

ZZ == SugarPolynomial := Boolean => (f, g) -> (
    f == (polynomial g)
    )

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
	r = r + (leadTerm g);
	g = g - (leadTerm g);
	);
    
    doubleSugar := opts.Strategy === "DoubleSugar" or opts.Strategy === "Saccharine";

    while g != 0 do (
	lg := leadTerm g;
	foundDivisor := false;

	-- try to remove lead term by some f, don't increase sugar if doubleSugar is on
	for f in F do (
	    lf := leadTerm f;
	    if (lg % lf) == 0 then (
		reducer := (lg//lf) * f;
		if doubleSugar and (sugar reducer) > (sugar g) then (
		    continue;
		    )
		else (
		    g = g - reducer;
		    foundDivisor = true;
		    break;
		    );
	        );
	    );

	if not foundDivisor then (
	    -- if first try didn't work on Saccharine then try again without doubleSugar
	    if opts.Strategy === "Saccharine" and doubleSugar then (
		doubleSugar = false;
		continue;
		)
	    -- if only head reducing then stop here and return g
	    else if opts.Reduce === "Head" then (
		break;
		)
	    -- otherwise move lead term of g to remainder and continue reducing g
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
    for f in F do (
	if all(G, g -> (leadTerm f) % (leadTerm g) != 0) then (
	    G = select(G, g -> (leadTerm g) % (leadTerm f) != 0);
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
SPair = new Type of List;

spair = method()
spair(ZZ, ZZ, List) := SPair => (i, j, F) -> (
    gamma := lcm(leadMonomial F#i, leadMonomial F#j);
    sug := max((sugar F#i) + (first degree (gamma//leadMonomial F#i)),
	       (sugar F#j) + (first degree (gamma//leadMonomial F#j)));
    new SPair from {i, j, gamma, sug}
    )

lcm(SPair) := RingElement => (p) -> (
    p#2
    )

sugar(SPair) := ZZ => (p) -> (
    p#3
    )

degree(SPair) := ZZ => (p) -> (
    degree lcm p
    )

spoly = method()
spoly(RingElement, RingElement) := RingElement =>
spoly(SugarPolynomial, SugarPolynomial) := SugarPolynomial => (f, g) -> (
    -- f = a polynomial
    -- g = a polynomial
    -- returns the s-polynomial of f and g

    gamma := lcm(leadMonomial f, leadMonomial g);
    (gamma // (leadTerm f)) * f - (gamma // (leadTerm g)) * g
    )
spoly(SPair, List) := (p, F) -> (
    -- p = an SPair
    -- F = a list of polynomials
    -- returns the s-polynomial given by p

    spoly(F#(p#0), F#(p#1))
    )

-------------------------------------------------------------------------------
--- s-pair selection and updates
-------------------------------------------------------------------------------
lcmCriterion = method()
lcmCriterion(SPair, List) := Boolean => (p, F) -> (
    -- p = an SPair
    -- F = the corresponding list of polynomials
    -- returns true if the pair satisfies the lcm criterion
    
    lcm p == (leadMonomial F#(p#0)) * (leadMonomial F#(p#1))
    )

selectPair = method(Options => {Strategy => "First"})
selectPair(List) := SPair => opts -> (P) -> (
    -- P = a list of SPairs in Buchberger's algorithm
    -- returns the next pair to process

    if opts.Strategy === "First" then (
	P#0
	)
    else if opts.Strategy === "Random" then (
	P#(random(#P))
	)
    else if opts.Strategy === "Degree" then (
	P#(argmin(P, degree))
	)
    else if opts.Strategy === "Normal" then (
	P#(argmin(P, lcm))
	)
    else if opts.Strategy === "Sugar" then (
	P#(argmin(P, p -> {sugar p, lcm p}))
	)
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
    newPairs := apply(#F-1, i -> spair(i, #F-1, F));

    if opts.Strategy === "LCM" then (
	newPairs = select(newPairs, p -> not lcmCriterion(p, F));
	)
    else (
	-- eliminate from old list
	lf := leadMonomial f;
	P = select(P, p ->
	    (lcm p) % lf != 0 or
	    lcm(leadMonomial F#(p#0), lf) == lcm p or
	    lcm(leadMonomial F#(p#1), lf) == lcm p);
	
	-- sugar paper eliminates lcm criterion here (this seems wrong)
    	P' := {};
	for p in newPairs do (
	    equiv := select(newPairs, p' -> lcm p == lcm p');
	    if any(equiv, p' -> lcmCriterion(p', F)) then continue;
	    P' = P' | equiv;
	    );
	newPairs = P';

    	-- eliminate if strictly divisible
    	newPairs = select(newPairs, p ->
	    all(newPairs, p' -> lcm p % lcm p' != 0 or lcm p == lcm p'));

    	-- keep 1 of each equivalence class and remove any with lcm criterion
    	P' = {};
	for p in newPairs do (
	    equiv := select(newPairs, p' -> lcm p == lcm p');
	    if any(equiv, p' -> lcmCriterion(p', F)) then continue;
	    if any(P', p' -> lcm p == lcm p') then continue;
	    P' = append(P', p);
	    );
	newPairs = P';
	);

    (P | newPairs, F)
    )

-------------------------------------------------------------------------------
--- main algorithm
-------------------------------------------------------------------------------
BuchbergerHistory = new Type of HashTable;

buchberger = method(Options => {
	SelectionStrategy => "Sugar",
	EliminationStrategy => "GebauerMoeller",
	ReductionStrategy => "Regular",
	Homogenize => false
	})
buchberger(Ideal) := BuchbergerHistory => opts -> (I) -> (
    -- I = an ideal in a polynomial ring
    -- returns number of pairs processed in computing a Groebner basis of I

    F := first entries gens I;
    if opts.SelectionStrategy === "Sugar" then (
	F = apply(F, f -> new SugarPolynomial from f);
	);

    -- initialize pairs P and polynomials G
    P := {};
    G := {};
    for f in F do (
	(P, G) = updatePairs(P, G, f, Strategy => opts.EliminationStrategy);
	);

    zeroReductions := 0;
    nonzeroReductions := 0;

    while #P > 0 do (
	p := selectPair(P, Strategy => opts.SelectionStrategy);
	P = delete(p, P);

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

    if opts.SelectionStrategy === "Sugar" then (
	G = apply(G, polynomial);
	);
    G = interreduce(minimalize(G));

    (zeroReductions, nonzeroReductions, G)
    )

-------------------------------------------------------------------------------
--- example ideals
-------------------------------------------------------------------------------
cyclic = method(Options => {CoefficientRing => ZZ/32003, MonomialOrder => GRevLex})
cyclic(ZZ) := Ideal => opts -> (n) -> (
    R := (opts.CoefficientRing)[vars(0..n-1), MonomialOrder => opts.MonomialOrder];
    F := toList apply(1..n-1, d -> sum(0..n-1, i -> product(d, k -> R_((i+k)%n))))
         | {product gens R - 1};
    ideal F
    )

hcyclic = method(Options => {CoefficientRing => ZZ/32003, MonomialOrder => GRevLex})
hcyclic(ZZ) := Ideal => opts -> (n) -> (
    R := (opts.CoefficientRing)[vars(0..n), MonomialOrder => opts.MonomialOrder];
    F := toList apply(1..n-1, d -> sum(0..n-1, i -> product(d, k -> R_((i+k)%n)))) 
         | {product(n, i -> R_i) - R_n^n};
    ideal F
    )

katsura = method(Options => {CoefficientRing => ZZ/32003, MonomialOrder => GRevLex})
katsura(ZZ) := Ideal => opts -> (n) -> (
    n = n-1;
    R := (opts.CoefficientRing)[vars(0..n), MonomialOrder => opts.MonomialOrder];
    L := gens R;
    u := i -> (
	 if i < 0 then i = -i;
	 if i <= n then L_i else 0_R
	 );
    f1 := -1 + sum for i from -n to n list u i;
    F := toList prepend(f1, apply(0..n-1, i -> - u i + sum(-n..n, j -> (u j) * (u (i-j)))));
    ideal F
    )

beginDocumentation()

TEST /// -- SugarPolynomial (construct with given sugar)
R = ZZ/32003[x,y]
f = new SugarPolynomial from {5, x^2 + x*y}
assert(polynomial f == x^2 + x*y)
assert(sugar f == 5)
///

TEST /// -- SugarPolynomial (construct without given sugar)
R = ZZ/32003[x,y]
f = new SugarPolynomial from x^2 + x*y
assert(polynomial f == x^2 + x*y)
assert(sugar f == 2)
///

TEST /// -- SugarPolynomial (construct without given sugar, sugar degree not in lead term)
R = ZZ/32003[x,y,z, MonomialOrder => Lex]
f = new SugarPolynomial from x^2*y + y*z^10
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
g = new SugarPolynomial from x^5*z + x^3*y + x^2*y^2 + x*y^2 + x
F = {new SugarPolynomial from x^2*z - x,
     new SugarPolynomial from x*y^2 + z^5,
     new SugarPolynomial from 4*x*z + z^3}
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
g = new SugarPolynomial from x^3*y*z^2 + x^2*z
F = {new SugarPolynomial from {6, x^2 + y},
     new SugarPolynomial from {10, x*y*z + z},
     new SugarPolynomial from {3, x*z^2 + y^2}}
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
assert(G' == {x*z + 3*y, x^2 + y*z, -3*y - (1/3)*z^3, (1/243)*z^8 + z})
assert(G'' == {x*z - (1/3)*z^3, x^2 - (1/9)*z^4, y + (1/9)*z^3, z^8 + 243*z})
///

TEST /// -- minimalize and interreduce (sugar polynomials)
R = QQ[x,y,z, MonomialOrder => Lex]
G = {x*y^2 + z, x*z + 3*y, x^2 + y*z, -3*y^3 + z^2, -3*y - (1/3)*z^3, (1/243)*z^8 + z}
G = apply(G, g -> new SugarPolynomial from g)
G' = minimalize(G)
G'' = interreduce(G')
assert(apply(G', polynomial) == {x*z + 3*y, x^2 + y*z, -3*y - (1/3)*z^3, (1/243)*z^8 + z})
assert(apply(G', sugar) == {2, 2, 3, 8})
assert(apply(G'', polynomial) == {x*z - (1/3)*z^3, x^2 - (1/9)*z^4, y + (1/9)*z^3, z^8 + 243*z})
assert(apply(G'', sugar) == {3, 4, 3, 8})
///

TEST /// -- spoly (basic example)
R = QQ[x,y]
f = x^2 + x*y
g = y^2 + x*y
assert(spoly(f, g) == 0)
fs = new SugarPolynomial from f
gs = new SugarPolynomial from g
assert(spoly(fs, gs) == new SugarPolynomial from {3, 0})
///

TEST /// -- spoly (division by lead coefficient over rationals)
R = QQ[x,y]
f = x^3*y^2 - x^2*y^3
g = 3*x^4*y + y^2
assert(spoly(f, g) == -x^3*y^3 - (1/3)*y^3)
fs = new SugarPolynomial from f
gs = new SugarPolynomial from g
assert(spoly(fs, gs) == new SugarPolynomial from {6, -x^3*y^3 - (1/3)*y^3})
///

TEST /// -- spoly (division by lead coefficient over finite field)
R = ZZ/32003[x,y]
f = x^3*y^2 - x^2*y^3
g = 3*x^4*y + y^2
assert(spoly(f, g) == -x^3*y^3 - (1/3)*y^3)
fs = new SugarPolynomial from f
gs = new SugarPolynomial from g
assert(spoly(fs, gs) == new SugarPolynomial from {6, -x^3*y^3 - (1/3)*y^3})
///

TEST /// -- spoly (lex order)
R = ZZ/32003[x,y, MonomialOrder => Lex]
f = x^2 - y^3
g = x*y^2 + x
assert(spoly(f, g) == -y^5 - x^2)
fs = new SugarPolynomial from f
gs = new SugarPolynomial from g
assert(spoly(fs, gs) == new SugarPolynomial from {5, -y^5 - x^2})
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

-- TODO: sugar fails because of leadCoefficient * SugarPolynomial
R = ZZ/32003[x]
interreduce({new SugarPolynomial from x})

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

-- cyclic roots numbers now agree with sugar paper
I = cyclic(5, MonomialOrder => Lex)
(i1, j1, G) = buchberger I
(i2, j2, G) = buchberger(I, ReductionStrategy => "DoubleSugar")

I = cyclic 5
(i1, j1, G) = buchberger I
(i2, j2, G) = buchberger(I, ReductionStrategy => "DoubleSugar")
(i3, j3, G) = buchberger(I, SelectionStrategy => "Normal")

-- parametric curves numbers now agree with sugar paper (except DoubleSugar)
R = QQ[x,y,z,t, MonomialOrder => Lex]
I = ideal(x^31 - x^6 - x - y, x^8 - z, x^10 - t)
(i1, j1, G) = buchberger I
(i2, j2, G) = buchberger(I, ReductionStrategy => "DoubleSugar")

R = QQ[x,y,z,t]
I = ideal(x^31 - x^6 - x - y, x^8 - z, x^10 - t)
(i1, j1, G) = buchberger I
(i2, j2, G) = buchberger(I, ReductionStrategy => "DoubleSugar")
(i3, j3, G) = buchberger(I, SelectionStrategy => "Normal")



