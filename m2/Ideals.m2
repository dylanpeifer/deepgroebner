newPackage(
        "Ideals",
        Version => "0.0.0",
        Date => "September 9, 2019",
        Authors => {{Name => "Dylan Peifer", 
                     Email => "djp282@cornell.edu", 
                     HomePage => "https://www.math.cornell.edu/~djp282"}},
        Headline => "A collection of example ideals",
        DebuggingMode => true
        )

export {"commats", "cyclic", "eco", "extcyc", "hcyclic", "katsura", "noon",
        "perms", "redcyc", "reimer",
        "chemkin", "haas", "jason210", "kotsireas", "lichtblau", "virasoro",
        "randomBinomialIdeal", "Pure"}

-------------------------------------------------------------------------------
--- ideal families
-------------------------------------------------------------------------------
commats = method(Options => {CoefficientRing => ZZ/32003, MonomialOrder => GRevLex})
commats ZZ := Ideal => opts -> n -> (
    R := (opts.CoefficientRing)(monoid[vars(0..2*n*n-1), MonomialOrder => opts.MonomialOrder]);
    A := genericMatrix(R, R_0, n, n);
    B := genericMatrix(R, R_(n*n), n, n);
    ideal flatten entries (A*B - B*A)
    )

cyclic = method(Options => {CoefficientRing => ZZ/32003, MonomialOrder => GRevLex})
cyclic ZZ := Ideal => opts -> n -> (
    R := (opts.CoefficientRing)(monoid[vars(0..n-1), MonomialOrder => opts.MonomialOrder]);
    F := toList apply(1..n-1, d -> sum(0..n-1, i -> product(d, k -> R_((i+k)%n))))
         | {product gens R - 1};
    ideal F
    )

eco = method(Options => {CoefficientRing => ZZ/32003, MonomialOrder => GRevLex})
eco ZZ := Ideal => opts -> n -> (
    R := (opts.CoefficientRing)[vars(0..n-1), MonomialOrder => opts.MonomialOrder];
    F := toList apply(0..n-2, k -> R_(n-1) * (R_k + sum(0..n-k-3, i -> R_i * R_(i+k+1))) - k - 1)
         | {sum(0..n-2, i -> R_i) + 1};
    ideal F
    )

extcyc = method(Options => {CoefficientRing => ZZ/32003, MonomialOrder => GRevLex})
extcyc ZZ := Ideal => opts -> n -> (
    R := (opts.CoefficientRing)[vars(0..n), MonomialOrder => opts.MonomialOrder];
    F := toList apply(1..n-1, d -> R_0^d + sum(0..n-1, i -> product(d, k -> R_((i+k)%n+1))))
         | {product(1..n, i -> R_i) - 1, R_0^n + 1};
    ideal F
    )

hcyclic = method(Options => {CoefficientRing => ZZ/32003, MonomialOrder => GRevLex})
hcyclic ZZ := Ideal => opts -> n -> (
    R := (opts.CoefficientRing)(monoid[vars(0..n), MonomialOrder => opts.MonomialOrder]);
    F := toList apply(1..n-1, d -> sum(0..n-1, i -> product(d, k -> R_((i+k)%n))))
         | {product(n, i -> R_i) - R_n^n};
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

noon = method(Options => {CoefficientRing => ZZ/32003, MonomialOrder => GRevLex})
noon ZZ := Ideal => opts -> n -> (
    R := (opts.CoefficientRing)[vars(0..n-1), MonomialOrder => opts.MonomialOrder];
    F := apply(0..n-1, i -> 10 * R_i * (sum(0..n-1, j -> R_j^2) - R_i^2) - 11 * R_i + 10);
    ideal F
    )

perms = method(Options => {CoefficientRing => ZZ/32003, MonomialOrder => GRevLex})
perms(ZZ, ZZ, ZZ) := Ideal => opts -> (m, n, s) -> (
    R := (opts.CoefficientRing)[vars(0..m*n-1), MonomialOrder => opts.MonomialOrder];
    permanents(s, genericMatrix(R, m, n))
    )

redcyc = method(Options => {CoefficientRing => ZZ/32003, MonomialOrder => GRevLex})
redcyc ZZ := Ideal => opts -> n -> (
    R := (opts.CoefficientRing)[vars(0..n), MonomialOrder => opts.MonomialOrder];
    F := toList apply(1..n-1, d -> sum(0..n-1, i -> product(d, k -> R_((i+k)%n))));
    F = apply(F, f -> sub(f, R_(n-1) => 1));
    F = append(F, sub(product(n, i -> R_i), R_(n-1) => R_(n-1)^n) - 1);
    ideal F
    )

reimer = method(Options => {CoefficientRing => ZZ/32003, MonomialOrder => GRevLex})
reimer ZZ := Ideal => opts -> n -> (
    R := (opts.CoefficientRing)[vars(0..n-1), MonomialOrder => opts.MonomialOrder];
    F := apply(2..n+1, d -> sum(0..n-1, i -> (-1)^i * 2 * R_i^d) - 1);
    ideal F
    )

-------------------------------------------------------------------------------
--- fixed ideals
-------------------------------------------------------------------------------
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

haas = method(Options => {CoefficientRing => ZZ/32003, MonomialOrder => GRevLex})
installMethod(haas, Ideal => opts -> () -> (
    -- from Hashemi, Efficient algorithms for computing Noether normalization
    R := (opts.CoefficientRing)[vars(0..3), MonomialOrder => opts.MonomialOrder];
    ideal"b8+dc4-c,
          c8+ab4-b,
	  64b7c7-16b3c3da+4c3d+4b3a-1"
    ))

jason210 = method(Options => {CoefficientRing => ZZ/32003, MonomialOrder => GRevLex})
installMethod(jason210, Ideal => opts -> () -> (
    R := (opts.CoefficientRing)[vars(0..7), MonomialOrder => opts.MonomialOrder];
    ideal "a6,
           b6,
           a2c4+b2d4+abc2e2+abd2f2+abcdeg+abcdfh"
    ))

kotsireas = method(Options => {CoefficientRing => ZZ/32003, MonomialOrder => GRevLex})
installMethod(kotsireas, Ideal => opts -> () -> (
    R := (opts.CoefficientRing)[vars(0..6), MonomialOrder => opts.MonomialOrder];
    ideal "ba-bd-ae+ed-2f+2,
           ba+bd-2bf-ae-2a-ed+2ef+2d,
           b2-2be-2b+e2-2e+g+1,
           b3a2-1,
           e3d2-1,
           g3f2-1"
    ))

lichtblau = method(Options => {CoefficientRing => ZZ/32003, MonomialOrder => GRevLex})
installMethod(lichtblau, Ideal => opts -> () -> (
    R := (opts.CoefficientRing)[vars(0..2), MonomialOrder => opts.MonomialOrder];
    ideal "b-110a2+495a3-1320a4+2772a5-5082a6+7590a7-8085a8+5555a9-2189a10+374a11,
           c-22a+110a2-330a3+1848a5-3696a6+3300a7-1650a8+550a9-88a10-22a11"
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

-------------------------------------------------------------------------------
--- random ideals
-------------------------------------------------------------------------------
randomChoice = method()
randomChoice(ZZ, List) := ZZ => (n, P) -> (
    -- n = a positive integer
    -- P = a list of n probabilities
    -- return an integer 0..n-1 chosen with given probabilities

    r := random 1.0;
    position(accumulate(plus, 0, P), x -> x > r)
    )

randomNonzero = method()
randomNonzero(Ring) := RingElement => R -> (
    -- R = a ring
    -- return a random nonzero element of R

    x := random R;
    for i from 1 to 99 do (
        if x != 0 then return x;
	x = random R;
        );
    error "failed to generate random nonzero element after 100 tries"
    )

randomMonomial = method()
randomMonomial(ZZ, PolynomialRing) := RingElement => (d, R) -> (
    -- d = the degree
    -- R = a polynomial ring
    -- return a random monomial of degree d in R

    -- cache the basis in case of repeated calls to randomMonomial
    if not R#?cache then R.cache = new CacheTable;
    if not R.cache#?(basis, d) then R.cache#(basis, d) = basis(d, R);

    m := R.cache#(basis, d);
    m_(0, random rank source m)
    )

randomBinomial = method(Options => {Homogeneous => false, Pure => false})
randomBinomial(ZZ, ZZ, PolynomialRing) := RingElement => opts -> (d1, d2, R) -> (
    -- d1 = the degree of one monomial
    -- d2 = the degree of the other monomial
    -- R = a polynomial ring
    -- return a random binomial with given degrees from R

    c := if opts.Pure then -1_R else randomNonzero coefficientRing R;
    m1 := randomMonomial(d1, R);
    m2 := randomMonomial(d2, R);
    for i from 1 to 99 do (
        if m1 > m2 then
    	    return m1 + c * m2
        else if m1 < m2 then
    	    return m2 + c * m1;
	m1 = randomMonomial(d1, R);
        m2 = randomMonomial(d2, R);
	);
    error "failed to generate two distinct random monomials after 100 tries"
    )
randomBinomial(List, PolynomialRing) := RingElement => opts -> (D, R) -> (
    -- D = a list of probabilites for each degree
    -- R = a polynomial ring
    -- return a random binomial with degrees chosen with given probability

    if opts.Homogeneous then (
	d := randomChoice(length D, D);
	randomBinomial(d, d, R, opts)
    ) else (
    	d1 := randomChoice(length D, D);
	d2 := randomChoice(length D, D);
	randomBinomial(d1, d2, R, opts)
    )
    )

degreeDistribution = method(Options => {Constants => false, Degrees => "Uniform"})
degreeDistribution(PolynomialRing, ZZ) := List => opts -> (R, d) -> (
    -- R = a polynomial ring
    -- d = the maximal degree
    -- return a probability distribution on degrees less than or equal to d

    p := if opts.Constants then 1 else 0;
    D := if opts.Degrees === "Uniform" then (
             {p} | toList(d:1)
         ) else if opts.Degrees === "Weighted" then (
	     n := numgens R;
             {p} | for i from 1 to d list binomial(n + i - 1, n - 1)
         ) else if opts.Degrees === "Maximum" then (
             toList(d:0) | {1}
         );
    D / (1.0 * sum D)
    )

randomBinomialIdeal = method(Options => {
	CoefficientRing => ZZ/32003,
	MonomialOrder => GRevLex,
	Constants => false,
	Degrees => "Uniform",
	Homogeneous => false,
	Pure => false
	})
randomBinomialIdeal(ZZ, ZZ, ZZ) := Ideal => opts -> (n, d, s) -> (
    -- n = number of variables
    -- d = max degree of monomials
    -- s = number of generators of ideal
    -- return a random binomial ideal

    R := (opts.CoefficientRing)[vars(0..n-1), MonomialOrder => opts.MonomialOrder];
    D := degreeDistribution(R, d, Constants => opts.Constants, Degrees => opts.Degrees);
    randomBinomialIdeal(R, d, s, opts)
    )
randomBinomialIdeal(PolynomialRing, ZZ, ZZ) := Ideal => opts -> (R, d, s) -> (
    -- R = a polynomial ring
    -- d = max degree of monomials
    -- s = number of generators of ideal
    -- return a random binomial ideal

    D := degreeDistribution(R, d, Constants => opts.Constants, Degrees => opts.Degrees);
    randomBinomialIdeal(R, D, s, opts)
    )
randomBinomialIdeal(ZZ, List, ZZ) := Ideal => opts -> (n, D, s) -> (
    -- n = number of variables
    -- D = probabilites for each degree
    -- s = number of generators of ideal
    -- return a random binomial ideal

    R := (opts.CoefficientRing)[vars(0..n-1), MonomialOrder => opts.MonomialOrder];
    randomBinomialIdeal(R, D, s, opts)
    )
randomBinomialIdeal(PolynomialRing, List, ZZ) := Ideal => opts -> (R, D, s) -> (
    -- R = a polynomial ring
    -- D = probabilites for each degree
    -- s = number of generators of ideal
    -- return a random binomial ideal

    ideal for i to s-1 list
              randomBinomial(D, R, Homogeneous => opts.Homogeneous, Pure => opts.Pure)
    )

beginDocumentation()

doc ///
Key
  Ideals
Headline
  A collection of example ideals
Description
  Text
    This package contains a collection of ideals that are useful for testing or benchmarking
    Groebner basis algorithms.
    
    There are ideals from several parametric families,
  Example
    cyclic 7
  Text
    some fixed ideals,
  Example
    virasoro()
  Text
    and random ideals.
    
    Ideals are generated in new rings (except for @TO randomBinomialIdeal@, which can take in
    a ring).
  Example
    setRandomSeed(12345);
    randomBinomialIdeal(5, 3, 10)
///

doc ///
Key
  cyclic
  (cyclic, ZZ)
Headline
  return the cyclic-n ideal
Usage
  I = cyclic(n)
Inputs
  n: ZZ
     the number of variables
Outputs
  I: Ideal
     the cyclic-n ideal
Description
  Text
    The cyclic-n ideal is a classical benchmarking problem.
  Example
    I = cyclic 5
///

doc ///
Key
  randomBinomialIdeal
  (randomBinomialIdeal, ZZ, ZZ, ZZ)
  (randomBinomialIdeal, PolynomialRing, ZZ, ZZ)
  (randomBinomialIdeal, ZZ, List, ZZ)
  (randomBinomialIdeal, PolynomialRing, List, ZZ)
Headline
  return a random binomial ideal
Usage
  I = randomBinomialIdeal(n, d, s)
  I = randomBinomialIdeal(R, d, s)
  I = randomBinomialIdeal(n, D, s)
  I = randomBinomialIdeal(R, D, s)
Inputs
  n: ZZ
     the number of variables
  R: PolynomialRing
     the polynomial ring containing the ideal
  d: ZZ
     the maximum degree of the monomials
  D: List
     the probabilities for each degree
  s: ZZ
     the number of generators of the ideal
Outputs
  I: Ideal
     a random binomial ideal
Description
  Text
    There are several ways to define a random binomial ideal. Here we choose
    binomials at random by choosing a pair of unequal random monomials.
    
    First we need a polynomial ring. If R is given then it is used. Otherwise,
    a new polynomial ring is constructed with n variables over the given
    coefficient ring and with the given monomial order.
    
    Next we need a distribution on degrees. If D is given then it is used.
    Otherwise, a distribution is constructed on 0..d using the values of
    Constants and Degrees.
    
    Now a list of s binomials is generated by first randomly selecting a degree
    and then randomly picking monomials of that degree.
  Example
    setRandomSeed(12345);
    I = randomBinomialIdeal(5, 3, 10)
///

doc ///
Key
  [randomBinomialIdeal, Homogeneous]
Headline
  if the returned ideal should be homogeneous
Usage
  I = randomBinomialIdeal(n, d, s, Homogeneous => true)
Description
  Text
    Choose whether the returned ideal should be homogeneous.
  Example
    I = randomBinomialIdeal(5, 3, 10, Homogeneous => true)
SeeAlso
  randomBinomialIdeal
///

doc ///
Key
  [randomBinomialIdeal, Pure]
Headline
  if the returned ideal should be pure
Usage
  I = randomBinomialIdeal(n, d, s, Pure => true)
Description
  Text
    Choose whether the returned ideal should be pure.
  Example
    I = randomBinomialIdeal(5, 3, 10, Pure => true)
SeeAlso
  randomBinomialIdeal
///

TEST ///
I = cyclic 3
R = ring I
assert(I == ideal(R_0 + R_1 + R_2, R_0*R_1 + R_1*R_2 + R_2*R_0, R_0*R_1*R_2 - 1))
///

TEST ///
I = hcyclic 3
R = ring I
assert(I == ideal(R_0 + R_1 + R_2, R_0*R_1 + R_1*R_2 + R_2*R_0, R_0*R_1*R_2 - R_3^3))
///

TEST ///
debug needsPackage "Ideals"
R = ZZ/32003[x,y,z]
assert(degreeDistribution(R, 1) == {0, 1})
assert(degreeDistribution(R, 1, Constants => true) == {0.5, 0.5})
assert(degreeDistribution(R, 1, Constants => true, Degrees => "Weighted") == {0.25, 0.75})
assert(degreeDistribution(R, 5) == {0, .2, .2, .2, .2, .2})
assert(degreeDistribution(R, 5, Constants => true) == {1, 1, 1, 1, 1, 1} / 6.0)
assert(degreeDistribution(R, 5, Constants => true, Degrees => "Weighted") ==  {1, 3, 6, 10, 15, 21} / 56.0)
///

end
