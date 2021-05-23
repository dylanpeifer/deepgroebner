newPackage(
        "Ideals",
        Version => "0.0.1",
        Date => "January 14, 2020",
        Authors => {{Name => "Dylan Peifer", 
                     Email => "djp282@cornell.edu", 
                     HomePage => "https://www.math.cornell.edu/~djp282"}},
        Headline => "A collection of example ideals",
        DebuggingMode => true
        )

export {"commutingMatrices", "cyclic", "eco", "katsura", "noon", "reimer",
        "chemkin", "haas", "jason210", "kotsireas", "lichtblau", "twistedCubic", "virasoro",
        "randomBinomialIdeal", "degreeDistribution", "Pure", "randomPolynomialIdeal",
	"randomToricIdeal"}

needsPackage "FourTiTwo"

-------------------------------------------------------------------------------
--- ideal families
-------------------------------------------------------------------------------
commutingMatrices = method(Options => {CoefficientRing => ZZ/32003, MonomialOrder => GRevLex})
commutingMatrices ZZ := Ideal => opts -> n -> (
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

twistedCubic = method(Options => {CoefficientRing => ZZ/32003, MonomialOrder => GRevLex})
installMethod(twistedCubic, Ideal => opts -> () -> (
    R := (opts.CoefficientRing)[vars(0..2), MonomialOrder => opts.MonomialOrder];
    ideal "b-a2,
           c-a3"
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

randomPoisson = method()
randomPoisson ZZ := ZZ =>
randomPoisson RR := ZZ => lambda -> (
    -- lambda = the parameter of a Poisson distribution
    -- return a Poisson random variable

    L := exp(-lambda);
    k := 1;
    p := random 1.0;
    while p > L do (
	k = k + 1;
	p = p * random 1.0;
	);
    k - 1
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

randomPolynomial = method()
randomPolynomial(List, ZZ, PolynomialRing) := RingElement =>
randomPolynomial(List, RR, PolynomialRing) := RingElement => (D, lambda, R) -> (
    -- D = a list of probabilities for each degree
    -- lambda = the parameter for a Poisson distribution on length
    -- R = a polynomial ring

    t := 2 + randomPoisson lambda;
    f := 0_R;
    for i to t - 1 do (
	d := randomChoice(length D, D);
	c := randomNonzero coefficientRing R;
	f = f + c * randomMonomial(d, R);
	);
    f
    )

degreeDistribution = method(Options => {Constants => false, Degrees => "Uniform"})
degreeDistribution(ZZ, ZZ) := List => opts -> (n, d) -> (
    -- n = the number of variables
    -- d = the maximal degree
    -- return a probability distribution on degrees less than or equal to d

    p := if opts.Constants then 1 else 0;
    D := if opts.Degrees === "Uniform" then (
             {p} | for i from 1 to d list binomial(n + i - 1, n - 1)
         ) else if opts.Degrees === "Weighted" then (
             {p} | toList(d:1)
         ) else if opts.Degrees === "Maximum" then (
             toList(d:0) | {1}
         );
    D / (1.0 * sum D)
    )
degreeDistribution(PolynomialRing, ZZ) := List => opts -> (R, d) -> (
    -- R = a polynomial ring
    -- d = the maximal degree
    -- return a probability distribution on degrees less than or equal to d

    degreeDistribution(numgens R, d, opts)
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

randomPolynomialIdeal = method(Options => {
	CoefficientRing => ZZ/32003,
	MonomialOrder => GRevLex,
	Constants => false,
	Degrees => "Uniform"
	})
randomPolynomialIdeal(ZZ, ZZ, ZZ, RR) := Ideal => opts -> (n, d, s, lambda) -> (
    -- n = a number of variables
    -- d = max degree of monomials
    -- s = number of generators of ideal
    -- lambda = parameter for the Poisson distribution on polynomial lengths
    -- return a random polynomial ideal

    R := (opts.CoefficientRing)[vars(0..n-1), MonomialOrder => opts.MonomialOrder];
    D := degreeDistribution(R, d, Constants => opts.Constants, Degrees => opts.Degrees);
    randomPolynomialIdeal(R, d, s, lambda, opts)    
    )
randomPolynomialIdeal(PolynomialRing, ZZ, ZZ, RR) := Ideal => opts -> (R, d, s, lambda) -> (
    -- R = a polynomial ring
    -- d = max degree of monomials
    -- s = number of generators of ideal
    -- lambda = parameter for the Poisson distribution on polynomial lengths
    -- return a random polynomial ideal

    D := degreeDistribution(R, d, Constants => opts.Constants, Degrees => opts.Degrees);
    randomPolynomialIdeal(R, D, s, lambda, opts)
    )
randomPolynomialIdeal(ZZ, List, ZZ, RR) := Ideal => opts -> (n, D, s, lambda) -> (
    -- n = number of variables
    -- D = probabilites for each degree
    -- s = number of generators of ideal
    -- lambda = parameter for the Poisson distribution on polynomial lengths
    -- return a random polynomial ideal

    R := (opts.CoefficientRing)[vars(0..n-1), MonomialOrder => opts.MonomialOrder];
    randomPolynomialIdeal(R, D, s, lambda, opts)
    )
randomPolynomialIdeal(PolynomialRing, List, ZZ, RR) := Ideal => opts -> (R, D, s, lambda) -> (
    -- R = a polynomial ring
    -- D = probabilites for each degree
    -- s = number of generators of ideal
    -- return a random binomial ideal

    ideal for i to s-1 list
              randomPolynomial(D, lambda, R)
    )

-------------------------------------------------------------------------------
--- random toric ideals

--- generated as ker A from a integer matrix A, which is a matrix of a random monomial map. 
--- several models are included:
---    - (ZZ,ZZ,ZZ) := (n,D,M) -> fixed number of (Laurent) monomials (so columns of matrix A),
---                               all with given L-1 norm, selected uniformly at random
---    - (ZZ,ZZ,ZZ,ZZ) := (n,L,U,M) -> fixed number of monomials with given positive degree & given genative degree,
---                                    selected uniformly at random (this allows for,e.g., putting neg. degree 0,
---                                    so only pos. exponents appear)
---    - (ZZ,ZZ,RR) := (n,D,p) -> E-R model (given parameter p), all with given L-1 norm
---    - (ZZ,ZZ,ZZ,RR) := (n,L,U,p) -> E-R model (given parameter p), with given positive degree & given negative degree
---    - (ZZ,ZZ,List) := (n,D,pOrM) and (ZZ,ZZ,ZZ,List) := (n,L,U,pOrM) -> graded model (not sure if complete yet?)
-------------------------------------------------------------------------------
allLaurentMonomials = method(TypicalValue=>List);
allLaurentMonomials(ZZ,ZZ) := (n,D) -> (
    -- input n 
    -- input D
    a := symbol a;
    x := symbol x;
    R:=ZZ/101[x_1..x_n,a_1..a_n];
    I:=ideal apply(toList(1..n),i-> a_i*x_i-1);
    F:=R/I;
    L:=QQ[x_1..x_n, MonomialOrder=>Lex,Inverses=>true];
    phi := map( L , F ,     matrix{join(toList(x_1..x_n), apply(toList(1..n),i->x_i^(-1)) ) } ); 
    B  :=  flatten flatten apply(toList(1..D),d->entries basis(d,F));
    apply(B,b->phi b)
)
allLaurentMonomials(ZZ,ZZ,ZZ) := (n,L,U) -> (
    -- input n
    -- input L<0
    -- input U>0
    a := symbol a;
    x := symbol x;
    R:=ZZ/101[x_1..x_n,a_1..a_n, Degrees=>join(toList(n:{1,0}), toList(n:{0,-1}))];
    I:=ideal apply(toList(1..n), i->a_i*x_i-1);
    F:=R/I;
    K:=QQ[x_1..x_n, MonomialOrder=>Lex,Inverses=>true];
    phi:= map(K,F, matrix{join(toList(x_1..x_n), apply(toList(1..n),i->x_i^(-1)))});
    B:= delete(sub(1,F), flatten flatten flatten apply(toList(0..U), i->apply(toList(L..0),j->entries basis({i,j},F))));
    apply(B, b->phi b)
)

randomLaurentMonomialSet = method();
randomLaurentMonomialSet(ZZ,ZZ,ZZ) := (n,D,M) -> (
    -- fixed M model with L1 norm monomial generationg model
    allMonomials := allLaurentMonomials(n,D);
    take(random(allMonomials),M)
)
randomLaurentMonomialSet(ZZ,ZZ,RR) := (n,D,p) -> (
    -- ER model with L1 norm monomial generating model
    allMonomials := allLaurentMonomials(n,D);
    select(allMonomials, m->random(0.0,1.0)<=p)
)
randomLaurentMonomialSet(ZZ,ZZ,ZZ,ZZ) := (n,L,U,M) -> (
    -- fixed M model with positive degree sum/negative degree sum monomial generating model
    allMonomials := allLaurentMonomials(n,L,U);
    take(random(allMonomials),M)
)
randomLaurentMonomialSet(ZZ,ZZ,ZZ,RR) := (n,L,U,p) -> (
    -- ER model with positive degree sum/negative degree sum monomial generating model
    allMonomials := allLaurentMonomials(n,L,U);
    select(allMonomials, m->random(0.0,1.0)<=p)
)
randomLaurentMonomialSet(ZZ,ZZ,List) := (n,D,pOrM) -> (
    -- start of graded model
    allMonomials := sort values partition(m-> first degree m, allLaurentMonomials(n,D));
    if all(pOrM,q->instance(q,ZZ)) then (
        flatten apply(toList(1..(2*D+1)), d->take(random(allMonomials_(d-1)), pOrM_(d-1)))
    )
    else if all(pOrM,q->instance(q,RR)) then (
        flatten apply(toList(1..(2*D+1)), d->select(allMonomials_(d-1),m->random(0.0,1.0)<=pOrM_(d-1)))
    )
)
randomLaurentMonomialSet(ZZ,ZZ,ZZ,List) := (n,L,U,pOrM) -> (
    -- start of graded model
    allMonomials := sort values partition(m-> first degree m, allLaurentMonomials(n,L,U));
    if all(pOrM,q->instance(q,ZZ)) then (
        flatten apply(toList(1..(U-L+1)), d->take(random(allMonomials_(d-1)), pOrM_(d-1)))
    )
    else if all(pOrM,q->instance(q,RR)) then (
        flatten apply(toList(1..(U-L+1)), d->select(allMonomials_(d-1),m->random(0.0,1.0)<=pOrM_(d-1)))
    )
)

randomToricIdeal = method(Options => {
	CoefficientRing => ZZ/32003,
	MonomialOrder => GRevLex,
	Constants => false, -- figure out what to do with this below? use/kill? 
	Degrees => "Uniform"
	})
randomToricIdeal(ZZ,ZZ,ZZ) := Ideal => opts -> (n,D,M) -> (
    -- n = a number of target variables = number of rows of A
    -- D = max degree of monomials 
    -- M = number of monomials desired  --- this will be the number of columns of A. 
    -- M = desired number of monomials = numer of source  variables = numcols A
    -- fixed M model with L1 norm monomial generationg model
    A := transpose matrix flatten apply(randomLaurentMonomialSet(n,D,M), m->exponents m);
    R := (opts.CoefficientRing)[vars(0..M-1), MonomialOrder => opts.MonomialOrder];
    toricMarkov (A,R) 
    )
randomToricIdeal(ZZ,ZZ,RR) := Ideal => opts -> (n,D,p) -> (
    -- n = a number of variables...for mons. TO DO
    -- D = max degree of monomials
    -- p = prob of selecting each monomial  (the E-R-type model)
    -- return a toric ideal
    -- ER model with L1 norm monomial generating model
    A := transpose matrix flatten apply(randomLaurentMonomialSet(n,D,p), m->exponents m);
    n = numcols A;
    R := (opts.CoefficientRing)[vars(0..n-1), MonomialOrder => opts.MonomialOrder];
    toricMarkov (A,R) 
    )
randomToricIdeal(ZZ,ZZ,ZZ,ZZ) := Ideal => opts -> (n,L,U,M) -> (
    -- n = a number of target variables = number of rows of A
    -- L = max neg degree of monomials
    -- U = max pos degree of monomials
    -- M = desired number of monomials = numer of source  variables = numcols A
    -- return a toric ideal in M variables 
    -- fixed M model with positive degree sum/negative degree sum monomial generating model
    A := transpose matrix flatten apply(randomLaurentMonomialSet(n,L,U,M), m->exponents m);
    R := (opts.CoefficientRing)[vars(0..M-1), MonomialOrder => opts.MonomialOrder];
    toricMarkov (A,R) 
    )
randomToricIdeal(ZZ,ZZ,ZZ,RR) := Ideal => opts -> (n,L,U,p) -> (
    -- n = a number of variables...for mons. TO DO
    -- L = max neg degree of monomials
    -- U = max pos degree of monomials
    -- p = probability of selecting each monomial (the E-R-type model)
    -- return a toric ideal
    -- ER model with positive degree sum/negative degree sum monomial generating model
    A := transpose matrix flatten apply(randomLaurentMonomialSet(n,L,U,p), m->exponents m);
    n = numcols A;
    R := (opts.CoefficientRing)[vars(0..n-1), MonomialOrder => opts.MonomialOrder];
    toricMarkov (A,R) 
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
  Example
    setRandomSeed(12345);
    randomBinomialIdeal(5, 3, 10)
  Text
    Ideals are generated in new rings (except for @TO randomBinomialIdeal@, which can take in
    a ring). The coefficient ring and monomial order can be changed with optional MonomialOrder
    and CoefficientRing arguments.
///

doc ///
Key
  commutingMatrices
  (commutingMatrices, ZZ)
Headline
  return the ideal given by commuting nxn matrices
Usage
  I = commutingMatrices(n)
Inputs
  n: ZZ
     the size of the matrices
Outputs
  I: Ideal
     the ideal given by commuting nxn matrices
Description
  Text
    Two generic nxn matrices will commute if the equations are satisfied.
  Example
    I = commutingMatrices 3
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
  eco
  (eco, ZZ)
Headline
  return the eco-n ideal
Usage
  I = eco(n)
Inputs
  n: ZZ
     the number of variables
Outputs
  I: Ideal
     the eco-n ideal
Description
  Text
    The eco-n ideal is a classical benchmarking problem.
  Example
    I = eco 5
///

doc ///
Key
  katsura
  (katsura, ZZ)
Headline
  return the katsura-n ideal
Usage
  I = katsura(n)
Inputs
  n: ZZ
     the number of variables
Outputs
  I: Ideal
     the katsura-n ideal
Description
  Text
    The katsura-n ideal is a classical benchmarking problem.
  Example
    I = katsura 5
///

doc ///
Key
  noon
  (noon, ZZ)
Headline
  return the noon-n ideal
Usage
  I = noon(n)
Inputs
  n: ZZ
     the number of variables
Outputs
  I: Ideal
     the noon-n ideal
Description
  Text
    The noon-n ideal is a classical benchmarking problem.
  Example
    I = noon 5
///

doc ///
Key
  reimer
  (reimer, ZZ)
Headline
  return the reimer-n ideal
Usage
  I = reimer(n)
Inputs
  n: ZZ
     the number of variables
Outputs
  I: Ideal
     the reimer-n ideal
Description
  Text
    The reimer-n ideal is a classical benchmarking problem.
  Example
    I = reimer 5
///

doc ///
Key
  degreeDistribution
  (degreeDistribution, ZZ, ZZ)
  (degreeDistribution, PolynomialRing, ZZ)
Headline
  return a distribution on degrees of monomials
Usage
  D = degreeDistribution(n, d)
  D = degreeDistribution(R, d)
Inputs
  n: ZZ
     the number of variables
  R: PolynomialRing
     the polynomial ring
  d: ZZ
     the maximal degree
Outputs
  D: List
     a list of probabilities
Description
  Text
    A distribution on degrees of monomials.
  Example
    D = degreeDistribution(3, 5)
SeeAlso
  randomBinomialIdeal
///

doc ///
Key
  [degreeDistribution, Degrees]
Headline
  the model for degree distribution
Usage
  D = degreeDistribution(n, d, Degrees => "Weighted")
Description
  Text
    Choose the particular model for degree distribution. Options are "Uniform",
    "Weighted", and "Maximum".
  Example
    D = degreeDistribution(3, 5, Degrees => "Weighted")
SeeAlso
  degreeDistribution
///

doc ///
Key
  [degreeDistribution, Constants]
Headline
  if the distribution should have nonzero value for constants
Usage
  D = degreeDistribution(n, d, Constants => true)
Description
  Text
    Choose whether the degree distribution should allow constants.
  Example
    D = degreeDistribution(3, 5, Constants => true)
    D = degreeDistribution(3, 5, Constants => false)
SeeAlso
  degreeDistribution
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
    coefficient ring and with the given monomial order. If generating many
    examples it is more efficient to provide R, as then intermediate calculations
    can be stored in it.
    
    Next we need a distribution on degrees. If D is given then it is used.
    Otherwise, a distribution is constructed on 0..d using the values of
    Constants and Degrees.
    
    Now a list of s binomials is generated by first randomly selecting a degree
    and then randomly picking monomials of that degree. Monomials are regenerated
    until two unequal monomials are generated, and coefficients are all nonzero.
  Example
    setRandomSeed(12345);
    I = randomBinomialIdeal(5, 3, 10)
SeeAlso
  degreeDistribution
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
R = ZZ/32003[x,y,z]
assert(degreeDistribution(R, 1) == {0, 1})
assert(degreeDistribution(R, 1, Constants => true, Degrees => "Weighted") == {0.5, 0.5})
assert(degreeDistribution(R, 1, Constants => true, Degrees => "Uniform") == {0.25, 0.75})
assert(degreeDistribution(R, 5, Constants => false, Degrees => "Weighted") == {0, .2, .2, .2, .2, .2})
assert(degreeDistribution(R, 5, Constants => true, Degrees => "Weighted") == {1, 1, 1, 1, 1, 1} / 6.0)
assert(degreeDistribution(R, 5, Constants => true, Degrees => "Uniform") ==  {1, 3, 6, 10, 15, 21} / 56.0)
///

end
