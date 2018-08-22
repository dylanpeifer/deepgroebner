newPackage(
        "SelectionStrategies",
        Version => "0.1", 
        Date => "March 15, 2018",
        Authors => {{Name => "Dylan Peifer", 
                     Email => "djp282@cornell.edu", 
                     HomePage => "https://www.math.cornell.edu/~djp282"}},
        Headline => "Test pair selection strategies in Buchberger's algorithm",
        DebuggingMode => true
        )

export {"buchbergerPairsProcessed",
	"buchbergerPairsPerStrategy",
	"buchbergerPairsMinimum"}

needsPackage "Reduction";

-------------------------------------------------------------------------------
--- utility functions
-------------------------------------------------------------------------------
spoly = method();
spoly(RingElement, RingElement) := RingElement => (f, g) -> (
    -- f = a polynomial
    -- g = a polynomial
    -- returns s-polynomial of f, g

    l := lcm(leadMonomial f, leadMonomial g);
    (l//(leadTerm f))*f - (l//(leadTerm g))*g
    )
spoly(List, List) := List => (f, g) -> (
    -- f = a {polynomial, sugardegree} pair
    -- g = a {polynomial, sugardegree} pair
    -- returns s-polynomial of f, g with sugar degree

    fdeg := f#1;
    gdeg := g#1;
    f = f#0;
    g = g#0;

    l := lcm(leadMonomial f, leadMonomial g);
    s := (l//(leadTerm f))*f - (l//(leadTerm g))*g;

    {s, max((first degree (l/(leadMonomial f))) + fdeg,
	    (first degree (l/(leadMonomial g))) + gdeg)}
    )

lcmCriterion = method()
lcmCriterion(List, List) := Boolean => (p, F) -> (
    -- p = a pair
    -- F = the list of polynomials
    -- returns if the pair satisfies the lcm criterion (criteria 2 of buchberger)

    gcd(leadMonomial (F#(p#0)), leadMonomial (F#(p#1))) == 1
    )

chainCriterion = method()
chainCriterion(List, List, List) := Boolean => (p, P, F) -> (
    -- p = a pair
    -- P = the set of pairs (without p)
    -- F = the list of polynomials
    
    l := lcm(leadMonomial (F#(p#0)), leadMonomial (F#(p#1)));
    for k from 0 to #F-1 when k != p#0 and k != p#1 do (
	if l % leadMonomial (F#k) == 0 then (
	    if not member(rsort({k, p#0}), P) and not member(rsort({k, p#1}), P) then
	    	return true;
	    );
	);
    false
    )

updatePairs = method()
updatePairs(List, List, RingElement) := List => (P, F, f) -> (
    -- P = a list of pairs in Buchberger's algorithm
    -- F = the corresponding list of polynomials
    -- f = a polynomial
    -- returns a sequence (P, F) containing the the new list of pairs P and new polynomials
    --     F obtained after adding f

    F = append(F, f);
    newPairs := (for i from 0 to #F-2 list {#F-1, i});

    (P | newPairs, F)
    )
updatePairs(List, List, List) := List => (P, F, f) -> (
    -- P = a list of pairs in Buchberger's algorithm
    -- F = the corresponding list of polynomials in {polynomial, sugardegree} form
    -- f = a {polynomial, sugardegree} pair
    -- returns a sequence (P, F) containing the new list of pairs P and new polynomials
    --     F obtained after adding f

    Fdeg := apply(F, x -> x#1);
    F = apply(F, x -> x#0);
    fdeg := f#1;
    f = f#0;

    (P, F) = updatePairs(P, F, f);
    (P, apply(F, append(Fdeg, fdeg), (x, y) -> {x, y}))
    )

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
--- pair selection strategies
-------------------------------------------------------------------------------
selectPair = method(Options => {Strategy => "First"})
selectPair(List, List) := List => opts -> (P, F) -> (
    -- P = a list of pairs in Buchberger's algorithm
    -- F = the corresponding list of polynomials
    -- returns the next pair to process

    i := 0;

    if opts.Strategy === "First" then
    	first P
    else if opts.Strategy === "Last" then
    	last P
    else if opts.Strategy === "Random" then
    	P#(random(#P))
    else if opts.Strategy === "Normal" then (
        i = argmin(P, p -> lcm(leadMonomial (F#(p#0)), leadMonomial (F#(p#1))));
	P#i
	)
    else if opts.Strategy === "DegreeNormal" then (
	i = argmin(P, p -> (
		mon := lcm(leadMonomial (F#(p#0)), leadMonomial (F#(p#1)));
		{first degree mon, mon}));
	P#i
	)
    else if opts.Strategy === "Strange" then (
        i = argmax(P, p -> lcm(leadMonomial (F#(p#0)), leadMonomial (F#(p#1))));
	P#i
	)
    else if opts.Strategy === "DegreeStrange" then (
	i = argmax(P, p -> (
		mon := lcm(leadMonomial (F#(p#0)), leadMonomial (F#(p#1)));
		{first degree mon, mon}));
	P#i
    	)	
    else if opts.Strategy === "Sugar" then (
    	i = argmin(P, p -> (
		s := spoly(F#(p#0), F#(p#1));
		{s#1, lcm(leadMonomial ((F#(p#0))#0), leadMonomial ((F#(p#1))#0))}));
	P#i		
	)
    else
    	error "invalid strategy"
    )

buchbergerPairsProcessed = method(Options => {Strategy => "First"})
buchbergerPairsProcessed(Ideal) := ZZ => opts -> (I) -> (
    -- I = an ideal in a polynomial ring
    -- returns number of pairs processed in computing a Groebner basis of I

    -- initialize pairs P and polynomials F
    P := {};
    F := {};
    for f in first entries gens I do (
	(P, F) = updatePairs(P, F, f);
	);
    if opts.Strategy === "Sugar" then
    	F = apply(F, x -> {x, first degree x});

    -- process pairs until none remain
    pairsProcessed := 0;
    while #P > 0 do (
	p := selectPair(P, F, Strategy => opts.Strategy);
	P = delete(p, P);

    	-- eliminate pair if possible
	if opts.Strategy === "Sugar" then (
	    if (lcmCriterion(p, apply(F, x -> x#0)) or 
		chainCriterion(p, P, apply(F, x -> x#0))) then
	        continue;
	    )
	else (
	    if lcmCriterion(p, F) or chainCriterion(p, P, F) then
	    	continue;
	    );

	s := spoly(F#(p#0), F#(p#1));
    	r := 0;

    	-- divAlg can't handle sugar, so switch division methods here
    	if opts.Strategy === "Sugar" then (
	    r = divide(s, F);
	    if r#0 != 0 then (
		(P, F) = updatePairs(P, F, r);
	    	);
	    )
	else (
	    r = first first entries divAlg(matrix {{s}}, matrix {F});
	    if r != 0 then (
	    	(P, F) = updatePairs(P, F, r);
	    	);
	    );

	pairsProcessed = pairsProcessed + 1;
	--if pairsProcessed >= 1000 then break;
	);

    -- for debug purposes, uncomment to get the Groebner basis computed
    -- if opts.Strategy === "Sugar" then
        -- F = apply(F, x -> x#0);
    -- interreduce(minimalize(F))

    pairsProcessed
    -- {#F, max(apply(F, f -> first degree f)), max(apply(F, f -> length terms f))}
    )

buchbergerPairsPerStrategy = method()
buchbergerPairsPerStrategy(Ideal) := List => (I) -> (
    -- I = an ideal in a polynomial ring
    -- returns the number of pair reductions needed for each strategy

    {{"First", buchbergerPairsProcessed(I, Strategy => "First")},
     {"Last", buchbergerPairsProcessed(I, Strategy => "Last")},
     {"Random", buchbergerPairsProcessed(I, Strategy => "Random")},
     {"Normal", buchbergerPairsProcessed(I, Strategy => "Normal")},
     {"DegreeNormal", buchbergerPairsProcessed(I, Strategy => "DegreeNormal")},
     {"Strange", buchbergerPairsProcessed(I, Strategy => "Strange")},
     {"DegreeStrange", buchbergerPairsProcessed(I, Strategy => "DegreeStrange")},
     {"Sugar", buchbergerPairsProcessed(I, Strategy => "Sugar")}}
    )

-------------------------------------------------------------------------------
--- finding minimal possible value of pairsProcessed
-------------------------------------------------------------------------------
Node := new Type of MutableHashTable
protect symbol Polynomials;
protect symbol Pairs;
protect symbol Remainders;
protect symbol PairsProcessed;
-- a node in a Buchberger tree
-- has keys:
--     Polynomials = a List of RingElements, the current polynomials in the basis
--     Pairs = a List of Lists, the current pairs left to process
--     Remainders = a List of remainders for the pairs
--     PairsProcessed = a ZZ, the total number of pairs reduced so far

buchbergerPairsMinimum = method(Options => {"Bound" => infinity, "Width" => infinity})
buchbergerPairsMinimum(Ideal) := Node => opts -> (I) -> (
    -- I = an ideal in a polynomial ring
    -- Bound = integer giving max pairs processed before stopping
    -- Width = integer giving the beam width for the tree search
    -- returns estimate for minimum value of pairsProcessed in Buchberger's algorithm

    -- initialize pairs P and polynomials F
    P := {};
    F := {};
    for f in first entries gens I do (
	(P, F) = updatePairs(P, F, f);
	);

    -- make the root of the tree
    root := new Node;
    root.Polynomials = F;
    root.Pairs = P;
    root.Remainders = {};
    root.PairsProcessed = 0;

    -- initialize lists for current level of tree and next level
    currentLevel := {root};
    nextLevel := {};

    -- go down levels until some path terminates and break is reached
    while #currentLevel > 0 do (

    	<< #currentLevel << endl;

    	-- create all remainders for each node, saving the nonzero pairs in node.Pairs
    	for node in currentLevel do (
	    for p in node.Pairs do (
	    	if (lcmCriterion(p, node.Polynomials) or
		    chainCriterion(p, node.Pairs, node.Polynomials)) then (
		    node.Pairs = delete(p, node.Pairs);
		    continue;
		    );

	    	s := spoly((node.Polynomials)#(p#0), (node.Polynomials)#(p#1));
	        r := first first entries divAlg(matrix {{s}}, matrix {node.Polynomials});
		if r == 0 then (
		    node.Pairs = delete(p, node.Pairs);
		    node.PairsProcessed = node.PairsProcessed + 1;
		    )
	        else (
		    node.Remainders = append(node.Remainders, r);
		    );
	        );
	    );

    	-- find the Width best nodes in the current level by best value of PairsProcessed
    	-- ignore nodes that have exceeded Bound pairs processed
	currentBest := for i from 0 to min((opts#"Width")-1, #currentLevel-1) list (
	    j := argmin(currentLevel, n -> n.PairsProcessed);
	    n := currentLevel#j;
	    currentLevel = drop(currentLevel, {j, j});
	    if n.PairsProcessed < opts#"Bound" then n else continue
	    );

	-- break if we finished a path
    	for node in currentBest do (
	    if #(node.Pairs) == 0 then
	    	return node.PairsProcessed;
	    );

    	-- make the next level and prepare for next iteration
        for node in currentBest do (
	    for i from 0 to #(node.Pairs)-1 do (
		p := (node.Pairs)#i;
		r := (node.Remainders)#i;
		child := new Node;
		(P, F) = updatePairs(delete(p, node.Pairs), node.Polynomials, r);
		child.Polynomials = F;
		child.Pairs = P;
		child.Remainders = {};
		child.PairsProcessed = node.PairsProcessed + 1;
		nextLevel = prepend(child, nextLevel);
		);
	    );
	currentLevel = nextLevel;
	nextLevel = {};
	);

    -- search finished before any path terminated
    null
    )

beginDocumentation()

end--

doc ///
Key
  SelectionStrategies
Headline
Description
  Text
  Example
Caveat
SeeAlso
///

doc ///
Key
Headline
Usage
Inputs
Outputs
Consequences
Description
  Text
  Example
  Code
  Pre
Caveat
SeeAlso
///

TEST ///
-- test code and assertions here
-- may have as many TEST sections as needed
///

needsPackage "SelectionStrategies"

-- random quadratics
-- First 10, Last 9958, Random 68, Normal 10, DegreeNormal 10, Strange 6643, DegreeStrange 6643, Sugar 10
-- min 10
R = ZZ/32003[a,b,c,d]
I = ideal(random(2, R), random(2, R), random(2, R))
buchbergerPairsPerStrategy(I)
buchbergerPairsMinimum(I)

-- question 1
-- First 2, Last 2, Random 2, Normal 2, DegreeNormal 2, Strange 2, DegreeStrange 2, Sugar 2
-- min 2
R = ZZ/32003[x,y,MonomialOrder=>Lex]
I = ideal(x^2 - y^3, x*y^2 + x)
buchbergerPairsPerStrategy(I)
buchbergerPairsMinimum(I)

-- example 12
-- First 9, Last 18, Random 14, Normal 9, DegreeNormal 9, Strange 18, DegreeStrange 18, Sugar 9
-- min 9
R = ZZ/32003[x,y,z]
I = ideal(x*y^2 + z, x*z + 3*y, x^2 + y*z)
buchbergerPairsPerStrategy(I)
buchbergerPairsMinimum(I)

-- small example
-- First 6, Last 64, Random 17, Normal 14, DegreeNormal 14, Strange 10, DegreeStrange 10, Sugar 14
-- min 6
R = ZZ/32003[x,y,z]
I = ideal(x^2*y + y, x^5 + x^2, x + y + z)
buchbergerPairsPerStrategy(I)
buchbergerPairsMinimum(I)

-- random quadratics (lex)
-- First 36, Random 1736, Normal 395, DegreeNormal 36, Sugar 36
R = ZZ/32003[a,b,c,d,MonomialOrder=>Lex]
I = ideal(random(2, R), random(2, R), random(2, R))
buchbergerPairsProcessed(I, Strategy => "First")
buchbergerPairsMinimum(I, "Bound" => 36)

-- random cubics
-- First 64, Random 4744, Normal 36, DegreeNormal 36, Sugar 36
R = ZZ/32003[a,b,c,d]
I = ideal(random(3, R), random(3, R), random(3, R))
buchbergerPairsProcessed(I, Strategy => "First")
buchbergerPairsMinimum(I, "Bound" => 36)

-- cyclic-5
-- First 249, Random >66960, Normal 252, DegreeNormal 252, Sugar 215
R = ZZ/32003[x,y,z,t,u]
I = ideal(x+y+z+t+u, x*y+y*z+z*t+t*u+u*x, x*y*z+y*z*t+z*t*u+t*u*x+u*x*y,
          x*y*z*t+y*z*t*u+z*t*u*x+t*u*x*y+u*x*y*z, x*y*z*t*u-1)
buchbergerPairsProcessed(I, Strategy => "First")
buchbergerPairsMinimum(I, "Bound" => 215)

-- katsura-4
-- First 146, Random 2931, Normal 52, DegreeNormal 52, Sugar 52
R = ZZ/32003[x,y,z,t,u]
I = ideal(2*x^2+2*y^2+2*z^2+2*t^2+u^2-u, x*y+2*y*z+2*z*t+2*t*u-t, 2*x*z+2*y*t+t^2+2*z*u-z,
          2*x*t+2*z*t+2*y*u-y, 2*x+2*y+2*z+2*t+u-1)
buchbergerPairsProcessed(I, Strategy => "First")

-- katsura-8
-- First 3366, Normal 1823, DegreeNormal 1823, Sugar 1823
R = ZZ/32003[a,b,c,d,e,f,g,h,i,j]
I = ideal(a^2+2*b^2+2*c^2+2*d^2+2*e^2+2*f^2+2*g^2+2*h^2+2*i^2-a*j,
          2*a*b+2*b*c+2*c*d+2*d*e+2*e*f+2*f*g+2*g*h+2*h*i-b*j,
          b^2+2*a*c+2*b*d+2*c*e+2*d*f+2*e*g+2*f*h+2*g*i-c*j,
          2*b*c+2*a*d+2*b*e+2*c*f+2*d*g+2*e*h+2*f*i-d*j,
          c^2+2*b*d+2*a*e+2*b*f+2*c*g+2*d*h+2*e*i-e*j,
          2*c*d+2*b*e+2*a*f+2*b*g+2*c*h+2*d*i-f*j,
          d^2+2*c*e+2*b*f+2*a*g+2*b*h+2*c*i-g*j,
          2*d*e+2*c*f+2*b*g+2*a*h+2*b*i-h*j,
          a+2*b+2*c+2*d+2*e+2*f+2*g+2*h+2*i-j)
buchbergerPairsProcessed(I, Strategy => "First")

-- cyclic-7
R = ZZ/32003[a,b,c,d,e,f,g]
I = ideal(a+b+c+d+e+f+g, a*b+b*c+c*d+d*e+e*f+a*g+f*g,
          a*b*c+b*c*d+c*d*e+d*e*f+a*b*g+a*f*g+e*f*g,
          a*b*c*d+b*c*d*e+c*d*e*f+a*b*c*g+a*b*f*g+a*e*f*g+d*e*f*g,
          a*b*c*d*e+b*c*d*e*f+a*b*c*d*g+a*b*c*f*g+a*b*e*f*g+a*d*e*f*g+c*d*e*f*g,
          a*b*c*d*e*f+a*b*c*d*e*g+a*b*c*d*f*g+a*b*c*e*f*g+a*b*d*e*f*g+a*c*d*e*f*g+b*c*d*e*f*g,
          a*b*c*d*e*f*g-h^7)
buchbergerPairsProcessed(I, Strategy => "First")

-- jason-210
R = ZZ/32003[a,b,c,d,e,f,g,h]
I = ideal(a^6, b^6, a^2*c^4+b^2*d^4+a*b*c^2*e^2+a*b*d^2*f^2+a*b*c*d*e*g+a*b*c*d*f*h)
buchbergerPairsProcessed(I, Strategy => "First")
