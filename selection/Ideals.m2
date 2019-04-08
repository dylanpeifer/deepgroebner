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

extcyc = method(Options => {CoefficientRing => ZZ/32003, MonomialOrder => GRevLex})
extcyc ZZ := Ideal => opts -> n -> (
    R := (opts.CoefficientRing)[vars(0..n), MonomialOrder => opts.MonomialOrder];
    F := toList apply(1..n-1, d -> R_0^d + sum(0..n-1, i -> product(d, k -> R_((i+k)%n+1))))
         | {product(1..n, i -> R_i) - 1, R_0^n + 1};
    ideal F
    )

redcyc = method(Options => {CoefficientRing => ZZ/32003, MonomialOrder => GRevLex})
redcyc ZZ := Ideal => opts -> n -> (
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

randomBinomialIdeal = method(Options => {CoefficientRing => ZZ/32003, MonomialOrder => GRevLex})
randomBinomialIdeal(ZZ, ZZ, ZZ) := Ideal => opts -> (d, n, k) -> (
    R := (opts.CoefficientRing)[vars(0..n-1), MonomialOrder => opts.MonomialOrder];
    F := for j to k-1 list (
	d1 := random(1, d);
	exps1 := compositions(n, d1);
	e1 := exps1#(random(#exps1));
	m1 := product(n, i -> R_i^(e1#i));
	d2 := random(1, d);
	exps2 := compositions(n, d2);
	e2 := exps2#(random(#exps2));
	while e2 == e1 do (
	    e2 = exps2#(random(#exps2));
	    );
	m2 := product(n, i -> R_i^(e2#i));
	(random coefficientRing R) * m1 + (random coefficientRing R) * m2
	);
    ideal F
    )