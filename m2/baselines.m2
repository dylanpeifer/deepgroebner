needsPackage "SelectionStrategies"
needsPackage "Ideals"

n := value(scriptCommandLine#1);
d := value(scriptCommandLine#2);
s := value(scriptCommandLine#3);

R = ZZ/32003[vars(0..n)];

F = openOutAppend(n|"-"|d|"-"|s|".csv")

F << "Random,First,Degree,Normal,Sugar,Size" << endl;

for i from 1 to 10000 do (
    I = randomBinomialIdeal(R, d, s);
    
    for sel in {"Random", "First", "Degree", "Normal", "Sugar"} do (
    	(i, j, G) = buchberger(I,
		               SelectionStrategy => sel,
                               Minimalize => false,
                               Interreduce => false);
	F << i+j << ",";
    );
    F << length first entries gens gb I << endl;
);

close F;

