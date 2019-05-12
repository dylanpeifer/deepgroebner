for i in 0 1 2 3
do
	for j in 1 2 3
	do
		pyinstrument -o results-$i-$j.html -r html bench_$i.py
	done
done
