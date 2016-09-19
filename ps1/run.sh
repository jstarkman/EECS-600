#! /bin/bash

start=$(date +%s.%N)

out_dir=$(date)
mkdir "$out_dir"

for ms in 2000 4000 # 8000 16000
do
	for bs in 100 1000 # 5000
	do
		for h1 in 64 128
		do
			for h2 in 16 32
			do
				echo "$(date): Doing $ms $bs $h1 $h2"
				python fully_connected_feed.py $ms $bs $h1 $h2 > "$out_dir/out_$ms-$bs-$h1-$h2.log"
				echo "\tReturned: $?"
			done
		done
	done
done

end=$(date +%s.%N)

dt=$(echo "$end - $start" | bc)
dd=$(echo "$dt/86400" | bc)
dt2=$(echo "$dt-86400*$dd" | bc)
dh=$(echo "$dt2/3600" | bc)
dt3=$(echo "$dt2-3600*$dh" | bc)
dm=$(echo "$dt3/60" | bc)
ds=$(echo "$dt3-60*$dm" | bc)

printf "Total runtime: %d:%02d:%02d:%02.4f\n" $dd $dh $dm $ds

