ncu -f -o ncu-report -k "regex:(hnet)|(rz)" --section MemoryWorkloadAnalysis_Tables --section SourceCounters python benchmark.py -tf32 -l a100_10240.config -op sgd -i 1
