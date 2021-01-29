import sys
from itertools import combinations

ffile = sys.argv[1]
npops = int(sys.argv[2])
with open(ffile) as ff:
    lin = next(ff)  # header
    header = lin.split()
pop_dt = {}

for i, j in combinations(range(npops), 2):
    pop_dt[f"pop{i}{j}"] = [ix+1 for ix, h in enumerate(header) if f"pop{i}{j}" in h]

for p in range(npops):
    p_ix = []
    for ix, h in enumerate(header):
        hh = h.split("_")[-1]
        if (f"pop{p}" in hh) and (len(hh) == len(f"pop{p}")):
            p_ix.append(ix+1)
    pop_dt[f"pop{p}"] = p_ix

f = open("cut_cmmnds.sh", 'w')
for p in pop_dt.keys():
    cl = ",".join(map(str, pop_dt[p]))
    f.write(f"cut -f{cl} {ffile} > {ffile}-{p}\n")
