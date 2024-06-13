import random


with open("heuristics_evaluation_set.txt") as infile:
    header = infile.readline()
    rows = infile.readlines()
    random.shuffle(rows)
    assert header not in rows

with open("heuristics_evaluation_set_shuffle.txt", "w") as outfile:
    outfile.write(header)
    for r in rows:
        outfile.write(r)

with open("heuristics_evaluation_set_shuffle.txt") as checkfile:
    header_new = checkfile.readline()
    rows_new = checkfile.readlines()
    assert header_new == header
    assert sorted(rows_new) == sorted(rows)

print("Done")
