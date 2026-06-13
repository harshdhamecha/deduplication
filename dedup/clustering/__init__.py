"""Stage 4 clustering (filled in Step 5).

Planned contents:
  union_find.py  Union-Find (disjoint-set) with path compression + union by rank.
  cluster.py     Build a similarity graph from the pairs flagged by Stages 2 & 3,
                 then take connected components. WHY cluster not pair: duplicates
                 are transitive (A~B, B~C => {A,B,C} is one cluster) and pairwise
                 deletion double-counts and can delete the wrong survivor.
"""
