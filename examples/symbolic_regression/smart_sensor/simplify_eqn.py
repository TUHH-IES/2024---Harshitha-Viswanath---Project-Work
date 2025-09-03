import pandas as pd
from sympy import sympify, simplify
from collections import defaultdict

df = pd.read_csv("grouping_results.csv")

#dictionary to group equivalent equations
grouped = defaultdict(list)

for idx, row in df.iterrows():
    try:
        expr = simplify(sympify(row['equation']))
        key = str(expr)
        grouped[key].append((row["group_id"], row["loss"], row["equation"]))
    except:
        print("Failed to process equation at index {idx}: {e}")


print("Grouped equivalent equations: ")
for group_id, (expr_str, equations) in enumerate(grouped.items()):
    print(f"Group {group_id} (simplified form: {expr_str}):")
    for gid, loss, orig_eq in equations:
        print(f"  [Original Group ID: {gid}] Loss: {loss:.6f}, Equation: {orig_eq}")
    print("-" * 50)
