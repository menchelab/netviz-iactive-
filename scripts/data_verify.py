"""
Data Verification Script for Gene Interaction Data

This script analyzes tab-separated (TSV) files containing gene interaction data.
It identifies and reports cases where the same gene pair (V1-V2) appears multiple times,
considering pairs bidirectionally (V1-V2 same as V2-V1).

For each gene pair that appears multiple times, it:
- Counts total occurrences across all layers
- Sums values from all additional columns in the dataset (Layers)
- Shows only columns that have non-zero values
- Displays results sorted by total count, limited to top 20 pairs

Usage:
    uv run python data_verify.py <tsv_file1> [tsv_file2 ...]

    Example:
        uv run python scriptsdata_verify.py Multiplex_DataDiVR/Multiplex_Net_Files/*Network.tsv

Input file format:
    - Tab-separated (TSV) file
    - Must contain 'V1' and 'V2' columns for gene pairs
    - Additional columns are treated as interaction layers

Output:
    - "OK" for files with no duplicate pairs
    - For files with duplicates:
        * Lists gene pairs with multiple occurrences
        * Shows count and layer-specific sums
"""

import pandas as pd
from collections import Counter
import sys

def analyze_file(filepath):
    try:
        df = pd.read_csv(filepath, sep="\t")
        # Get all columns except V1 and V2
        other_columns = [col for col in df.columns if col not in ["V1", "V2"]]

        # Create pairs and track indices for each pair, considering both directions
        v1_v2_pairs = []
        pair_indices = {}
        for idx, (v1, v2) in enumerate(zip(df['V1'], df['V2'])):
            # Store both directions as a sorted tuple to treat them as the same pair
            sorted_pair = tuple(sorted([v1, v2]))
            v1_v2_pairs.append(sorted_pair)
            if sorted_pair not in pair_indices:
                pair_indices[sorted_pair] = []
            pair_indices[sorted_pair].append(idx)

        pair_counts = Counter(v1_v2_pairs)
        filtered_counts = {
            pair: count for pair, count in pair_counts.items() if count > 1
        }
        top_pairs = sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)[
            :20
        ]
        if not top_pairs:
            print(
                "\033[92mOK: \033[1m{}\033[0m".format(filepath)
            )
            return

        # Calculate sums and find columns with non-zero values
        pair_sums = {}
        columns_with_values = set()
        max_v1_len = max(len(v1) for (v1, v2) in v1_v2_pairs)
        max_v2_len = max(len(v2) for (v1, v2) in v1_v2_pairs)

        for (v1, v2), _ in top_pairs:
            row_indices = pair_indices[(v1, v2)]
            sums = {}
            total_sum = 0
            for col in other_columns:
                # Sum values for both directions
                col_sum = int(df.iloc[row_indices][col].sum())
                if col_sum > 0:
                    columns_with_values.add(col)
                sums[col] = col_sum
                total_sum += col_sum
            pair_sums[(v1, v2)] = {"sums": sums, "total": total_sum}

        # Filter to only columns with non-zero values
        active_columns = sorted(list(columns_with_values))

        # Calculate column widths
        col_widths = {}
        for col in active_columns:
            col_name = col.replace(".tsv", "")
            max_sum = max(pair_sums[(v1, v2)]["sums"][col] for (v1, v2), _ in top_pairs)
            col_widths[col] = (
                max(len(col_name), len(str(max_sum))) + 2
            )

        print(
            f"\n\033[91mWARNING: multiple definitions of the same interaction-set found in the input file.\033[0m\n"
            f"\033[91mThis will result in additional duplicated intralayer edges!!\n"
            f"\n\033[1;91m{filepath}\033[0m\n"
            f"\033[91mList of affected combinations and layers\n(duplication, count per layer where layer active):\033[0m\n"
        )

        total_width = 3 + max_v1_len + max_v2_len + 8 + 8 + sum(col_widths.values())
        print("-" * total_width)

        header = f"{'':<3}{'V1':<{max_v1_len}} {'V2':<{max_v2_len}} {'count':>6} {'duplic':>6}"
        for col in active_columns:
            col_name = col.replace('.tsv', '')
            header += f"{col_name:>{col_widths[col]}}"
        print(header.rstrip())
        print("-" * total_width)

        for (v1, v2), count in top_pairs:
            data = pair_sums[(v1, v2)]
            total = data["total"]
            # Always show smaller name first for consistency
            first, second = sorted([v1, v2])
            line = f"{'':<3}{first:<{max_v1_len}} {second:<{max_v2_len}} "
            line += f"\033[93m{count:>6}\033[0m"
            line += f"\033[91m{total:>6}\033[0m"

            for col in active_columns:
                sum_val = data["sums"][col]
                if sum_val > 0:
                    line += f"\033[91m{sum_val:>{col_widths[col]}}\033[0m"
                else:
                    line += f"{sum_val:>{col_widths[col]}}" 
            print(line.rstrip())

        print("-" * total_width)

    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python data_verify.py <tsv_file1> [tsv_file2 ...]")
        sys.exit(1)

    for filepath in sys.argv[1:]:
        analyze_file(filepath)


if __name__ == "__main__":
    main()
