import sys
import pandas as pd

def move_column_right(filename, col_num):
    # Read TSV file
    df = pd.read_csv(filename, sep='\t')
    
    # Ensure column index is valid
    if col_num < 0 or col_num >= len(df.columns) - 1:
        print("Invalid column number.")
        return
    
    # Swap columns
    cols = df.columns.tolist()
    cols[col_num], cols[col_num + 1] = cols[col_num + 1], cols[col_num]
    df = df[cols]
    
    # Save the modified file to the same filename
    df.to_csv(filename, sep='\t', index=False)
    print(f"File saved as {filename}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <filename> <column_number>")
    else:
        filename = sys.argv[1]
        col_num = int(sys.argv[2])
        move_column_right(filename, col_num)

