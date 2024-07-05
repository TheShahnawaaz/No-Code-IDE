from tabulate import tabulate

# Create the outer table
outer_table = [
    ["Name", "Age", "Details"],
    ["John", 30, "See inner table below"],
    ["Sara", 25, "See inner table below"]
]

# Create the inner table
inner_table = [
    ["Hobbies", "Location"],
    ["Reading", "New York"],
    ["Gardening", "San Francisco"]
]

# Convert the inner table to a string
inner_table_string = tabulate(inner_table, headers="firstrow")

# Replace the "See inner table below" cell with the inner table string
for row in outer_table:
    if "See inner table below" in row:
        row[row.index("See inner table below")] = inner_table_string

# Print the outer table
print(tabulate(outer_table, headers="firstrow"))
