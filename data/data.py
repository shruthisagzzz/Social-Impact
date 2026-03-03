import pandas as pd

# Load the Excel file
df = pd.read_excel("Final DataSet.xlsx", sheet_name="Data")

# Function to balance each group to exactly 100 entries
def balance_company_data(group):
    if len(group) > 100:
        return group.sample(100, random_state=42)
    elif len(group) < 100:
        repeat_factor = 100 // len(group)
        remainder = 100 % len(group)
        # Repeat full group and add sample of remaining
        expanded = pd.concat([group] * repeat_factor + [group.sample(remainder, replace=True, random_state=42)])
        return expanded.reset_index(drop=True)
    else:
        return group.reset_index(drop=True)

# Apply balancing for each company
balanced_df = df.groupby('Company', group_keys=False).apply(balance_company_data)

# Optionally save to new Excel file
balanced_df.to_excel("Balanced_Company_Data.xlsx", index=False)

print("✅ Balanced dataset created with 100 data snippets per company.")
