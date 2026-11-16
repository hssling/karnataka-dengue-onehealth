"""Extract the annual columns of the KSNDMC Annual Report 2020 district rainfall table
(Table 1.1, 'District/Taluk/Hobli rainfall pattern', district rows; supplied as CSV) and
cross-check them against the report's district summary table (p.3).

The two printed tables of the same report differ by <=2 mm in 5 of 30 districts
(rounding in the source). Values from Table 1.1 (the supplied CSV) are used as-is; using
the summary-table values instead leaves every reported estimate unchanged at the
published precision (checked 16 Nov 2026)."""
import sys, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
# repo: SRC = data/rainfall/raw/ksndmc_2020_district_table.csv; OUT = data/rainfall/ka_dist_rainfall_2020.csv
SRC = r"D:\Dengue project\analysis\submission\ohb_special_issue\revision_R2\f424699b-6fc2-45b5-acdf-42a989ebc592.csv"
OUT = r"D:\Dengue project\analysis\submission\ohb_special_issue\revision_R2\final_response\analysis\data\ka_dist_rainfall_2020.csv"
r = pd.read_csv(SRC, encoding="utf-8-sig", header=None, skiprows=1)
r = r[[1, 11, 12, 13]]; r.columns = ["District", "Annual Normal (mm)", "Annual Actual (mm)", "Annual Departure (%)"]
r["District"] = r["District"].str.strip().str.title()
# District summary table printed on p.3 of the PDF report (Normal, Actual, Departure)
PDF = {"Chitradurga": (540, 769, 42), "Kalaburagi": (770, 1068, 39), "Koppala": (614, 849, 38), "Kolar": (735, 983, 34),
       "Ballari": (614, 814, 33), "Chikkaballapura": (736, 962, 31), "Yadgir": (719, 925, 29), "Bengaluru Urban": (846, 1082, 28),
       "Bagalkote": (582, 739, 27), "Raichur": (654, 826, 26), "Davanagere": (659, 826, 25), "Dharwad": (787, 987, 25),
       "Bidar": (838, 1049, 25), "Vijayapura": (591, 737, 25), "Tumakuru": (669, 828, 24), "Uttara Kannada": (2936, 3538, 20),
       "Belagavi": (826, 994, 20), "Bengaluru Rural": (798, 956, 20), "Gadag": (624, 734, 18), "Mandya": (699, 820, 17),
       "Ramanagara": (840, 959, 14), "Udupi": (4535, 5159, 14), "Chamarajanagara": (787, 883, 12), "Hassan": (1142, 1162, 2),
       "Mysuru": (837, 822, -2), "Dakshina Kannada": (4006, 3917, -2), "Chikkamagaluru": (1833, 1754, -4), "Haveri": (800, 764, -4),
       "Kodagu": (2729, 2541, -7), "Shivamogga": (2325, 2122, -9)}
assert len(r) == 30 and set(r.District) == set(PDF), set(r.District) ^ set(PDF)
r["pdf_actual"] = r.District.map(lambda d: PDF[d][1])
r["diff_mm"] = r["Annual Actual (mm)"] - r["pdf_actual"]
print(r[r.diff_mm != 0][["District", "Annual Actual (mm)", "pdf_actual", "diff_mm"]].to_string(index=False))
print("normals identical:", all(r["Annual Normal (mm)"] == r.District.map(lambda d: PDF[d][0])),
      "| max |actual diff| mm:", r.diff_mm.abs().max())
assert r.diff_mm.abs().max() <= 2
calc = (r["Annual Actual (mm)"] / r["Annual Normal (mm)"] - 1) * 100
print("max |departure - computed| (pp):", round(float((calc - r["Annual Departure (%)"]).abs().max()), 2))
r[["District", "Annual Normal (mm)", "Annual Actual (mm)", "Annual Departure (%)"]].to_csv(OUT, index=False)
print("written", OUT)
