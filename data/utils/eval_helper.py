# import tabulate

# class Report:

#     def __init__(self, title: str = "Report", headers: list | None = None, tablefmt: str = 'presto', floatfmt: str = ".1f"):
#         if headers is None:
#             self.headers = ["Metric", "Value"]
#         else:
#             self.headers = headers
#         self.title = title
#         self.tablefmt = tablefmt
#         self.data = []
#         self.floatfmt = floatfmt

#     def add_row(self, row: list):
#         if len(row) != len(self.headers):
#             raise ValueError(f"Row length {len(row)} does not match headers length {len(self.headers)}.")
#         self.data.append(row)

#     def __str__(self):
#         return tabulate.tabulate(
#             self.data, headers=self.headers,
#             tablefmt=self.tablefmt, showindex=False,
#             stralign="left", numalign="left", floatfmt=self.floatfmt)

# if __name__ == "__main__":
#     # Example usage of the Report class
#     report = Report(title="Sample Report", headers=["Metric", "Value"])
#     report.add_row(["Accuracy", 0.95])
#     report.add_row(["F1 Score", 0.90])
#     print(report)