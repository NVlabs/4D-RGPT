import tabulate
from tabulate import _table_formats
from tabulate import Line, DataRow, TableFormat

_custom_format = TableFormat(
    lineabove=Line(begin='+', hline='-', sep='+', end='+'),
    linebelowheader=Line(begin='+', hline='-', sep='+', end='+'),
    linebetweenrows=None,
    linebelow=Line(begin='+', hline='-', sep='+', end='+'),
    headerrow=DataRow(begin=' ', sep='&', end=' '),
    datarow=DataRow(begin=' ', sep='&', end=' '),
    with_header_hide=None,
    padding=1
)

_table_formats["custom"] = _custom_format

# def metrics_report(
#     data: dict[str, float], headers: list[str], floatfmt=".2%"
# ) -> str:
#     row = [data[key] for key in headers]
#     max_width = max(len(str(cell)) for cell in headers)
#     padded_headers = [str(cell).ljust(max_width) for cell in headers]

#     return tabulate(
#         [row], headers=padded_headers, tablefmt="custom", floatfmt=floatfmt,
#         numalign="right"
#     )

class Report:

    def __init__(self, title: str, headers: list[str], tablefmt: str = 'custom', floatfmt: str = ".1f"):
        self.headers = headers
        self.title = title
        self.tablefmt = tablefmt
        self.data = []
        self.floatfmt = floatfmt

    def add_row(self, row: list):
        if len(row) != len(self.headers):
            raise ValueError(f"Row length {len(row)} does not match headers length {len(self.headers)}.")
        self.data.append(row)

    def __str__(self):
        return tabulate.tabulate(
            self.data, headers=self.headers,
            tablefmt=self.tablefmt, showindex=False,
            stralign="left", numalign="left", floatfmt=self.floatfmt)

if __name__ == "__main__":
    # Example usage of the Report class
    report = Report(title="Sample Report", headers=["Metric", "Value"])
    report.add_row(["Accuracy", 0.95])
    report.add_row(["F1 Score", 0.90])
    print(report)