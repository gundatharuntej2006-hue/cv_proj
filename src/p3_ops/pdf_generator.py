from pathlib import Path
from tbcore.schemas import PipelineResult


class TriagePDFReportGenerator:
    def generate_report(self, result: PipelineResult, output_path: str = "report/triage_report.pdf") -> str:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(f"%PDF-1.4 Mock Triage Report for {result.image_id}\n")
        return str(out)
